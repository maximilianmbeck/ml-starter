import functools
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
from ctypes import Structure, c_double, c_uint32
from dataclasses import dataclass
from multiprocessing.managers import SyncManager, ValueProxy
from multiprocessing.synchronize import Event
from typing import Iterable, Pattern, TypeVar

from torch.optim.optimizer import Optimizer

from ml.core.common_types import Batch
from ml.core.config import conf_field
from ml.core.state import State
from ml.lr_schedulers.base import SchedulerAdapter
from ml.trainers.base import ModelT, TaskT
from ml.trainers.mixins.monitor_process import MonitorProcessConfig, MonitorProcessMixin

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class GPUStatsConfig(MonitorProcessConfig):
    gpu_stats_ping_interval: int = conf_field(10, help="How often to check stats (in seconds)")
    gpu_stats_only_log_once: bool = conf_field(False, help="If set, only log read stats one time")


GPUStatsConfigT = TypeVar("GPUStatsConfigT", bound=GPUStatsConfig)

NUMBER_REGEX: Pattern[str] = re.compile(r"[\d\.]+")


class GPUStats(Structure):
    _fields_ = [
        ("index", c_uint32),
        ("memory_used", c_double),
        ("temperature", c_double),
        ("utilization", c_double),
    ]


@dataclass(frozen=True)
class GPUStatsInfo:
    index: int
    memory_used: float
    temperature: float
    utilization: float

    @classmethod
    def from_stats(cls, stats: GPUStats) -> "GPUStatsInfo":
        return cls(
            index=stats.index,
            memory_used=stats.memory_used,
            temperature=stats.temperature,
            utilization=stats.utilization,
        )


@functools.lru_cache
def get_num_gpus() -> int:
    command = "nvidia-smi --query-gpu=index --format=csv --format=csv,noheader"

    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")
            return len(list(rows))

    except Exception:
        logger.exception("Caught exception while trying to query `nvidia-smi`")
        return 0


def parse_number(s: str) -> float:
    match = NUMBER_REGEX.search(s)
    if match is None:
        raise ValueError(s)
    return float(match.group())


def parse_gpu_stats(row: str) -> GPUStats:
    cols = row.split(",")
    index = int(cols[0].strip())
    memory_total, memory_used, temperature, utilization = (parse_number(col) for col in cols[1:])

    return GPUStats(
        index=index,
        memory_used=100 * memory_used / memory_total,
        temperature=temperature,
        utilization=utilization,
    )


def gen_gpu_stats(loop_secs: int = 5) -> Iterable[GPUStats]:
    fields = ",".join(["index", "memory.total", "memory.used", "temperature.gpu", "utilization.gpu"])
    command = f"nvidia-smi --query-gpu={fields} --format=csv,noheader --loop={loop_secs}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_device_ids = None if visible_devices is None else {int(i.strip()) for i in visible_devices.split(",")}
    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")
            for row in rows:
                try:
                    stats = parse_gpu_stats(row)
                except ValueError:
                    continue
                if visible_device_ids is None or stats.index in visible_device_ids:
                    yield stats

    except subprocess.CalledProcessError:
        logger.exception("Caught exception while trying to query `nvidia-smi`")


def worker(
    ping_interval: int,
    smems: list[ValueProxy[GPUStats]],
    main_event: Event,
    events: list[Event],
    start_event: Event,
) -> None:
    start_event.set()

    logger.debug("Starting GPU stats monitor with PID %d", os.getpid())

    for gpu_stat in gen_gpu_stats(ping_interval):
        if gpu_stat.index >= len(smems):
            logger.warning("GPU index %d is out of range", gpu_stat.index)
            continue
        smems[gpu_stat.index].set(gpu_stat)
        events[gpu_stat.index].set()
        main_event.set()


class GPUStatsMonitor:
    def __init__(self, ping_interval: float, manager: SyncManager) -> None:
        self._ping_interval = ping_interval
        self._manager = manager

        num_gpus = get_num_gpus()
        self._main_event = manager.Event()
        self._events = [manager.Event() for _ in range(num_gpus)]
        self._start_event = manager.Event()

        self._smems = [
            manager.Value(
                GPUStats,
                GPUStats(
                    index=i,
                    memory_used=0.0,
                    temperature=0.0,
                    utilization=0.0,
                ),
            )
            for i in range(num_gpus)
        ]
        self._gpu_stats: dict[int, GPUStatsInfo] = {}
        self._proc: mp.Process | None = None

    def get_if_set(self) -> dict[int, GPUStatsInfo]:
        gpu_stats: dict[int, GPUStatsInfo] = {}
        if self._main_event.is_set():
            self._main_event.clear()
            for i, event in enumerate(self._events):
                if event.is_set():
                    event.clear()
                    gpu_stats[i] = GPUStatsInfo.from_stats(self._smems[i].get())
        return gpu_stats

    def get(self) -> dict[int, GPUStatsInfo]:
        self._gpu_stats.update(self.get_if_set())
        return self._gpu_stats

    def start(self, wait: bool = False) -> None:
        if self._proc is not None:
            raise RuntimeError("GPUStatsMonitor already started")
        if self._main_event.is_set():
            self._main_event.clear()
        for event in self._events:
            if event.is_set():
                event.clear()
        if self._start_event.is_set():
            self._start_event.clear()
        self._gpu_stats.clear()
        self._proc = mp.Process(
            target=worker,
            args=(self._ping_interval, self._smems, self._main_event, self._events, self._start_event),
            daemon=False,
        )
        self._proc.start()
        if wait:
            self._start_event.wait()

    def stop(self) -> None:
        if self._proc is None:
            raise RuntimeError("GPUStatsMonitor not started")
        if self._proc.is_alive():
            self._proc.terminate()
            logger.debug("Terminated GPU stats monitor; joining...")
            self._proc.join()
        self._proc = None


class GPUStatsMixin(MonitorProcessMixin[GPUStatsConfigT, ModelT, TaskT]):
    """Defines a trainer mixin for getting GPU statistics."""

    def __init__(self, config: GPUStatsConfigT) -> None:
        super().__init__(config)

        self._gpu_stats_monitor: GPUStatsMonitor | None = None
        if shutil.which("nvidia-smi") is not None:
            self._gpu_stats_monitor = GPUStatsMonitor(config.gpu_stats_ping_interval, self._mp_manager)

    def on_training_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        super().on_training_start(state, task, model, optim, lr_sched)

        if self._gpu_stats_monitor is not None:
            self._gpu_stats_monitor.start()

    def on_training_end(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        super().on_training_end(state, task, model, optim, lr_sched)

        if self._gpu_stats_monitor is not None:
            self._gpu_stats_monitor.stop()

    def on_step_start(
        self,
        state: State,
        train_batch: Batch,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        super().on_step_start(state, train_batch, task, model, optim, lr_sched)

        if (monitor := self._gpu_stats_monitor) is None:
            return

        stats = monitor.get_if_set() if self.config.gpu_stats_only_log_once else monitor.get()

        for gpu_stat in stats.values():
            if gpu_stat is None:
                continue
            self.logger.log_scalar(f"gpu/{gpu_stat.index}/mem_used", gpu_stat.memory_used, namespace="trainer")
            self.logger.log_scalar(f"gpu/{gpu_stat.index}/temp", gpu_stat.temperature, namespace="trainer")
            self.logger.log_scalar(f"gpu/{gpu_stat.index}/gpu_util", gpu_stat.utilization, namespace="trainer")
