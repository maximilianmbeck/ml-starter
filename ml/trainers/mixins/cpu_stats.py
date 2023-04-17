import atexit
import logging
import multiprocessing as mp
import os
import time
from ctypes import Structure, c_double, c_uint16, c_uint64
from dataclasses import dataclass
from multiprocessing.managers import SyncManager, ValueProxy
from multiprocessing.synchronize import Event
from typing import TypeVar

import psutil
from torch.optim.optimizer import Optimizer

from ml.core.common_types import Batch
from ml.core.config import conf_field
from ml.core.state import State
from ml.lr_schedulers.base import SchedulerAdapter
from ml.trainers.base import ModelT, TaskT
from ml.trainers.mixins.monitor_process import MonitorProcessConfig, MonitorProcessMixin

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class CPUStatsConfig(MonitorProcessConfig):
    cpu_stats_ping_interval: int = conf_field(1, help="How often to check stats (in seconds)")
    cpu_stats_only_log_once: bool = conf_field(False, help="If set, only log read stats one time")


CPUStatsConfigT = TypeVar("CPUStatsConfigT", bound=CPUStatsConfig)


class CPUStats(Structure):
    _fields_ = [
        ("cpu_percent", c_double),
        ("mem_percent", c_double),
        ("mem_rss", c_uint64),
        ("mem_vms", c_uint64),
        ("mem_shared", c_uint64),
        ("mem_rss_total", c_uint64),
        ("mem_vms_total", c_uint64),
        ("child_cpu_percent", c_double),
        ("child_mem_percent", c_double),
        ("num_child_procs", c_uint16),
    ]


@dataclass
class CPUStatsInfo:
    cpu_percent: float
    mem_percent: float
    mem_rss: int
    mem_vms: int
    mem_shared: int
    mem_rss_total: int
    mem_vms_total: int
    child_cpu_percent: float
    child_mem_percent: float
    num_child_procs: int

    @classmethod
    def from_stats(cls, stats: CPUStats) -> "CPUStatsInfo":
        return cls(
            cpu_percent=stats.cpu_percent,
            mem_percent=stats.mem_percent,
            mem_rss=stats.mem_rss,
            mem_vms=stats.mem_vms,
            mem_shared=stats.mem_shared,
            mem_rss_total=stats.mem_rss_total,
            mem_vms_total=stats.mem_vms_total,
            child_cpu_percent=stats.child_cpu_percent,
            child_mem_percent=stats.child_mem_percent,
            num_child_procs=stats.num_child_procs,
        )


def worker(ping_interval: float, stats: ValueProxy[CPUStats], event: Event, pid: int) -> None:
    proc, cur_pid = psutil.Process(pid), os.getpid()
    logger.info("Starting CPU stats monitor for PID %d with PID %d", pid, cur_pid)

    get_children = lambda: {p.pid: p for p in proc.children(recursive=True) if p.pid != cur_pid}
    child_procs = get_children()

    while True:
        try:
            # Updates child processes, preserving the previous child process
            # object. Otherwise the CPU percentage will be zero.
            new_procs = get_children()
            child_procs = {**new_procs, **child_procs}
            child_procs = {pid: child_procs[pid] for pid in new_procs.keys()}

            # Gets process memory info.
            mem_info = proc.memory_info()
            mem_rss_total = sum(p.memory_info().rss for p in child_procs.values()) + mem_info.rss
            mem_vms_total = sum(p.memory_info().vms for p in child_procs.values()) + mem_info.vms

            # Gets child CPU and memory percentages.
            child_cpu_percent_total = sum(p.cpu_percent() for p in child_procs.values()) if child_procs else 0.0
            child_mem_percent_total = sum(p.memory_percent() for p in child_procs.values()) if child_procs else 0.0

            # Sets the CPU stats.
            stats.set(
                CPUStats(
                    cpu_percent=proc.cpu_percent(),
                    mem_percent=proc.memory_percent(),
                    mem_rss=int(mem_info.rss),
                    mem_vms=int(mem_info.vms),
                    mem_shared=int(getattr(mem_info, "shared", 0)),
                    mem_rss_total=int(mem_rss_total),
                    mem_vms_total=int(mem_vms_total),
                    child_cpu_percent=child_cpu_percent_total / len(child_procs),
                    child_mem_percent=child_mem_percent_total / len(child_procs),
                    num_child_procs=len(child_procs),
                ),
            )

            event.set()

        except psutil.NoSuchProcess:
            logger.info("No parent process; probably cleaning up")

        time.sleep(ping_interval)


class CPUStatsMonitor:
    def __init__(self, ping_interval: float, manager: SyncManager) -> None:
        self._manager = manager
        self._event = manager.Event()
        self._cpu_stats_smem = self._manager.Value(
            CPUStats,
            CPUStats(
                cpu_percent=0.0,
                mem_percent=0.0,
                mem_rss=0,
                mem_vms=0,
                mem_shared=0,
                mem_rss_total=0,
                mem_vms_total=0,
                child_cpu_percent=0.0,
                child_mem_percent=0.0,
                num_child_procs=0,
            ),
        )
        self._cpu_stats: CPUStatsInfo | None = None

        self._proc = mp.Process(
            target=worker,
            args=(ping_interval, self._cpu_stats_smem, self._event, os.getpid()),
            daemon=False,
        )
        self._proc.start()
        atexit.register(self.stop)

    def get_if_set(self) -> CPUStatsInfo | None:
        if self._event.is_set():
            self._event.clear()
            return CPUStatsInfo.from_stats(self._cpu_stats_smem.get())
        return None

    def get(self) -> CPUStatsInfo | None:
        if (stats := self.get_if_set()) is not None:
            self._cpu_stats = stats
        return self._cpu_stats

    def stop(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()
            logger.debug("Terminated CPU stats monitor; joining...")
            self._proc.join()


class CPUStatsMixin(MonitorProcessMixin[CPUStatsConfigT, ModelT, TaskT]):
    """Defines a trainer mixin for getting CPU statistics."""

    def __init__(self, config: CPUStatsConfigT) -> None:
        super().__init__(config)

        self._cpu_stats_monitor = CPUStatsMonitor(self.config.cpu_stats_ping_interval, self._mp_manager)

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

        monitor = self._cpu_stats_monitor
        stats = monitor.get_if_set() if self.config.cpu_stats_only_log_once else monitor.get()

        if stats is not None:
            self.logger.log_scalar("cpu/percent", stats.cpu_percent, namespace="trainer")
            self.logger.log_scalar("cpu/child_percent", stats.child_cpu_percent, namespace="trainer")
            self.logger.log_scalar("mem/percent", stats.mem_percent, namespace="trainer")
            self.logger.log_scalar("mem/rss", stats.mem_rss, namespace="trainer")
            self.logger.log_scalar("mem/vms", stats.mem_vms, namespace="trainer")
            self.logger.log_scalar("mem/shared", stats.mem_shared, namespace="trainer")
            self.logger.log_scalar("mem/rss/total", stats.mem_rss_total, namespace="trainer")
            self.logger.log_scalar("mem/vms/total", stats.mem_vms_total, namespace="trainer")
            self.logger.log_scalar("mem/child_percent", stats.child_mem_percent, namespace="trainer")
            self.logger.log_scalar("child_procs", stats.num_child_procs, namespace="trainer")
