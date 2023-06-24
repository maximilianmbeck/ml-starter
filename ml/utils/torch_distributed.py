"""Defines utilities for training distributed PyTorch models.

The canonical way to use this library is to call :func:`launch_subprocess`
from the main function of your script. This will launch a subprocess for each
device and initialize the process group for distributed training. You can
modify the number of processes and the backend by changing the provided config.
"""

import functools
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import MISSING

from ml.core.config import conf_field
from ml.utils.checkpoint import is_missing
from ml.utils.distributed import get_init_method, get_rank, get_world_size, set_dist
from ml.utils.logging import INFOALL, configure_logging
from ml.utils.networking import get_unused_port
from ml.utils.parallel import init_parallelism

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MultiprocessConfig:
    rank: int = conf_field(-1, help="The rank of the process")
    local_rank: int = conf_field(-1, help="The local rank of the process")
    world_size: int = conf_field(MISSING, help="The total number of processes")
    local_world_size: int = conf_field(MISSING, help="The number of processes per machine")
    master_addr: str = conf_field("127.0.0.1", help="The address of the master process")
    master_port: int = conf_field(MISSING, help="The port of the master process")
    init_method: str = conf_field("env://", help="The initialization method")
    model_parallelism: int = conf_field(1, help="The number of model parallel processes")
    pipeline_parallelism: int = conf_field(1, help="The number of pipeline parallel processes")
    backend: str | None = conf_field(None, help="The distributed backend")
    model_parallel_backend: str | None = conf_field(None, help="The model parallel backend")
    pipeline_parallel_backend: str | None = conf_field(None, help="The pipeline parallel backend")
    data_parallel_backend: str | None = conf_field(None, help="The data parallel backend")
    launch_method: str = conf_field("forkserver", help="The launch method for multiprocessing")

    @classmethod
    def resolve(cls, config: "MultiprocessConfig") -> None:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if is_missing(config, "world_size"):
            config.world_size = device_count
        if is_missing(config, "local_world_size"):
            config.local_world_size = min(device_count, config.world_size)
        if is_missing(config, "master_port"):
            config.master_port = get_unused_port()


def init_process_group_from_backend(backend: str | dist.Backend | None = None) -> None:
    if backend is None:
        backend = get_distributed_backend()
    init_method, world_size, rank = get_init_method(), get_world_size(), get_rank()

    logger.log(INFOALL, "Initializing %d / %d using %s - %s", rank, world_size, init_method, backend)
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % device_count)

    logger.info("Initialized process group; running dummy all-reduce")
    dist.all_reduce(torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Dummy all-reduce succeeded")


def init_dist(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    master_addr: str,
    master_port: int,
    init_method: str,
    backend: str | dist.Backend | None = None,
) -> None:
    """Initializes distributed environment.

    Args:
        rank: The rank of the current process.
        local_rank: The local rank of the current process.
        world_size: The total number of processes.
        local_world_size: The number of processes per machine.
        master_addr: The address of the master process.
        master_port: The port of the master process.
        init_method: The initialization method.
        backend: The distributed backend.
    """
    set_dist(rank, local_rank, world_size, local_world_size, master_addr, master_port, init_method)
    init_process_group_from_backend(backend)


@functools.lru_cache
def default_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def get_distributed_backend() -> dist.Backend:
    # Used to change the distributed backend to something other than NCCL.
    # For example, if you're on a system with some strange NCCL errors, you
    # can try changing this environment variable to `gloo`.
    return dist.Backend(os.environ.get("TORCH_DISTRIBUTED_BACKEND", default_backend()))


def set_distributed_backend(backend: str) -> None:
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = backend


def init_and_run(func: Callable[[], None], cfg: MultiprocessConfig) -> None:
    configure_logging(rank=cfg.rank, world_size=cfg.world_size)

    init_dist(
        rank=cfg.rank,
        local_rank=cfg.local_rank,
        world_size=cfg.world_size,
        local_world_size=cfg.local_world_size,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        init_method=cfg.init_method,
        backend=cfg.backend,
    )

    init_parallelism(
        model_parallelism=cfg.model_parallelism,
        pipeline_parallelism=cfg.pipeline_parallelism,
        mp_backend=cfg.backend if cfg.model_parallel_backend is None else cfg.model_parallel_backend,
        pp_backend=cfg.backend if cfg.pipeline_parallel_backend is None else cfg.pipeline_parallel_backend,
        dp_backend=cfg.backend if cfg.data_parallel_backend is None else cfg.data_parallel_backend,
    )

    func()


def _func_wrapped(
    func: Callable[[], None],
    setup: Callable[[], None] | None,
    cfg: MultiprocessConfig,
    error_queue: "mp.SimpleQueue[str | None]",
) -> None:
    try:
        if setup is not None:
            setup()

        init_and_run(func, cfg)

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt; exiting")

    except Exception:
        error_queue.put(traceback.format_exc())
        sys.exit(1)

    error_queue.put(None)


def launch_subprocesses(
    func: Callable[[], None],
    cfg: MultiprocessConfig,
    setup: Callable[[], None] | None = None,
) -> None:
    """Launches a function in multiple subprocesses.

    Args:
        func: The function to launch.
        cfg: The configuration for the function.
        setup: A function to run before launching the subprocesses.

    Raises:
        RuntimeError: If the function fails in any subprocess.
    """
    MultiprocessConfig.resolve(cfg)

    if cfg.world_size <= 1:
        logger.warning("Multi-process trainer expects more than one device; running single-process")
        cfg.rank = 0
        init_and_run(func, cfg)
        return

    logger.info("Launching %d training workers", cfg.world_size)
    ctx = mp.get_context(cfg.launch_method)
    error_queues: list["mp.SimpleQueue[str | None]"] = []
    procs = []
    for rank in range(cfg.world_size):
        error_queue = ctx.SimpleQueue()
        cfg.rank = rank
        cfg.local_rank = rank % cfg.local_world_size
        proc = ctx.Process(
            target=_func_wrapped,
            args=(func, setup, cfg, error_queue),
            daemon=False,
            name=f"worker-{rank}",
        )
        logger.debug("Started process %d", rank)
        proc.start()
        error_queues.append(error_queue)
        procs.append(proc)

    pctx = mp.ProcessContext(procs, error_queues)
    while not pctx.join():
        pass

    for rank, error_queue in enumerate(error_queues):
        error = error_queue.get()
        if error:
            raise RuntimeError(f"Process {rank} failed with error:\n{error}")
