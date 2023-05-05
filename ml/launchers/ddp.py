"""Defines a Distributed Data Parallel launcher.

This is a light-weight wrapper around PyTorch's built-in Distributed Data
Parallel class.

For multiple devices, data is split along the batch dimension, passed to each
device, which computes losses independently. The loss tensors are gathered to
the master device to compute a single loss. In other words, each device
belongs to exactly one process.
"""

import functools
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

import torch
import torch.multiprocessing as mp
from omegaconf import MISSING, DictConfig, OmegaConf

from ml.core.config import conf_field
from ml.core.registry import Objects, register_launcher
from ml.launchers.base import BaseLauncher, BaseLauncherConfig
from ml.scripts.train import train_main_with_objects
from ml.utils.distributed import (
    set_init_method,
    set_master_addr,
    set_master_port,
    set_rank,
    set_world_size,
)
from ml.utils.logging import configure_logging
from ml.utils.networking import get_unused_port
from ml.utils.torch_distributed import get_distributed_backend, init_process_group

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MultiprocessConfig:
    rank: int = conf_field(-1, help="The rank of the process")
    world_size: int = conf_field(MISSING, help="The total number of processes")
    devices_per_rank: int = conf_field(1, help="The number of devices per rank")
    master_addr: str = conf_field("localhost", help="The address of the master process")
    master_port: int = conf_field(MISSING, help="The port of the master process")


def process_main(cfg: MultiprocessConfig, raw_config: DictConfig) -> None:
    set_master_addr(cfg.master_addr)
    set_master_port(cfg.master_port)
    set_rank(cfg.rank)
    set_world_size(cfg.world_size)
    set_init_method("env://")
    configure_logging(rank=cfg.rank, world_size=cfg.world_size)
    logger.info("Initializing process group")
    init_process_group(backend=get_distributed_backend())

    objs = Objects.parse_raw_config(raw_config)
    train_main_with_objects(objs)


def func_wrapped(
    func: Callable[[MultiprocessConfig], None],
    cfg: MultiprocessConfig,
    error_queue: "mp.Queue[str]",
) -> None:
    try:
        func(cfg)
    except KeyboardInterrupt:
        pass
    except Exception:
        error_queue.put(traceback.format_exc())
        sys.exit(1)


@dataclass
class DDPLauncherConfig(BaseLauncherConfig):
    multiprocess: MultiprocessConfig = conf_field(MultiprocessConfig())

    @classmethod
    def resolve(cls: type["DDPLauncherConfig"], config: "DDPLauncherConfig") -> None:
        super().resolve(config)

        device_count = torch.cuda.device_count()
        if config.multiprocess.devices_per_rank > device_count:
            raise ValueError(
                f"Requested {config.multiprocess.devices_per_rank} devices per rank, "
                f"but only {device_count} are available"
            )
        if OmegaConf.is_missing(config.multiprocess, "world_size"):
            config.multiprocess.world_size = device_count // config.multiprocess.devices_per_rank
        if OmegaConf.is_missing(config.multiprocess, "master_port"):
            config.multiprocess.master_port = get_unused_port()


@register_launcher("ddp", DDPLauncherConfig)
class DDPLauncher(BaseLauncher[DDPLauncherConfig]):
    def launch(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DDPLauncher requires CUDA")

        func = functools.partial(process_main, raw_config=self.raw_config)

        # Config should have valid values at this point, post-resolution.
        cfg = self.config.multiprocess

        if cfg.world_size <= 1:
            logger.warning("Multi-process DDPTrainer expects more than one device")
            cfg.rank = 0
            func(cfg)
            return

        def set_env(rank: int) -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        # This is essentially the same as `mp.spawn` but with specific control
        # over CUDA_VISIBLE_DEVICES.
        logger.info("Launching %d training workers", cfg.world_size)
        ctx = mp.get_context("spawn")
        error_queues = []
        procs = []
        for rank in range(cfg.world_size):
            error_queue = ctx.SimpleQueue()
            cfg.rank = rank
            set_env(rank)
            proc = ctx.Process(target=func_wrapped, args=(func, cfg, error_queue), daemon=False)
            logger.debug("Started process %d", rank)
            proc.start()
            error_queues.append(error_queue)
            procs.append(proc)
        pctx = mp.ProcessContext(procs, error_queues)
        while not pctx.join():
            pass
