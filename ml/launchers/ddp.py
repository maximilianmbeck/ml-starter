"""Defines a Distributed Data Parallel trainer.

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
from omegaconf import DictConfig

from ml.core.registry import Objects, register_launcher
from ml.launchers.base import BaseLauncher, BaseLauncherConfig
from ml.scripts.train import train_main
from ml.trainers.base import MultiprocessConfig
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
    train_main(objs)


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
    pass


@register_launcher("ddp", DDPLauncherConfig)
class DDPLauncher(BaseLauncher[DDPLauncherConfig]):
    def launch(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DDPLauncher requires CUDA")
        device_count = torch.cuda.device_count()

        func = functools.partial(process_main, raw_config=self.raw_config)

        cfg = MultiprocessConfig(
            rank=-1,
            world_size=device_count,
            devices_per_rank=1,
            master_addr="localhost",
            master_port=get_unused_port(),
        )

        if device_count <= 1:
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
            proc = ctx.Process(
                target=func_wrapped,
                args=(func, cfg, error_queue),
                daemon=False,
            )
            logger.debug("Started process %d", rank)
            proc.start()
            error_queues.append(error_queue)
            procs.append(proc)
        pctx = mp.ProcessContext(procs, error_queues)
        while not pctx.join():
            pass
