"""Defines a launcher for multiprocess training.

This can be used with distributed data parallel (DDP) or fully sharded data
parallel (FSDP) training. The launcher will spawn a process for each device
and initialize the process group for DDP or FSDP training.

This launcher expects to run on a single machine with one or more GPUs.
"""

import functools
import logging
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from ml.core.config import conf_field
from ml.core.registry import Objects, register_launcher
from ml.launchers.base import BaseLauncher, BaseLauncherConfig
from ml.scripts.train import train_main_with_objects
from ml.utils.distributed import set_dist
from ml.utils.logging import configure_logging
from ml.utils.torch_distributed import MultiprocessConfig, init_process_group_from_backend, launch_subprocesses

logger: logging.Logger = logging.getLogger(__name__)


def process_main(cfg: MultiprocessConfig, raw_config: DictConfig) -> None:
    set_dist(cfg.rank, cfg.world_size, cfg.master_addr, cfg.master_port, "env://")
    configure_logging(rank=cfg.rank, world_size=cfg.world_size)
    logger.info("Initializing process group")
    init_process_group_from_backend()

    objs = Objects.parse_raw_config(raw_config)
    train_main_with_objects(objs)


@dataclass
class MultiProcessLauncherConfig(BaseLauncherConfig):
    multiprocess: MultiprocessConfig = conf_field(MultiprocessConfig())

    @classmethod
    def resolve(cls: type["MultiProcessLauncherConfig"], config: "MultiProcessLauncherConfig") -> None:
        super().resolve(config)

        # Resolve multiprocess config.
        config.multiprocess.resolve()


@register_launcher("mp", MultiProcessLauncherConfig)
class MultiProcessLauncher(BaseLauncher[MultiProcessLauncherConfig]):
    def launch(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MultiProcessLauncher requires CUDA")

        func = functools.partial(process_main, raw_config=self.raw_config)
        launch_subprocesses(func, self.config.multiprocess)
