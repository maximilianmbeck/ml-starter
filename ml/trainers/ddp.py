"""Defines a Distributed Data Parallel trainer.

This is a light-weight wrapper around PyTorch's built-in Distributed Data
Parallel class.

For multiple devices, data is split along the batch dimension, passed to each
device, which computes losses independently. The loss tensors are gathered to
the master device to compute a single loss.

Currently this trainer doesn't do anything different from the vanilla trainer
besides warning when there is more than one GPU. It will be implemented once
we have a cluster that necessitates it.

Summary table:

|         | device 1    | device 2    | ... | device N       |
|---------|-------------|-------------|-----|----------------|
| data    | data[0::N]  | data[1::N]  | ... | data[N - 1::N] |
| step    | model(x_1)  | model(x_2)  | ... | model(x_N)     |
| loss    | E(x_1, o_1) | E(x_2, o_2) | ... | E(x_N, o_N)    |
"""

import logging
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch.multiprocessing as mp
from torch import nn

from ml.core.config import conf_field
from ml.core.registry import register_trainer
from ml.models.base import BaseModel
from ml.tasks.base import BaseTask
from ml.trainers.base import MultiprocessConfig
from ml.trainers.vanilla import VanillaTrainer, VanillaTrainerConfig
from ml.utils.distributed import get_world_size
from ml.utils.networking import get_unused_port

logger = logging.getLogger(__name__)


@dataclass
class MultiprocessTrainingConfig:
    devices_per_rank: int = conf_field(1, help="Number of devices on each rank")


@dataclass
class DDPTrainerConfig(VanillaTrainerConfig):
    multiprocess: MultiprocessTrainingConfig = MultiprocessTrainingConfig()


DDPTrainerConfigType = TypeVar("DDPTrainerConfigType", bound=DDPTrainerConfig)  # pylint: disable=invalid-name


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


@register_trainer("ddp", DDPTrainerConfig)
class DDPTrainer(VanillaTrainer[DDPTrainerConfigType]):
    def get_task_model(self, task: BaseTask, model: BaseModel) -> nn.Module:
        task_model = super().get_task_model(task, model)
        if get_world_size() > 1:
            task_model = nn.parallel.DistributedDataParallel(task_model)
        return task_model

    def launch(self, func: Callable[[MultiprocessConfig], None]) -> None:
        device_count = self.device.device_count()
        devices_per_rank = self.config.multiprocess.devices_per_rank

        if device_count % devices_per_rank != 0:
            raise ValueError(f"Can't evenly split {device_count=} so that each rank has {devices_per_rank}")

        cfg = MultiprocessConfig(
            rank=-1,
            world_size=device_count // devices_per_rank,
            devices_per_rank=devices_per_rank,
            master_addr="localhost",
            master_port=get_unused_port(),
        )

        if device_count <= 1:
            logger.warning("Multi-process DDPTrainer expects more than one device")
            cfg.rank = 0
            func(cfg)
            return

        def set_env(rank: int) -> None:
            start_device_id = rank * cfg.devices_per_rank
            device_ids = [i + start_device_id for i in range(cfg.devices_per_rank)]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)

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
