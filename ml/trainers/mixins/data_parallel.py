"""Defines a trainer mixin for data and model parallelism.

This defines how to wrap the model when launching multi-GPU or multi-node jobs.
There are two wrappers:

- ``DistributedDataParallel`` (DDP)
- ``FullyShardedDataParallel`` (FSDP)

DDP is the default wrapper unless ``conf.parallel.use_fsdp`` is set to ``True``.
DDO runs each model replica on a single GPU processing a subset of the batch,
and then synchronizes gradients across all GPUs. FSDP supports more complex
sharding of the model across GPUs and nodes, and also supports CPU offloading.
"""

import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from ml.core.common_types import Batch, Loss
from ml.core.config import conf_field
from ml.core.state import State
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT
from ml.utils.distributed import get_world_size
from ml.utils.parallel import parallel_group_info

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T", bound=nn.Module)


class TaskModel(nn.Module, Generic[ModelT, TaskT, Batch, Loss]):
    def __init__(self, task: TaskT, model: ModelT) -> None:
        super().__init__()

        self.task = task
        self.model = model

    def forward(self, batch: Batch, state: State) -> Loss:
        self.task.on_before_forward_step(self.model, batch, state)
        output = self.task.run_model(self.model, batch, state)
        self.task.on_after_forward_step(self.model, batch, output, state)
        loss: Loss = self.task.compute_loss(self.model, batch, state, output)
        self.task.on_after_compute_loss(self.model, batch, output, loss, state)
        return loss


@dataclass
class ParallelConfig:
    use_fsdp: bool = conf_field(False, help="If set, use FSDP; otherwise, use DDP")
    cpu_offload: bool = conf_field(False, help="CPU offloading for FSDP")
    sharding_strategy: ShardingStrategy = conf_field(ShardingStrategy.HYBRID_SHARD, help="Sharding strategy")
    sync_module_states: bool = conf_field(True, help="Whether to sync module states on initialization")


def ddp(model: nn.Module, cfg: ParallelConfig) -> DDP:
    group_info = parallel_group_info()

    return DDP(model, process_group=group_info.dp.group)


def _all_params_are_cuda(model: nn.Module) -> bool:
    return all(p.is_cuda for p in model.parameters())


def fsdp(model: nn.Module, cfg: ParallelConfig) -> FSDP:
    group_info = parallel_group_info()

    process_group: tuple[ProcessGroup, ProcessGroup] | ProcessGroup
    if cfg.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        process_group = group_info.mp.group, group_info.dp.group
    else:
        process_group = group_info.mp.group

    if cfg.cpu_offload:
        logger.warning("CPU offloading doesn't support gradient accumulation")

    return FSDP(
        model,
        process_group=process_group,
        sharding_strategy=cfg.sharding_strategy,
        sync_module_states=cfg.sync_module_states and _all_params_are_cuda(model),
        cpu_offload=CPUOffload(cfg.cpu_offload),
    )


def dp(model: T, cfg: ParallelConfig) -> T | DDP | FSDP:
    """Wraps a model for data parallel training, if necessary.

    Args:
        model: The model to wrap.
        cfg: The model configuration.

    Returns:
        The wrapped model.
    """
    if get_world_size() <= 1:
        return model
    return fsdp(model, cfg) if cfg.use_fsdp else ddp(model, cfg)


@dataclass
class TrainerParallelConfig(BaseTrainerConfig):
    parallel: ParallelConfig = conf_field(ParallelConfig(), help="Parallelism configuration options")


ParallelConfigT = TypeVar("ParallelConfigT", bound=TrainerParallelConfig)


class ParallelMixin(BaseTrainer[ParallelConfigT, ModelT, TaskT]):
    """Defines a trainer mixin for fully sharded data parallel models."""

    def _get_task_model(self, task: TaskT, model: ModelT) -> nn.Module:
        device, dtype = self._device.get_device(), self._weight_precision
        model.init(device, dtype)
        task.to(device, dtype)
        task_model: nn.Module = TaskModel(task=task, model=model)
        task_model = dp(task_model, self.config.parallel)
        return task_model
