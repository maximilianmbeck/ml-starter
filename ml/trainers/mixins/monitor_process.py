"""Defines a base trainer mixin for handling subprocess monitoring jobs."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch.optim.optimizer import Optimizer

from ml.core.state import State
from ml.lr_schedulers.base import SchedulerAdapter
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MonitorProcessConfig(BaseTrainerConfig):
    pass


MonitorProcessConfigT = TypeVar("MonitorProcessConfigT", bound=MonitorProcessConfig)


class MonitorProcessMixin(
    BaseTrainer[MonitorProcessConfigT, ModelT, TaskT],
    Generic[MonitorProcessConfigT, ModelT, TaskT],
):
    """Defines a base trainer mixin for handling monitoring processes."""

    def __init__(self, config: MonitorProcessConfigT) -> None:
        super().__init__(config)

        self._mp_manager = mp.Manager()

    def on_training_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        super().on_training_start(state, task, model, optim, lr_sched)

        self._mp_manager = mp.Manager()

    def on_training_end(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        super().on_training_end(state, task, model, optim, lr_sched)

        self._mp_manager.shutdown()
