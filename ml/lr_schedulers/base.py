from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch.optim.optimizer import Optimizer

from ml.core.config import BaseConfig, BaseObjectWithPointers
from ml.core.state import State


class SchedulerAdapter:
    """Defines a general-purpose learning rate scheduler adapter."""

    last_state: State | None

    def __init__(self, scheduler: "BaseLRScheduler", optimizer: Optimizer) -> None:
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.last_state = None

        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]

        self.lr_scale = 0.0

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def step(self, state: State) -> None:
        self.last_state = state
        self.lr_scale = self.scheduler.get_lr_scale(state)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * self.lr_scale


@dataclass
class BaseLRSchedulerConfig(BaseConfig):
    """Defines the base config for all learning rate schedulers."""


LRSchedulerConfigT = TypeVar("LRSchedulerConfigT", bound=BaseLRSchedulerConfig)


class BaseLRScheduler(BaseObjectWithPointers[LRSchedulerConfigT], Generic[LRSchedulerConfigT], ABC):
    """Defines the base learning rate scheduler."""

    @abstractmethod
    def get_lr_scale(self, state: State) -> float:
        """Given a state, returns the current learning rate.

        Args:
            state: The current trainer state

        Returns:
            The computed learning rate to use
        """

    def get(self, optimizer: Optimizer) -> SchedulerAdapter:
        return SchedulerAdapter(self, optimizer)
