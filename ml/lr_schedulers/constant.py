"""Defines a constant learning rate scheduler."""

from dataclasses import dataclass

from ml.core.config import conf_field
from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class ConstantLRSchedulerConfig(BaseLRSchedulerConfig):
    factor: float = conf_field(1.0, help="The learning rate scale factor")


@register_lr_scheduler("constant", ConstantLRSchedulerConfig)
class ConstantLRScheduler(BaseLRScheduler[ConstantLRSchedulerConfig]):
    def get_lr_scale(self, state: State) -> float:
        return self.config.factor
