"""Defines a linear warmup scheduler without decay."""

from dataclasses import dataclass

from ml.core.config import conf_field
from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class LinearNoDecayLRSchedulerConfig(BaseLRSchedulerConfig):
    warmup_steps: int = conf_field(1000, help="Number of warmup steps")


@register_lr_scheduler("linear_no_decay", LinearNoDecayLRSchedulerConfig)
class LinearNoDecayLRScheduler(BaseLRScheduler[LinearNoDecayLRSchedulerConfig]):
    def get_lr_scale(self, state: State) -> float:
        if state.num_steps < self.config.warmup_steps:
            return state.num_steps / self.config.warmup_steps
        return 1.0
