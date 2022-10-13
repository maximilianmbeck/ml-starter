from dataclasses import dataclass

from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class ConstantLRSchedulerConfig(BaseLRSchedulerConfig):
    pass


@register_lr_scheduler("constant", ConstantLRSchedulerConfig)
class ConstantLRScheduler(BaseLRScheduler[ConstantLRSchedulerConfig]):
    def get_lr_scale(self, state: State) -> float:
        return 1.0
