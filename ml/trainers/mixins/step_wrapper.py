"""Defines a mixin to wrap some steps in a context manager.

This is used by other components which want to know when a step is being
run, such as when doing profiling.
"""

from abc import ABC
from types import TracebackType
from typing import ContextManager, Literal, TypeVar

from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT

StepType = Literal[
    "backward",
    "build_rl_dataset",
    "change_mode",
    "clip_grads",
    "collect_rl_samples",
    "forward",
    "get_single_loss",
    "log_losses",
    "on_epoch_end",
    "on_epoch_start",
    "on_step_end",
    "on_step_start",
    "step",
    "update_state",
    "write_logs",
    "zero_grads",
]


BaseTrainerConfigT = TypeVar("BaseTrainerConfigT", bound=BaseTrainerConfig)


class StepContext(ContextManager):
    """Context manager to get the current step type."""

    CURRENT_STEP: StepType | None = None

    def __init__(self, step: StepType) -> None:
        self.step = step

    def __enter__(self) -> None:
        StepContext.CURRENT_STEP = self.step

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        StepContext.CURRENT_STEP = None


class StepContextMixin(BaseTrainer[BaseTrainerConfigT, ModelT, TaskT], ABC):
    def step_context(self, step: StepType) -> ContextManager:
        return StepContext(step)
