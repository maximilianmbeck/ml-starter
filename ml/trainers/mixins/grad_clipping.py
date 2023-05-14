"""Defines a trainer mixin for supporting gradient clipping.

Gradient clipping occurs after the gradients are computed and before the
optimizer step. It is done in-place, so the gradients are modified. There
are basically three types of gradient clipping:

1. Norm-based clipping
2. Value-based clipping
3. Global norm-based clipping

Norm-based clipping is the most common type of gradient clipping. It
clips the norm of each gradient to a maximum value, by dividing by the norm
if the norm is greater than some threshold.

Value-based clipping clips each gradient value to a maximum value, by
clamping the gradient value to the maximum value.

Global norm-based clipping clips the norm of all gradients to a maximum
value, by dividing all gradients by the total norm if the total norm is
greater than some threshold.
"""

from dataclasses import dataclass
from typing import Any, TypeVar

from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT
from ml.trainers.mixins.mixed_precision import (
    MixedPrecisionTrainerConfig,
    MixedPrecisionTrainerMixin,
)


@dataclass
class GradientClipping:
    clip_grad_norm: float | None = conf_field(None, help="What to clip the gradient norm to")
    clip_grad_value: float | None = conf_field(None, help="What to clip the gradient value to")
    norm_type: Any = conf_field(2, help="Type of norm to use")


@dataclass
class GradientClippingConfig(MixedPrecisionTrainerConfig, BaseTrainerConfig):
    grad_clipping: GradientClipping = conf_field(GradientClipping(), help="Gradient clipping configuration")


GradientClippingConfigT = TypeVar("GradientClippingConfigT", bound=GradientClippingConfig)


class GradientClippingTrainerMixin(
    MixedPrecisionTrainerMixin[GradientClippingConfigT, ModelT, TaskT],
    BaseTrainer[GradientClippingConfigT, ModelT, TaskT],
):
    """Defines a trainer mixin for doing gradient clipping."""

    def clip_grads(self, model: nn.Module, optim: Optimizer) -> None:
        clip_norm = self.config.grad_clipping.clip_grad_norm
        clip_value = self.config.grad_clipping.clip_grad_value
        norm_type = self.config.grad_clipping.norm_type

        if clip_norm is not None:
            if isinstance(model, FSDP):
                total_norm = model.clip_grad_norm_(clip_norm, norm_type)
                self.logger.log_scalar("total_norm", total_norm.item(), namespace="optim")
            else:
                self.unscale_mixed_precision(optim)
                total_norm = nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip_norm, norm_type)
                self.logger.log_scalar("total_norm", total_norm.item(), namespace="optim")

        if clip_value is not None:
            self.unscale_mixed_precision(optim)
            nn.utils.clip_grad.clip_grad_value_(model.parameters(), clip_value)
