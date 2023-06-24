"""Defines a mixin for doing FP16 scaling.

FP16 scaling is a technique for training with FP16 precision while maintaining
FP32 precision for the model weights. This is done by scaling the loss by a
large factor (e.g. 2^16) and then scaling the gradients by the inverse of that
factor. So if the scale factor starts to decrease, it means that the loss is
overflowing and training is diverging.
"""

from dataclasses import dataclass
from typing import Any, ContextManager, TypeVar

import torch
from torch import Tensor
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT


@dataclass
class MixedPrecisionConfig:
    enabled: bool = conf_field(True, help="If set, should FP16 training be enabled")
    init_scale: float = conf_field(2.0**16, help="Initial scaling factor")
    growth_factor: float = conf_field(2.0, help="Factor by which the scale is multiplied if no gradient NaNs occur")
    backoff_factor: float = conf_field(0.5, help="Factor by which the scale is multiplied if gradient NaNs occur")
    growth_interval: int = conf_field(2000, help="How often to grow the scale")


@dataclass
class MixedPrecisionTrainerConfig(BaseTrainerConfig):
    mixed_precision: MixedPrecisionConfig = conf_field(MixedPrecisionConfig(), help="Mixed precision configuration")


MixedPrecisionConfigT = TypeVar("MixedPrecisionConfigT", bound=MixedPrecisionTrainerConfig)


class MixedPrecisionTrainerMixin(BaseTrainer[MixedPrecisionConfigT, ModelT, TaskT]):
    """Defines a trainer mixin for doing FP16 scaling."""

    def __init__(self, config: MixedPrecisionConfigT) -> None:
        super().__init__(config)

        self.grad_scaler: torch.cuda.amp.GradScaler | None
        if self._device_type == "cuda" and self.config.mixed_precision.enabled:
            self.grad_scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.mixed_precision.init_scale,
                growth_factor=self.config.mixed_precision.growth_factor,
                backoff_factor=self.config.mixed_precision.backoff_factor,
                growth_interval=self.config.mixed_precision.growth_interval,
                enabled=self.config.mixed_precision.enabled and self._device.supports_grad_scaler(),
            )
        else:
            self.grad_scaler = None

    def scale_mixed_precision(self, tensor: Tensor) -> Tensor:
        if self.grad_scaler is not None:
            return self.grad_scaler.scale(tensor)
        return tensor

    def unscale_mixed_precision(self, optim: Optimizer) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optim)

    def step_optimizer(self, optim: Optimizer) -> None:
        if self.grad_scaler is None:
            optim.step()
        else:
            self.grad_scaler.step(optim)
            self.grad_scaler.update()

    def log_mp_scale(self) -> None:
        if (scaler := self.grad_scaler) is not None:
            if (scale := getattr(scaler, "_scale", None)) is not None:
                self.logger.log_scalar("fp16_scale", scale)

    def load_state_dict(self, ckpt: dict[str, Any]) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(ckpt["grad_scaler"])

        super().load_state_dict(ckpt)

    def update_state_dict(self, ckpt: dict[str, Any]) -> None:
        if self.grad_scaler is not None:
            assert "grad_scaler" not in ckpt

            ckpt["grad_scaler"] = self.grad_scaler.state_dict()

        super().update_state_dict(ckpt)

    def autocast_context(self) -> ContextManager:
        return self._device.autocast_context(enabled=self.config.mixed_precision.enabled)
