import logging
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import torch
from torch import Tensor, nn

from ml.core.config import BaseConfig, BaseObject
from ml.loggers.base import MultiLogger
from ml.utils.colors import Color, colorize

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig(BaseConfig):
    """Defines the base config for all modules."""


ModelConfigType = TypeVar("ModelConfigType", bound=BaseModelConfig)  # pylint: disable=invalid-name


def summarize(names: List[Tuple[str, torch.device]]) -> str:
    return "".join(f"\n ↪ {colorize(k, Color.RED)} - {device}" for k, device in names)


class BaseModel(BaseObject[ModelConfigType], Generic[ModelConfigType], nn.Module):
    """Defines the base module type."""

    def __init__(self, config: ModelConfigType) -> None:
        nn.Module.__init__(self)
        BaseObject.__init__(self, config)

        # Used to log values to the trainer.
        self.logger = MultiLogger(default_namespace="model")

    def init(self, devices: List[torch.device], dtype: Optional[torch.dtype] = None) -> None:
        # Moves all non-meta tensors to the first device.
        def move_to_device(t: Tensor) -> Tensor:
            if t.is_meta:
                return t
            if t.is_floating_point():
                return t.to(device=devices[0], dtype=dtype, non_blocking=True)
            return t.to(device=devices[0], non_blocking=True)

        self._apply(move_to_device)

        # Checks no more `meta` devices.
        device_set = set(devices)

        bad_params = {(name, p.device) for name, p in self.named_parameters() if p.device not in device_set}
        if bad_params:
            bad_param_names = sorted(list(bad_params))[:5]
            logger.warning(
                "Got %d params which are on a different device from %s. First %d:%s",
                len(bad_params),
                device_set,
                len(bad_param_names),
                summarize(bad_param_names),
            )

        bad_buffers = {(name, b.device) for name, b in self.named_buffers() if b.device not in device_set}
        if bad_buffers:
            bad_buffer_names = sorted(list(bad_buffers))[:5]
            logger.warning(
                "Got %d buffers which are on a different device from %s. First %d:\n%s",
                len(bad_buffers),
                device_set,
                len(bad_buffer_names),
                summarize(bad_buffer_names),
            )

    @torch.jit.ignore
    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.jit.ignore
    def get_dtype(self) -> torch.dtype:
        return next(p for p in self.parameters() if p.is_floating_point()).dtype

    @torch.jit.ignore
    def tensor_to(self, tensor: Tensor, non_blocking: bool = False) -> Tensor:
        device, dtype = self.get_device(), self.get_dtype()
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor.to(device, dtype, non_blocking=non_blocking)
        return tensor.to(device, non_blocking=non_blocking)
