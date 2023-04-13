from typing import Callable

import torch

from ml.core.env import is_metal_disabled
from ml.utils.device.base import BaseDevice


class MetalDevice(BaseDevice):
    """Mixin to support Metal training."""

    @classmethod
    def has_device(cls) -> bool:
        # Use the DISABLE_METAL environment variable if MPS has issues, since
        # it is still in the very early days of support.
        return torch.backends.mps.is_available() and not is_metal_disabled()

    @classmethod
    def get_device(cls) -> torch.device:
        return torch.device("mps", 0)

    @classmethod
    def get_floating_point_type(cls) -> torch.dtype:
        return torch.float32

    @classmethod
    def get_torch_compile_backend(cls) -> str | Callable:
        return "aot_ts"
