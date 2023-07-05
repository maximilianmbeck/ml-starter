"""MPS device support for Metal GPUs (i.e., Apple Silicon)."""

import os
from typing import Callable

import torch

from ml.core.env import is_metal_disabled
from ml.utils.device.base import BaseDevice


def get_env_bool(key: str) -> bool:
    val = int(os.environ.get(key, 0))
    assert val in (0, 1), f"Invalid value for {key}: {val}"
    return val == 1


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
        # Allows users to override the default floating point type.
        if get_env_bool("USE_FP64"):
            return torch.float64
        elif get_env_bool("USE_FP32"):
            return torch.float32
        elif get_env_bool("USE_BF16"):
            return torch.bfloat16
        elif get_env_bool("USE_FP16"):
            return torch.float16

        return torch.float32

    @classmethod
    def get_torch_compile_backend(cls) -> str | Callable:
        return "aot_ts"
