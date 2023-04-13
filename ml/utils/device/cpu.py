from typing import Callable

import torch

from ml.utils.device.base import BaseDevice


class CPUDevice(BaseDevice):
    """Mixin to support CPU training."""

    @classmethod
    def has_device(cls) -> bool:
        return True

    @classmethod
    def get_device(cls) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def get_floating_point_type(cls) -> torch.dtype:
        return torch.float32

    @classmethod
    def get_torch_compile_backend(cls) -> str | Callable:
        return "aot_ts"
