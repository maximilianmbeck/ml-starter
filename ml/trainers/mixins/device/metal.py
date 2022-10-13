import os

import torch

from ml.trainers.mixins.device.base import BaseDevice


class MetalDevice(BaseDevice):
    """Mixin to support Metal training."""

    @classmethod
    def has_device(cls) -> bool:
        # Use the DISABLE_METAL environment variable if MPS has issues, since
        # it is still in the very early days of support.
        return torch.backends.mps.is_available() and not int(os.environ.get("DISABLE_METAL", "0"))

    @classmethod
    def get_device(cls) -> torch.device:
        return torch.device("mps", 0)

    @classmethod
    def get_floating_point_type(cls) -> torch.dtype:
        return torch.float32
