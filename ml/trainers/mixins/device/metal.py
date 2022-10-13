import os
from typing import List

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
    def get_devices(cls) -> List[torch.device]:
        return [torch.device("mps")]

    @classmethod
    def get_floating_point_type(cls) -> torch.dtype:
        return torch.float32
