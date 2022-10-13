from typing import List

import torch

from ml.trainers.mixins.device.base import BaseDevice


class CPUDevice(BaseDevice):
    """Mixin to support CPU training."""

    @classmethod
    def has_device(cls) -> bool:
        return True

    @classmethod
    def get_devices(cls) -> List[torch.device]:
        return [torch.device("cpu")]

    @classmethod
    def get_floating_point_type(cls) -> torch.dtype:
        return torch.float32
