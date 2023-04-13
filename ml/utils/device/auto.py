import functools
import logging
from typing import Type

from ml.utils.device.base import BaseDevice
from ml.utils.device.cpu import CPUDevice
from ml.utils.device.gpu import GPUDevice
from ml.utils.device.metal import MetalDevice
from ml.utils.logging import INFOALL

logger: logging.Logger = logging.getLogger(__name__)

# These devices are ordered by priority, so an earlier device in the list
# is preferred to a later device in the list.
ALL_DEVICES: list[Type[BaseDevice]] = [
    MetalDevice,
    GPUDevice,
    CPUDevice,
]


class AutoDevice:
    """Mixin to automatically detect the device type to use."""

    @classmethod
    @functools.lru_cache(None)
    def detect_device(cls) -> Type[BaseDevice]:
        for device_type in ALL_DEVICES:
            if device_type.has_device():
                logger.log(INFOALL, "Device: [%s]", device_type.get_device())
                return device_type
        raise RuntimeError("Could not automatically detect the device to use")

    @classmethod
    def get_device_from_key(cls, key: str) -> Type[BaseDevice]:
        if key == "auto":
            return AutoDevice.detect_device()
        if key == "cpu":
            return CPUDevice
        if key == "metal":
            return MetalDevice
        if key == "gpu":
            return GPUDevice
        raise NotImplementedError(f"Device type not found: {key}")
