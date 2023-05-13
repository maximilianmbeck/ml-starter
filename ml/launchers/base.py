"""Base launcher class and config."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from ml.core.config import BaseConfig, BaseObject

T = TypeVar("T", bound="BaseLauncher")


@dataclass
class BaseLauncherConfig(BaseConfig):
    pass


LauncherConfigT = TypeVar("LauncherConfigT", bound=BaseLauncherConfig)


class BaseLauncher(BaseObject[LauncherConfigT], Generic[LauncherConfigT]):
    def launch(self) -> None:
        """Launches the training process."""
        raise NotImplementedError
