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
        """Launches the training process.

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """

        raise NotImplementedError
