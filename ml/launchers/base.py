from dataclasses import dataclass
from typing import Generic, TypeVar

from ml.core.config import BaseConfig, BaseObjectWithPointers
from ml.trainers.base import BaseTrainer

T = TypeVar("T", bound="BaseLauncher")


@dataclass
class BaseLauncherConfig(BaseConfig):
    pass


LauncherConfigT = TypeVar("LauncherConfigT", bound=BaseLauncherConfig)


class BaseLauncher(BaseObjectWithPointers[LauncherConfigT], Generic[LauncherConfigT]):
    def launch(self, trainer: BaseTrainer) -> None:
        """Launches the training process.

        Args:
            trainer: The trainer being launched

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """

        raise NotImplementedError
