from dataclasses import dataclass
from typing import Generic, TypeVar

from ml.core.config import BaseConfig, BaseObjectWithPointers
from ml.trainers.base import BaseTrainer


@dataclass
class BaseLauncherConfig(BaseConfig):
    pass


LauncherConfigT = TypeVar("LauncherConfigT", bound=BaseLauncherConfig)
TrainerT = TypeVar("TrainerT", bound=BaseTrainer)


class BaseLauncher(BaseObjectWithPointers[LauncherConfigT], Generic[LauncherConfigT]):
    def launch(self) -> None:
        """Launches the training process.

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """

        raise NotImplementedError
