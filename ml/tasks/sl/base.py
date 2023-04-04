# pylint: disable=too-many-public-methods

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch.utils.data.dataset import Dataset

from ml.core.common_types import Batch, Loss, Output
from ml.core.state import Phase
from ml.models.base import BaseModel
from ml.tasks.base import BaseTask, BaseTaskConfig

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SupervisedLearningTaskConfig(BaseTaskConfig):
    pass


SupervisedLearningTaskConfigT = TypeVar("SupervisedLearningTaskConfigT", bound=SupervisedLearningTaskConfig)
ModelT = TypeVar("ModelT", bound=BaseModel)


class SupervisedLearningTask(
    BaseTask[SupervisedLearningTaskConfigT, ModelT, Batch, Output, Loss],
    Generic[SupervisedLearningTaskConfigT, ModelT, Batch, Output, Loss],
    ABC,
):
    """Defines the base task type."""

    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for a given phase.

        Args:
            phase: The dataset phase to get

        Raises:
            NotImplementedError: If this method is not overridden
        """

        raise NotImplementedError("`get_dataset` should be implemented by the task")
