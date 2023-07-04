"""Defines the base supervised learning task type.

This class expects you to implement the following functions:

.. code-block:: python

    class MySupervisedLearningTask(ml.SupervisedLearningTask[Config, Model, Batch, Output, Loss]):
        def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
            ...

        def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
            ...

        def get_dataset(self, phase: ml.Phase) -> Dataset:
            ...
"""

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
    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for a given phase.

        Args:
            phase: The dataset phase to get
        """
        raise NotImplementedError("`get_dataset` should be implemented by the task")
