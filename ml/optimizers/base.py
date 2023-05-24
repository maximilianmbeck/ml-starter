"""Defines the base optimizer adapter.

This class usually just wraps PyTorch optimizers, providing some common
hyperparameter configurations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import nn
from torch.optim.optimizer import Optimizer

from ml.core.config import BaseConfig, BaseObject

OptimizerT = TypeVar("OptimizerT", bound=Optimizer)


@dataclass
class BaseOptimizerConfig(BaseConfig):
    """Defines the base config for all optimizers."""


OptimizerConfigT = TypeVar("OptimizerConfigT", bound=BaseOptimizerConfig)


class BaseOptimizer(BaseObject[OptimizerConfigT], Generic[OptimizerConfigT, OptimizerT], ABC):
    """Defines the base optimizer type."""

    @property
    def common_kwargs(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def get(self, model: nn.Module) -> OptimizerT:
        """Given a base module, returns an optimizer.

        Args:
            model: The model to get an optimizer for

        Returns:
            The constructed optimizer
        """
