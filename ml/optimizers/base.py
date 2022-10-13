from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

from torch import nn
from torch.optim.optimizer import Optimizer

from ml.core.config import BaseConfig, BaseObjectWithPointers


@dataclass
class BaseOptimizerConfig(BaseConfig):
    """Defines the base config for all optimizers."""


OptimizerConfigType = TypeVar("OptimizerConfigType", bound=BaseOptimizerConfig)  # pylint: disable=invalid-name


class BaseOptimizer(BaseObjectWithPointers[OptimizerConfigType], Generic[OptimizerConfigType], ABC):
    """Defines the base optimizer type."""

    @property
    def common_kwargs(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def get(self, model: nn.Module) -> Optimizer:
        """Given a base module, returns an optimizer.

        Args:
            model: The model to get an optimizer for

        Returns:
            The constructed optimizer
        """
