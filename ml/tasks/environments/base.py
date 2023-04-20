"""Defines a generic reinforcement learning environment class."""

import logging
from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from torch import Tensor

from ml.core.common_types import RLAction, RLState

logger = logging.getLogger(__name__)


class Environment(ABC, Generic[RLState, RLAction]):
    @abstractmethod
    def reset(self, seed: int | None = None) -> RLState:
        """Gets the initial environment state.

        Args:
            seed: The initial random seed to use

        Returns:
            The initial state of the environment.
        """

    @abstractmethod
    def render(self, state: RLState) -> np.ndarray | Tensor:
        """Renders the environment.

        Args:
            state: The state to render

        Returns:
            The rendered environment as a single frame, as an image array.
        """

    @abstractmethod
    def sample_action(self) -> RLAction:
        """Samples an action from the environment's action space.

        Returns:
            The sampled action.
        """

    @abstractmethod
    def step(self, action: RLAction) -> RLState:
        """Performs a single step in the environment.

        Args:
            action: The action to perform in the environment.

        Returns:
            The next state of the environment.
        """

    @abstractmethod
    def terminated(self, state: RLState) -> bool:
        """Checks if the environment has finished.

        Args:
            state: The most recent state

        Returns:
            If the environment has finished
        """

    @property
    def fps(self) -> int:
        return 30
