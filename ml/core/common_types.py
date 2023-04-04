from typing import TypeVar

from torch import Tensor

Batch = TypeVar("Batch")
Output = TypeVar("Output")
Loss = TypeVar("Loss", bound=Tensor | dict[str, Tensor])

RLAction = TypeVar("RLAction")
RLState = TypeVar("RLState")
