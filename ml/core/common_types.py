from typing import Any, TypeVar

from torch import Tensor

Batch = TypeVar("Batch", bound=Tensor | tuple[Any, ...] | list[Any] | dict[Any, Any] | None)
Output = TypeVar("Output", bound=Tensor | tuple[Any, ...] | list[Any] | dict[Any, Any] | None)
Loss = TypeVar("Loss", bound=Tensor | dict[str, Tensor])
