from typing import Any, Union

from torch import Tensor

Batch = Any
Output = Any
Loss = Union[Tensor, dict[str, Tensor]]
