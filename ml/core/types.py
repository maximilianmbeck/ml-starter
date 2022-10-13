from typing import Any, Dict, Union

from torch import Tensor

Batch = Any
Output = Any
Loss = Union[Tensor, Dict[str, Tensor]]
