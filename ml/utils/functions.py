"""Helper functions for applying functions to containers."""

from dataclasses import is_dataclass
from typing import Any, Callable, Mapping, Sequence

from torch import Tensor


def recursive_apply(item: Any, func: Callable[[Tensor], Tensor]) -> Any:  # noqa: ANN401
    """Applies a function recursively to tensors in an item.

    Args:
        item: The item to apply the function to
        func: The function to apply (for the tensor)

    Returns:
        The same item, with the function applied
    """
    if isinstance(item, (str, int, float)):
        return item
    if isinstance(item, Tensor):
        return func(item)
    if is_dataclass(item):
        return item.__class__(**{k: recursive_apply(v, func) for k, v in item.__dict__.items()})
    if isinstance(item, Mapping):
        return {k: recursive_apply(v, func) for k, v in item.items()}
    if isinstance(item, Sequence):
        return [recursive_apply(i, func) for i in item]
    return item
