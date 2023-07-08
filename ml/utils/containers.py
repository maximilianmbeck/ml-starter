"""Helper functions for applying functions to containers."""

from dataclasses import is_dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

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


def recursive_chunk(item: Any, num_chunks: int, dim: int = 0) -> Iterable[Any]:  # noqa: ANN401
    """Recursively chunk tensors N times.

    Args:
        item: The item to recursively chunk
        num_chunks: The number of splits to make
        dim: The split dimension

    Yields:
        N chunks of items
    """
    if isinstance(item, (str, int, float)):
        yield from (item for _ in range(num_chunks))
    elif isinstance(item, Tensor):
        yield from item.chunk(num_chunks, dim=dim)
    elif is_dataclass(item):
        yield from (
            item.__class__(**{k: i for k, i in zip(item.__dict__, ii)})
            for ii in zip(*(recursive_chunk(v, num_chunks, dim) for v in item.__dict__.values()))
        )
    elif isinstance(item, Mapping):
        yield from (dict(zip(item, ii)) for ii in zip(*(recursive_chunk(i, num_chunks, dim) for i in item.values())))
    elif isinstance(item, Sequence):
        yield from (list(ii) for ii in zip(*(recursive_chunk(i, num_chunks, dim) for i in item)))
    else:
        yield from (item for _ in range(num_chunks))
