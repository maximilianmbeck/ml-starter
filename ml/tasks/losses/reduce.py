"""Defines helper functions for doing tensor reductions.

This just makes it so that tensor reductions can be defined as strings but
retain type checking, for example:

.. code-block:: python

    from ml.tasks.losses.reduce import reduce

    t = torch.randn(10, 10)
    t_reduced = reduce(t, "mean", dim=0)
"""

from typing import Literal, cast, get_args

from torch import Tensor, nn

ReduceType = Literal["mean", "sum", None]


def cast_reduce_type(s: str | None) -> ReduceType:
    args = get_args(ReduceType)
    assert s in args, f"Invalid reduce type: '{s}' Valid options are {args}"
    return cast(ReduceType, s)


def reduce(t: Tensor, reduce_type: ReduceType, dim: int | None = None) -> Tensor:
    match reduce_type:
        case None:
            return t
        case "mean":
            return t.mean(dim)
        case "sum":
            return t.sum(dim)
        case _:
            raise NotImplementedError(f"Unexpected sample reduction type: {reduce_type}")


class SampleReduce(nn.Module):
    def __init__(self, reduce_type: ReduceType) -> None:
        super().__init__()

        self.reduce_type = reduce_type

    def forward(self, loss: Tensor) -> Tensor:
        loss_flat = loss.flatten(1)
        return reduce(loss_flat, self.reduce_type, dim=1)
