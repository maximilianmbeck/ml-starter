from typing import Literal, cast, get_args

from torch import nn

from ml.tasks.losses.reduce import ReduceType, SampleReduce

LossFnType = Literal["xent", "l1", "mse", "ssim", "smoothing_l1", "smoothing_l2"]


def cast_loss_fn_type(s: str) -> LossFnType:
    args = get_args(LossFnType)
    assert s in args, f"Invalid loss function type: '{s}' Valid options are {args}"
    return cast(LossFnType, s)


def get_loss_fn(loss_fn: LossFnType, reduce_type: ReduceType) -> nn.Module:
    """Gets the loss function module by string key.

    Args:
        loss_fn: The loss function to get
        reduce_type: The type of reduction to use, passed to `SampleReduce`
            (examples are `mean` and `sum`)

    Returns:
        The loss function module

    Raises:
        KeyError: If the requested loss function is not found
    """

    reduce_func = SampleReduce(reduce_type)

    if loss_fn == "xent":
        return nn.Sequential(nn.CrossEntropyLoss(reduction="none"), reduce_func)
    if loss_fn == "l1":
        return nn.Sequential(nn.L1Loss(reduction="none"), reduce_func)
    if loss_fn == "mse":
        return nn.Sequential(nn.MSELoss(reduction="none"), reduce_func)
    raise KeyError(f"Invalid loss function: {loss_fn}")
