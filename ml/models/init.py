import math
from typing import Literal, cast, get_args

import torch
from torch import Tensor, nn

InitializationType = Literal[
    "orthogonal",
    "normal",
    "biased_normal",
    "uniform",
    "kaiming_uniform",
    "kaiming_normal",
    "xavier_uniform",
    "xavier_normal",
    "ones",
]


def cast_init_type(s: str) -> InitializationType:
    args = get_args(InitializationType)
    assert s in args, f"Invalid initialization type: '{s}' Valid options are {args}"
    return cast(InitializationType, s)


def _uniform_bias(weight: Tensor, bias: Tensor | None) -> Tensor | None:
    if bias is None:
        return None
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in == 0:
        nn.init.zeros_(bias)
    else:
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
    return bias


def init_(
    weight: Tensor,
    bias: Tensor | None,
    init: InitializationType,
    *,
    normal_std: float = 0.01,
    uniform_scale: float = 0.02,
) -> tuple[Tensor, Tensor | None]:
    """Initializes the weight and bias in-place, using an initialization key.

    The weight and bias are from a convolution or linear layer.

    Args:
        weight: The weight tensor
        bias: The bias tensor
        init: The initialization type to use
        normal_std: The standard deviation for normal initialization
        uniform_scale: The scale amount for uniform initialization

    Returns:
        The initialized weight and bias (which can be discarded, since the
        initialization happens in-place).

    Raises:
        NotImplementedError: If the initialization mode isn't implemented
    """

    # Don't do anything for meta tensors.
    if weight.is_meta:
        return weight, bias
    if isinstance(weight, nn.Parameter):
        weight = weight.data
    if isinstance(bias, nn.Parameter):
        bias = bias.data
    match init:
        case "orthogonal":
            if weight.dtype == torch.float16:
                return (
                    weight.copy_(nn.init.orthogonal_(weight.float(), gain=0.01).to(weight)),
                    None if bias is None else nn.init.zeros_(bias),
                )
            return nn.init.orthogonal_(weight), None if bias is None else nn.init.zeros_(bias)
        case "normal":
            return nn.init.normal_(weight, std=normal_std), None if bias is None else nn.init.zeros_(bias)
        case "biased_normal":
            return (
                nn.init.normal_(weight, std=normal_std),
                None if bias is None else nn.init.normal_(bias, std=normal_std),
            )
        case "uniform":
            return nn.init.uniform_(weight, b=uniform_scale), None if bias is None else nn.init.zeros_(bias)
        case "kaiming_uniform":
            return nn.init.kaiming_uniform_(weight), _uniform_bias(weight, bias)
        case "kaiming_normal":
            return nn.init.kaiming_normal_(weight), _uniform_bias(weight, bias)
        case "xavier_uniform":
            return nn.init.xavier_uniform_(weight), _uniform_bias(weight, bias)
        case "xavier_normal":
            return nn.init.xavier_normal_(weight), _uniform_bias(weight, bias)
        case "ones":
            return nn.init.ones_(weight), None if bias is None else nn.init.zeros_(bias)
        case _:
            raise NotImplementedError(f"Unexpected initialization: {init}")
