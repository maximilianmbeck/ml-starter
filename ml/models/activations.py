from typing import Literal, cast, get_args

from torch import Tensor, nn

ActivationType = Literal[
    "no_act",
    "relu",
    "relu6",
    "clamp6",
    "leaky_relu",
    "elu",
    "celu",
    "selu",
    "sigmoid",
    "log_sigmoid",
    "hard_sigomid",
    "tanh",
    "softsign",
    "softplus",
    "silu",
    "mish",
    "hard_swish",
    "soft_shrink",
    "hard_shrink",
    "tanh_shrink",
    "soft_sign",
]


def cast_activation_type(s: str) -> ActivationType:
    args = get_args(ActivationType)
    assert s in args, f"Invalid activation type: '{s}' Valid options are {args}"
    return cast(ActivationType, s)


class Clamp(nn.Module):
    __constants__ = ["min_value", "max_value", "inplace"]

    def __init__(
        self,
        *,
        value: float | None = None,
        value_range: tuple[float, float] | None = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        assert (value is None) != (value_range is None), "Exactly one of `value` or `value_range` must be specified."

        if value is not None:
            value_range = (-value, value)
        else:
            assert value_range is not None

        self.min_value, self.max_value = value_range
        self.inplace = inplace

        assert self.min_value < self.max_value, f"{self.min_value=} >= {self.max_value=}"

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp_(self.min_value, self.max_value) if self.inplace else x.clamp(self.min_value, self.max_value)


class Clamp6(Clamp):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(value=6.0, inplace=inplace)


def get_activation(act: ActivationType, *, inplace: bool = True) -> nn.Module:
    """Returns an activation function from a keyword string.

    Args:
        act: The keyword for the activation function (None for identity)
        inplace: If set, use the inplace version of the activation function

    Returns:
        The activation function as a module

    Raises:
        NotImplementedError: If the activation function is invalid
    """

    match act:
        case "no_act":
            return nn.Identity()
        case "relu":
            return nn.ReLU(inplace=inplace)
        case "relu6":
            return nn.ReLU6(inplace=inplace)
        case "clamp6":
            return Clamp6(inplace=inplace)
        case "leaky_relu":
            return nn.LeakyReLU(inplace=inplace)
        case "elu":
            return nn.ELU(inplace=inplace)
        case "celu":
            return nn.CELU(inplace=inplace)
        case "selu":
            return nn.SELU(inplace=inplace)
        case "sigmoid":
            return nn.Sigmoid()
        case "log_sigmoid":
            return nn.LogSigmoid()
        case "hard_sigomid":
            return nn.Hardsigmoid(inplace=inplace)
        case "tanh":
            return nn.Tanh()
        case "softsign":
            return nn.Softsign()
        case "softplus":
            return nn.Softplus()
        case "silu":
            return nn.SiLU()
        case "mish":
            return nn.Mish(inplace=inplace)
        case "hard_swish":
            return nn.Hardswish(inplace=inplace)
        case "soft_shrink":
            return nn.Softshrink()
        case "hard_shrink":
            return nn.Hardshrink()
        case "tanh_shrink":
            return nn.Tanhshrink()
        case "soft_sign":
            return nn.Softsign()
        case _:
            raise NotImplementedError(f"Activation function '{act}' is not implemented.")
