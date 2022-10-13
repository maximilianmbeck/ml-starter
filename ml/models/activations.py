from typing import Literal, cast, get_args

from torch import Tensor, nn

ActivationType = Literal[
    "no_act",
    "relu",
    "relu6",
    "clamp6",
    "leaky_relu",
    "elu",
    "selu",
    "sigmoid",
    "log_sigmoid",
    "tanh",
    "softsign",
    "softplus",
    "silu",
]


def cast_activation_type(s: str) -> ActivationType:
    args = get_args(ActivationType)
    assert s in args, f"Invalid activation type: '{s}' Valid options are {args}"
    return cast(ActivationType, s)


class Clamp6(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp_(-6, 6) if self.inplace else x.clamp(-6, 6)


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

    if act == "no_act":
        return nn.Identity()
    if act == "relu":
        return nn.ReLU(inplace)
    if act == "relu6":
        return nn.ReLU6(inplace)
    if act == "clamp6":
        return Clamp6(inplace)
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    if act == "elu":
        return nn.ELU(alpha=1.0, inplace=inplace)
    if act == "selu":
        return nn.SELU(inplace=inplace)
    if act == "sigmoid":
        return nn.Sigmoid()
    if act == "log_sigmoid":
        return nn.LogSigmoid()
    if act == "tanh":
        return nn.Tanh()
    if act == "softsign":
        return nn.Softsign()
    if act == "softplus":
        return nn.Softplus(beta=1, threshold=20)
    if act == "silu":
        return nn.SiLU(inplace=inplace)
    raise NotImplementedError(f"Invalid activation function: {act}")
