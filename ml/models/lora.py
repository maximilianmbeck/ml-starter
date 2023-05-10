"""Helper utilities for using LoRA layers.

LoRA layers are drop-in replacements for certain modules, which can be used
for fine-tuning pre-trained models. It is described in the paper
`LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

.. highlight:: python
.. code-block:: python

    from ml.models.lora import lora

    # The pre-trained model weights can be loaded into the LoRA model.
    model = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 5))
    lora_model = nn.Sequential(lora(nn.Linear(5, 7)), lora(nn.Linear(7, 5)))
    lora_model.load_state_dict(model.state_dict())  # No errors

    from ml.models.lora import LoRALinear

    # Alternatively, you can just substitute the module name.
    model = nn.Sequential(LoRALinear(5, 7), LoRALinear(7, 5))

The modules which can be wrapped with LoRA modules are:

- ``nn.Embedding``
- ``nn.Linear``
- ``nn.Conv1d``
- ``nn.Conv2d``

In the paper, the authors typically use values of 1, 2, 4, or 8 for the
``r`` parameter. The ``lora_alpha`` parameter is typically set to 1.0, but
can be tuned to improve performance.
"""

import math
from typing import Any, TypeVar, cast

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

T = TypeVar("T")


def _lora_post_hook(module: "_LoRA", incompatible_keys: _IncompatibleKeys) -> None:
    if "lora_a" in incompatible_keys.missing_keys:
        incompatible_keys.missing_keys.remove("lora_a")
    if "lora_b" in incompatible_keys.missing_keys:
        incompatible_keys.missing_keys.remove("lora_b")


class _LoRA(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # This allows modules to use LoRA layers as drop-in replacements for
        # non-LoRA pretrained models without throwing annoying errors for
        # state dict incompatibility.
        self.register_load_state_dict_post_hook(_lora_post_hook)

    def freeze_non_lora_params(self) -> None:
        """Freezes all parameters except for LoRA parameters."""

        for name, param in self.named_parameters():
            if name not in ("lora_a", "lora_b"):
                param.requires_grad_(False)


class LoraEmbedding(nn.Embedding, _LoRA):
    __constants__ = nn.Embedding.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights", "merged"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze_non_lora_params: bool = True,
        merge_weights: bool = False,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
        )

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge_weights = merge_weights

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None

        if r > 0:
            self.lora_a = nn.Parameter(self.weight.new_empty((r, num_embeddings)))
            self.lora_b = nn.Parameter(self.weight.new_empty((embedding_dim, r)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.reset_parameters()

        if freeze_non_lora_params:
            self.freeze_non_lora_params()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.zeros_(self.lora_a)
            nn.init.normal_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraEmbedding":
        super().train()

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data += (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
                self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = super().forward(x)
            after_a = F.embedding(
                x,
                self.lora_a.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_a @ self.lora_b.transpose(0, 1)) * self.scaling
            return result

        return super().forward(x)


class LoraLinear(nn.Linear, _LoRA):
    __constants__ = nn.Linear.__constants__ + [
        "r",
        "lora_alpha",
        "scaling",
        "merge_weights",
        "fan_in_fan_out",
        "merged",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        freeze_non_lora_params: bool = True,
        merge_weights: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
        )

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None

        if r > 0:
            self.lora_a = nn.Parameter(self.weight.new_empty((r, in_features)))
            self.lora_b = nn.Parameter(self.weight.new_empty((out_features, r)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        if freeze_non_lora_params:
            self.freeze_non_lora_params()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def _t(self, w: Tensor) -> Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True) -> "LoraLinear":
        super().train()

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self._t(self.lora_b @ self.lora_a) * self.scaling
                self.merged = False

        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data += self._t(self.lora_b @ self.lora_a) * self.scaling
                self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.linear(x, self._t(self.weight), bias=self.bias)
            mm = self.dropout(x) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            result += mm * self.scaling
            return result

        return F.linear(x, self._t(self.weight), bias=self.bias)


class LoraConv1d(nn.Conv1d, _LoRA):
    __constants__ = nn.Conv1d.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze_non_lora_params: bool = True,
        merge_weights: bool = False,
        stride: int | tuple[int] = 1,
        padding: str | int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge_weights = merge_weights

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None

        if r > 0:
            self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *self.kernel_size)))
            self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r, 1)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.reset_parameters()

        if freeze_non_lora_params:
            self.freeze_non_lora_params()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConv1d":
        super().train()

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data += self.lora_b @ self.lora_a * self.scaling
                self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            mm_a = F.conv1d(self.dropout(x), self.lora_a, None, self.stride, self.padding, self.dilation, self.groups)
            mm = F.conv1d(mm_a, self.lora_b)
            result += mm * self.scaling
            return result

        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LoraConv2d(nn.Conv2d, _LoRA):
    __constants__ = nn.Conv2d.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze_non_lora_params: bool = True,
        merge_weights: bool = False,
        stride: int | tuple[int, int] = (1, 1),
        padding: str | int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge_weights = merge_weights

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None

        if r > 0:
            self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *self.kernel_size)))
            self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r, 1, 1)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.reset_parameters()

        if freeze_non_lora_params:
            self.freeze_non_lora_params()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConv2d":
        super().train()

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data += self.lora_b @ self.lora_a * self.scaling
                self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            mm_a = F.conv2d(self.dropout(x), self.lora_a, None, self.stride, self.padding, self.dilation, self.groups)
            mm = F.conv2d(mm_a, self.lora_b)
            result += mm * self.scaling
            return result

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def lora(
    module: nn.Embedding | nn.Linear | nn.Conv1d | nn.Conv2d,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    freeze_non_lora_params: bool = True,
    merge_weights: bool = False,
) -> nn.Module:
    """Wraps a module with LoRA.

    This function takes a base module and returns the LoRA version of that
    module. The new module is effectively a drop-in replacement for the
    original module; for example, it can load the same state dict, and it has
    the same input and output shapes.

    Args:
        module: The module to wrap.
        r: The number of LoRA components to use. If 0, then LoRA is not used.
        alpha: The scaling factor for the LoRA components. A higher value
            means that more weight is given to the LoRA components.
        dropout: The dropout probability applied to the input value before
            computing the LoRA components.
        freeze_non_lora_params: If set, freeze the non-LoRA parameters for
            the layer during training.
        merge_weights: Whether to merge the LoRA components into the original
            weights. If True, then the LoRA components are merged into the
            weights during training, and the original weights are used during
            evaluation. If False, then the LoRA components are used during
            both training and evaluation.

    Returns:
        The LoRA version of the module.

    Raises:
        ValueError: If the module is not supported.
    """

    if isinstance(module, nn.Embedding):
        return LoraEmbedding(
            module.num_embeddings,
            module.embedding_dim,
            r=r,
            lora_alpha=alpha,
            merge_weights=merge_weights,
            freeze_non_lora_params=freeze_non_lora_params,
        )

    if isinstance(module, nn.Linear):
        return LoraLinear(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=alpha,
            merge_weights=merge_weights,
            freeze_non_lora_params=freeze_non_lora_params,
        )

    if isinstance(module, nn.Conv1d):
        return LoraConv1d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge_weights=merge_weights,
            freeze_non_lora_params=freeze_non_lora_params,
            stride=cast(tuple[int], module.stride),
            padding=cast(tuple[int], module.padding),
            dilation=cast(tuple[int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )

    if isinstance(module, nn.Conv2d):
        return LoraConv2d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int, int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge_weights=merge_weights,
            freeze_non_lora_params=freeze_non_lora_params,
            stride=cast(tuple[int, int], module.stride),
            padding=cast(tuple[int, int], module.padding),
            dilation=cast(tuple[int, int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )

    raise ValueError(f"Unsupported module type {type(module)}")
