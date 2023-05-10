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
- ``nn.LSTM``
- ``nn.GRU``

In the paper, the authors typically use values of 1, 2, 4, or 8 for the
``r`` parameter. The ``lora_alpha`` parameter is typically set to 1.0, but
can be tuned to improve performance.
"""

import math
import warnings
import weakref
from typing import Any, TypeVar, cast, overload

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

T = TypeVar("T")

SupportedModule = nn.Embedding | nn.Linear | nn.Conv1d | nn.Conv2d | nn.LSTM | nn.GRU


def _lora_post_hook(module: "_Lora", incompatible_keys: _IncompatibleKeys) -> None:
    lora_keys = [k for k in incompatible_keys.missing_keys if k.startswith("lora_")]
    for lora_key in lora_keys:
        incompatible_keys.missing_keys.remove(lora_key)


class _Lora(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # This allows modules to use LoRA layers as drop-in replacements for
        # non-LoRA pretrained models without throwing annoying errors for
        # state dict incompatibility.
        self.register_load_state_dict_post_hook(_lora_post_hook)


class LoraEmbedding(nn.Embedding, _Lora):
    __constants__ = nn.Embedding.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
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

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, num_embeddings)))
        self.lora_b = nn.Parameter(self.weight.new_empty((embedding_dim, r)))
        self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            nn.init.zeros_(self.lora_a)
            nn.init.normal_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraEmbedding":
        super().train()

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge and not self.merged:
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


class LoraLinear(nn.Linear, _Lora):
    __constants__ = nn.Linear.__constants__ + ["r", "lora_alpha", "scaling", "merge", "fan_in_fan_out", "merged"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge
        self.fan_in_fan_out = fan_in_fan_out

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_features)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_features, r)))
        self.weight.requires_grad = False

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def _t(self, w: Tensor) -> Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True) -> "LoraLinear":
        super().train()

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self._t(self.lora_b @ self.lora_a) * self.scaling
                self.merged = False

        else:
            if self.merge and not self.merged:
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


class LoraConv1d(nn.Conv1d, _Lora):
    __constants__ = nn.Conv1d.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
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

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *self.kernel_size)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r, 1)))
        self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConv1d":
        super().train()

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        else:
            if self.merge and not self.merged:
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


class LoraConv2d(nn.Conv2d, _Lora):
    __constants__ = nn.Conv2d.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
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

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *self.kernel_size)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r, 1, 1)))
        self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConv2d":
        super().train()

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        else:
            if self.merge and not self.merged:
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


class _LoraRNN(nn.RNNBase, _Lora):
    __constants__ = nn.RNNBase.__constants__ + ["r", "lora_alpha", "scaling"]

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        gate_mul: int,
        r: int,
        lora_alpha: float = 1.0,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        super().__init__(
            mode=mode,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        num_directions = 2 if bidirectional else 1
        gate_size = gate_mul * hidden_size

        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions

                suffix = "_reverse" if direction == 1 else ""
                w_ih: Tensor = getattr(self, f"weight_ih_l{layer}{suffix}")
                w_hh: Tensor = getattr(self, f"weight_hh_l{layer}{suffix}")
                w_ih.requires_grad = False
                w_hh.requires_grad = False
                lora_a_ih = nn.Parameter(w_ih.new_empty((r, gate_size)))
                lora_b_ih = nn.Parameter(w_ih.new_empty((layer_input_size, r)))
                lora_a_hh = nn.Parameter(w_hh.new_empty((r, gate_size)))
                lora_b_hh = nn.Parameter(w_hh.new_empty((real_hidden_size, r)))
                setattr(self, f"lora_a_ih_l{layer}{suffix}", lora_a_ih)
                setattr(self, f"lora_b_ih_l{layer}{suffix}", lora_b_ih)
                setattr(self, f"lora_a_hh_l{layer}{suffix}", lora_a_hh)
                setattr(self, f"lora_b_hh_l{layer}{suffix}", lora_b_hh)

                if self.proj_size != 0:
                    w_hr: Tensor = getattr(self, f"weight_hr_l{layer}{suffix}")
                    w_hr.requires_grad = False
                    lora_a_hr = nn.Parameter(w_hr.new_empty((r, proj_size)))
                    lora_b_hr = nn.Parameter(w_hr.new_empty((hidden_size, r)))
                    setattr(self, f"lora_a_hr_l{layer}{suffix}", lora_a_hr)
                    setattr(self, f"lora_b_hr_l{layer}{suffix}", lora_b_hr)

        self._init_flat_weights()

        self.reset_parameters()

    def _lora_names(self, weight_name: str) -> tuple[str, str]:
        weight_name = weight_name[len("weight_") :]
        lora_a_name, lora_b_name = f"lora_a_{weight_name}", f"lora_b_{weight_name}"
        return lora_a_name, lora_b_name

    def _get_weight(self, weight_name: str) -> Tensor:
        weight = getattr(self, weight_name)
        if weight_name.startswith("bias_"):
            return weight
        lora_a_name, lora_b_name = self._lora_names(weight_name)
        if not hasattr(self, lora_a_name) or not hasattr(self, lora_b_name):
            return weight
        lora_a, lora_b = getattr(self, lora_a_name), getattr(self, lora_b_name)
        return weight + (lora_a.transpose(0, 1) @ lora_b.transpose(0, 1)) * self.scaling

    def _init_flat_weights(self) -> None:
        self._flat_weights = [self._get_weight(wn) if hasattr(self, wn) else None for wn in self._flat_weights_names]
        self._flat_weight_refs = [weakref.ref(w) if w is not None else None for w in self._flat_weights]
        self.flatten_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        for wn in self._flat_weights_names:
            lora_a_name, lora_b_name = self._lora_names(wn)
            if hasattr(self, lora_a_name) and hasattr(self, lora_b_name):
                lora_a, lora_b = getattr(self, lora_a_name), getattr(self, lora_b_name)
                nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
                nn.init.zeros_(lora_b)


class LoraLSTM(nn.LSTM, _LoraRNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r: int,
        lora_alpha: float = 1.0,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        _LoraRNN.__init__(
            self,
            mode="LSTM",
            input_size=input_size,
            hidden_size=hidden_size,
            gate_mul=4,
            r=r,
            lora_alpha=lora_alpha,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )


class LoraGRU(nn.GRU, _LoraRNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r: int,
        lora_alpha: float = 1.0,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        _LoraRNN.__init__(
            self,
            mode="GRU",
            input_size=input_size,
            hidden_size=hidden_size,
            gate_mul=3,
            r=r,
            lora_alpha=lora_alpha,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )


@overload
def lora(module: nn.Embedding, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> LoraEmbedding:
    ...


@overload
def lora(module: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> LoraLinear:
    ...


@overload
def lora(module: nn.Conv1d, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> LoraConv1d:
    ...


@overload
def lora(module: nn.Conv2d, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> LoraConv2d:
    ...


@overload
def lora(module: nn.LSTM, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> LoraLSTM:
    ...


@overload
def lora(module: nn.GRU, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> LoraGRU:
    ...


def lora(module: SupportedModule, r: int, alpha: float = 1.0, dropout: float = 0.0, merge: bool = False) -> nn.Module:
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
            computing the LoRA components. This parameter is not supported
            for RNNs (because it would require modifying the underyling kernel).
        merge: Whether to merge the LoRA components into the original
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
            merge=merge,
        )

    if isinstance(module, nn.Linear):
        return LoraLinear(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=alpha,
            merge=merge,
        )

    if isinstance(module, nn.Conv1d):
        return LoraConv1d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
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
            merge=merge,
            stride=cast(tuple[int, int], module.stride),
            padding=cast(tuple[int, int], module.padding),
            dilation=cast(tuple[int, int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )

    if isinstance(module, nn.LSTM):
        if dropout > 0.0:
            warnings.warn("LoRA dropout is not supported for LSTMs")

        return LoraLSTM(
            module.input_size,
            module.hidden_size,
            r=r,
            lora_alpha=alpha,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            proj_size=module.proj_size,
        )

    if isinstance(module, nn.GRU):
        if dropout > 0.0:
            warnings.warn("LoRA dropout is not supported for GRUs")

        return LoraGRU(
            module.input_size,
            module.hidden_size,
            r=r,
            lora_alpha=alpha,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            proj_size=module.proj_size,
        )

    raise ValueError(f"Unsupported module type {type(module)}")
