import math
from typing import TypeVar

import torch.nn.functional as F
from torch import Tensor, nn

T = TypeVar("T")


class LoRAEmbedding(nn.Embedding):
    __constants__ = nn.Embedding.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
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

        self.lora_dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
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

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.zeros_(self.lora_a)
            nn.init.normal_(self.lora_b)

    def train(self, mode: bool = True) -> "LoRAEmbedding":
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


class LoRALinear(nn.Linear):
    __constants__ = nn.Linear.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights", "fan_in_fan_out"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
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

        self.lora_dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
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

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def _t(self, w: Tensor) -> Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True) -> "LoRALinear":
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
            mm = self.lora_dropout(x) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            result += mm * self.scaling
            return result

        return F.linear(x, self._t(self.weight), bias=self.bias)


class LoRAConv1D(nn.Conv1d):
    __constants__ = nn.Conv1d.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
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

        self.lora_dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None

        if r > 0:
            self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, kernel_size)))
            self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoRAConv1D":
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
            result = F.conv1d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            mm = self.lora_dropout(x) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            result += mm * self.scaling
            return result

        return F.conv1d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LoRAConv2D(nn.Conv2d):
    __constants__ = nn.Conv2d.__constants__ + ["r", "lora_alpha", "scaling", "merge_weights"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
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

        self.lora_dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a: nn.Parameter | None = None
        self.lora_b: nn.Parameter | None = None

        if r > 0:
            self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *kernel_size)))
            self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b") and self.lora_a is not None and self.lora_b is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoRAConv2D":
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
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            mm = self.lora_dropout(x) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            result += mm * self.scaling
            return result

        return F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
