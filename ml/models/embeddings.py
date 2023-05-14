"""Defines a general-purpose API for transformer embedding layers.

.. highlight:: python
.. code-block:: python

    from ml.models.embeddings import get_positional_embeddings, cast_embedding_kind

    embeddings = get_positional_embeddings(
        max_tsz=1024,
        embed_dim=128,
        kind="sinusoidal",
        learnable=False,
    )

    x = torch.arange(3, 5, 8)

    # Time-based positional embeddings - the time tensor supplies the
    # times for each element in the input.
    times = torch.randint(0, 1024, (3, 5))
    y1 = embeddings(x, times=times)

    # Offset-based positional embeddings - the input is assumed to be in
    # temporal order, and the offset is the offset of the first element.
    y2 = embeddings(x, offset=1)

    assert y1.shape == y2.shape == x.shape

    # This lets you parametrize the embedding kind as a string.
    embeddings = get_positional_embeddings(..., kind=cast_embedding_kind(my_kind))

Choices for the embedding kind are:

- ``"identity"``: No positional embeddings are added.
- ``"learned"``: Positional embeddings are learned.
- ``"sinusoidal"``: Sinusoidal embeddings.
- ``"rotary"``: Rotary embeddings (popular for training transformers).
"""

from typing import Literal, cast, get_args, overload

import torch
from torch import Tensor, nn

from ml.models.init import InitializationType, init_

EmbeddingKind = Literal["identity", "learned", "sinusoidal", "rotary"]


def cast_embedding_kind(k: str) -> EmbeddingKind:
    args = get_args(EmbeddingKind)
    assert k in args, f"Invalid initialization type: '{k}' Valid options are {args}"
    return cast(EmbeddingKind, k)


class IdentityPositionalEmbeddings(nn.Module):
    def forward(self, x: Tensor, offset: int = 0, times: Tensor | None = None) -> Tensor:
        return x


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        max_tsz: int,
        embed_dim: int,
        weight_init: InitializationType = "normal",
        learnable: bool = True,
    ) -> None:
        """Defines a learned embeddings module.

        Args:
            max_tsz: The maximum sequence length.
            embed_dim: The embedding dimension.
            weight_init: The initialization type for the embedding weight.
            learnable: Whether the embeddings are learnable.
        """
        super().__init__()

        self.max_tsz = max_tsz
        self.embed_dim = embed_dim
        self.weight_init = weight_init

        self.embeddings = nn.Parameter(torch.empty(max_tsz, embed_dim), requires_grad=learnable)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_(self.embeddings.data, None, self.weight_init)

    def forward(self, x: Tensor, offset: int = 0, times: Tensor | None = None) -> Tensor:
        return x + (self.embeddings[None, offset : offset + x.size(1)] if times is None else self.embeddings[times])


class SinusoidalEmbeddings(LearnedPositionalEmbeddings):
    def __init__(
        self,
        max_tsz: int,
        embed_dim: int,
        learnable: bool = True,
        base: int = 10_000,
    ) -> None:
        """Defines a sinusoidal embeddings module.

        Args:
            max_tsz: The maximum sequence length.
            embed_dim: The embedding dimension.
            learnable: Whether the embeddings are learnable.
            base: The base for the sinusoidal embeddings.
        """
        self.learnable = learnable
        self.base = base

        super().__init__(max_tsz, embed_dim, learnable=learnable)

    def reset_parameters(self) -> None:
        self.embeddings.data.copy_(self.get_embeddings(self.max_tsz))

    def get_embeddings(self, tsz: int) -> Tensor:
        positions = torch.arange(tsz, dtype=torch.float32)
        dim = torch.arange(self.embed_dim, dtype=torch.float32)
        dim = self.base ** (2 * (dim // 2) / self.embed_dim)
        embeddings = positions[:, None] / dim[None, :]
        embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])
        embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])
        return embeddings


class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
        max_tsz: int,
        embed_dim: int,
        learnable: bool = False,
        base: int = 10_000,
    ) -> None:
        """Defines a rotary embeddings module.

        Args:
            max_tsz: The maximum sequence length.
            embed_dim: The embedding dimension.
            learnable: Whether the embeddings are learnable.
            base: The base for the sinusoidal embeddings.
        """
        super().__init__()

        assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4."

        self.embed_dim = embed_dim
        self.learnable = learnable
        self.base = base

        cos, sin = self.get_embeddings(max_tsz)
        self.cos, self.sin = nn.Parameter(cos, requires_grad=learnable), nn.Parameter(sin, requires_grad=learnable)

    def get_embeddings(self, tsz: int) -> tuple[Tensor, Tensor]:
        half_d = self.embed_dim // 2
        theta = 1.0 / (self.base ** (torch.arange(0, half_d, 2).float() / half_d))
        seq_idx = torch.arange(tsz).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        return idx_theta2.cos(), idx_theta2.sin()

    def _neg_half(self, x: Tensor) -> Tensor:
        quarter_d = self.embed_dim // 4
        return torch.cat([-x[..., quarter_d:], x[..., :quarter_d]], dim=-1)

    def forward(self, x: Tensor, offset: int = 0, times: Tensor | None = None) -> Tensor:
        half_d = self.embed_dim // 2
        x_rope, x_pass = x[..., :half_d], x[..., half_d:]
        neg_half_x = self._neg_half(x_rope)
        cos_part = self.cos[None, offset : offset + x.shape[1]] if times is None else self.cos[times]
        sin_part = self.sin[None, offset : offset + x.shape[1]] if times is None else self.sin[times]
        x_rope = x_rope * cos_part + neg_half_x * sin_part
        return torch.cat((x_rope, x_pass), dim=-1)


@overload
def get_positional_embeddings(
    max_tsz: int,
    embed_dim: int,
    kind: Literal["identity"],
) -> IdentityPositionalEmbeddings:
    ...


@overload
def get_positional_embeddings(
    max_tsz: int,
    embed_dim: int,
    kind: Literal["learned"],
    *,
    weight_init: InitializationType = "normal",
    learnable: bool | None = None,
) -> LearnedPositionalEmbeddings:
    ...


@overload
def get_positional_embeddings(
    max_tsz: int,
    embed_dim: int,
    kind: Literal["sinusoidal"],
    *,
    learnable: bool | None = None,
    base: int = 10_000,
) -> SinusoidalEmbeddings:
    ...


@overload
def get_positional_embeddings(
    max_tsz: int,
    embed_dim: int,
    kind: Literal["rotary"],
    *,
    learnable: bool | None = None,
    base: int = 10_000,
) -> RotaryEmbeddings:
    ...


@overload
def get_positional_embeddings(
    max_tsz: int,
    embed_dim: int,
    kind: EmbeddingKind,
    *,
    weight_init: InitializationType = "normal",
    learnable: bool | None = None,
    base: int = 10_000,
) -> IdentityPositionalEmbeddings | LearnedPositionalEmbeddings | SinusoidalEmbeddings | RotaryEmbeddings:
    ...


def get_positional_embeddings(
    max_tsz: int,
    embed_dim: int,
    kind: EmbeddingKind,
    *,
    weight_init: InitializationType = "normal",
    learnable: bool | None = None,
    base: int = 10_000,
) -> nn.Module:
    """Defines the common module for adding positional embeddings.

    Args:
        max_tsz: The maximum sequence length.
        embed_dim: The embedding dimension.
        kind: The type of embedding to use.
        weight_init: The weight initialization for learned embeddings.
        learnable: Whether the embeddings are learnable; if not provided,
            uses sensible defaults.
        base: The base for the sinusoidal embeddings.

    Returns:
        The positional embeddings module.

    Raises:
        ValueError: If an invalid embedding kind is supplied.
    """
    match kind:
        case "identity":
            return IdentityPositionalEmbeddings()

        case "learned":
            return LearnedPositionalEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                weight_init=weight_init,
                learnable=True if learnable is None else learnable,
            )

        case "sinusoidal":
            return SinusoidalEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                learnable=False if learnable is None else learnable,
                base=base,
            )

        case "rotary":
            return RotaryEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                learnable=False if learnable is None else learnable,
                base=base,
            )

        case _:
            raise ValueError(f"Invalid embedding kind: {kind}")
