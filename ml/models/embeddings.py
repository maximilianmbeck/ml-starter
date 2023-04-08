import torch
from torch import Tensor, nn


class SinusoidalEmbeddings(nn.Module):
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

        super().__init__()

        self.embed_dim = embed_dim
        self.learnable = learnable
        self.base = base

        self.embeddings = nn.Parameter(self.get_embeddings(max_tsz), requires_grad=learnable)

    def get_embeddings(self, tsz: int) -> Tensor:
        positions = torch.arange(tsz, dtype=torch.float32)
        dim = torch.arange(self.embed_dim, dtype=torch.float32)
        dim = self.base ** (2 * (dim // 2) / self.embed_dim)
        embeddings = positions[:, None] / dim[None, :]
        embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])
        embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])
        return embeddings

    def forward(self, x: Tensor) -> Tensor:
        """Computes the sinusoidal embeddings for the given input.

        Args:
            x: The input tensor, with shape (B, T, D).

        Returns:
            The tensor with the embeddings added, wiith shape (B, T, D).
        """

        return x + self.embeddings[None, : x.size(1)]


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

    def forward(self, x: Tensor) -> Tensor:
        """Computes the rotary embeddings for the given input.

        Args:
            x: The input tensor, with shape (B, T, D).

        Returns:
            The tensor with the embeddings added, wiith shape (B, T, D).
        """

        half_d = self.embed_dim // 2
        x_rope, x_pass = x[..., :half_d], x[..., half_d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos[None, : x.shape[1]]) + (neg_half_x * self.sin[None, : x.shape[1]])
        return torch.cat((x_rope, x_pass), dim=-1)
