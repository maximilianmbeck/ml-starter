"""Defines a module which learns a codebook.

Codebook learning is done by supervising the target codebook values at
training time, then sampling the codebook values at inference time. This is
analogous to the way that K-Means works, but with a learned codebook instead
of a fixed one.

The model forward pass returns both the computed codebook values and the loss
for the codebook values. The loss moves the codebook embeddings to minimize
the distance between themselves and the input embeddings.

.. code-block:: python

    codebook = Codebook(
        in_dims=128,
        out_dims=64,
        num_codes=256,
        num_codebooks=4,
    )

    x = torch.randn(32, 128)
    target = torch.randint(0, 256, (32, 4))
    y, loss = codebook(x, target)
    assert y.shape == (32, 64)
    assert loss.shape == (32, 4)
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn


class Codebook(nn.Module):
    __constants__ = ["in_dims", "out_dims", "num_codes", "num_codebooks"]

    codebook_inds: Tensor

    def __init__(self, in_dims: int, out_dims: int, num_codes: int, num_codebooks: int) -> None:
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_codes = num_codes
        self.num_codebooks = num_codebooks

        self.codebook = nn.Parameter(torch.empty(num_codes, num_codebooks, out_dims))
        self.proj = nn.Linear(in_dims, num_codebooks * num_codes)
        self.weight_proj = nn.Linear(in_dims, num_codebooks, bias=False)
        self.xent = nn.CrossEntropyLoss(reduction="none")
        self.reset_params()

        self.register_buffer("codebook_inds", torch.arange(self.num_codebooks), persistent=False)

    def reset_params(self) -> None:
        nn.init.normal_(self.codebook, std=1 / math.sqrt(self.out_dims * self.num_codebooks))
        nn.init.normal_(self.proj.weight, std=1 / math.sqrt(self.in_dims))
        nn.init.zeros_(self.proj.bias)

    def get_codebook_output(self, nearest: Tensor) -> Tensor:
        return self.codebook[nearest, self.codebook_inds].sum(-2)

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Gets the nearest codebook item and the codebook loss.

        Args:
            x: The input tensor, used to choose the codebook, with shape
                ``(*, in_dims)``.
            target: The true codebook indices. If None, the codebook targets
                are computed as the nearest codebook item to the input. Should
                have shape ``(*, num_codebooks)`` with integer values in the
                range ``[0, num_codes)``.
            tau: The Gumbel-Softmax temperature.
            hard: Whether to use the hard Gumbel-Softmax.

        Returns:
            The codebook embedding with shape ``(*, out_dims)``, and the
            codebook loss.
        """
        xp = self.proj(x).unflatten(-1, (self.num_codebooks, self.num_codes))  # (..., num_codebooks, num_codes)
        xw = torch.softmax(self.weight_proj(x), -1)  # (..., num_codebooks)

        with torch.no_grad():
            if target is None:
                target = xp.argmax(-1)  # (..., num_codebooks)
            codebook_embs = self.codebook[target, self.codebook_inds]  # (..., num_codebooks, out_dims)

        x_loss = self.xent(xp.permute(0, -1, *range(1, len(xp.shape) - 1)), target)
        return (codebook_embs * xw[..., None]).sum(-2), x_loss

    def infer(self, x: Tensor) -> Tensor:
        """For a given embedding, samples the codebook.

        Args:
            x: The input tensor, with shape ``(*, in_dims)``.

        Returns:
            The codebook embedding with shape ``(*, out_dims)``.
        """
        nearest = self.proj(x).unflatten(-1, (self.num_codebooks, self.num_codes)).argmax(-1)  # (..., num_codebooks)
        return self.get_codebook_output(nearest)  # (..., out_dims)
