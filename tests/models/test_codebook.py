"""Tests the codebook learning module.

This test checks that the codebook module is working as intended.
"""

import torch

from ml.models.codebook import Codebook


def test_codebook() -> None:
    codebook = Codebook(
        in_dims=32,
        out_dims=16,
        num_codes=24,
        num_codebooks=4,
    )

    x = torch.randn(32, 2, 3, 32)
    target = torch.randint(0, 24, (32, 2, 3, 4))

    y, loss = codebook.forward(x, target)
    assert y.shape == (32, 2, 3, 16)
    assert loss.shape == (32, 2, 3, 4)

    y = codebook.infer(x)
    assert y.shape == (32, 2, 3, 16)
