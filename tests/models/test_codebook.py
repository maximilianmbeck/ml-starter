"""Tests the codebook learning module.

This test checks that the codebook module is working as intended.
"""

import torch
from torch import optim

from ml.models.codebook import Codebook


def test_codebook() -> None:
    codebook = Codebook(
        in_dims=32,
        out_dims=16,
        num_codes=24,
        num_codebooks=4,
    )

    opt = optim.SGD(codebook.parameters(), lr=1e-3)

    x = torch.randn(32, 2, 3, 32)
    target = torch.randint(0, 24, (32, 2, 3, 4))

    # Runs forward pass with explicit targets.
    y, loss = codebook.forward(x, target)
    assert y.shape == (32, 2, 3, 16)
    assert loss.shape == (32, 2, 3, 4)

    # Runs inference pass.
    with torch.no_grad():
        y = codebook.infer(x)
        assert y.shape == (32, 2, 3, 16)

    # Runs the backward pass and check that it works as intended.
    loss.sum().backward()
    opt.step()
    _, loss2 = codebook.forward(x, target)
    assert (loss2 <= loss).sum() > loss.numel() * 0.75

    # Runs forward pass without explicit targets.
    y, loss = codebook.forward(x)
    assert y.shape == (32, 2, 3, 16)
    assert loss.shape == (32, 2, 3, 4)
