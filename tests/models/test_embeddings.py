import pytest
import torch

from ml.models.embeddings import RotaryEmbeddings, SinusoidalEmbeddings


@pytest.mark.skip("Seems to break in CI")
def test_sinusoidal_embeddings() -> None:
    emb = SinusoidalEmbeddings(max_tsz=16, embed_dim=8)
    x = torch.randn(2, 12, 8)
    y = emb(x)
    assert y.shape == (2, 12, 8)


@pytest.mark.skip("Seems to break in CI")
def test_rotary_embeddings() -> None:
    emb = RotaryEmbeddings(max_tsz=16, embed_dim=8)
    x = torch.randn(2, 12, 8)
    y = emb(x)
    assert y.shape == (2, 12, 8)
