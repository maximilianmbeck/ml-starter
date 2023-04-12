from typing import get_args

import pytest
import torch

from ml.models.embeddings import EmbeddingKind, get_positional_embeddings


@pytest.mark.parametrize("kind", get_args(EmbeddingKind))
def test_embeddings_api(kind: EmbeddingKind) -> None:
    x = torch.randn(3, 5, 8)
    times = torch.arange(1, 6)[None, :].repeat(3, 1)
    emb = get_positional_embeddings(max_tsz=12, embed_dim=8, kind=kind)
    y1 = emb(x, times=times)
    y2 = emb(x, offset=1)
    assert y1.shape == (3, 5, 8)
    assert y2.shape == (3, 5, 8)
    assert torch.allclose(y1, y2)
