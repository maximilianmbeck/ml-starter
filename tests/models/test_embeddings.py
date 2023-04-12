from typing import get_args

import pytest
import torch

from ml.models.embeddings import EmbeddingKind, get_positional_embeddings


@pytest.mark.parametrize("kind", get_args(EmbeddingKind))
@pytest.mark.parametrize("use_times", [True, False])
def test_embeddings_api(kind: EmbeddingKind, use_times: bool) -> None:
    x = torch.randn(3, 5, 8)
    times = torch.randint(0, 11, (3, 5)) if use_times else None
    emb = get_positional_embeddings(max_tsz=12, embed_dim=8, kind=kind)
    y = emb(x, times=times)
    assert y.shape == (3, 5, 8)
