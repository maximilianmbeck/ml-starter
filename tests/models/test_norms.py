from typing import get_args

import pytest
import torch

from ml.models.norms import NormType, get_norm_1d, get_norm_2d, get_norm_3d, get_norm_linear


@pytest.mark.parametrize("norm_type", get_args(NormType))
def test_norm_1d(norm_type: NormType) -> None:
    norm = get_norm_1d(norm_type, dim=8, groups=2)
    x = torch.randn(3, 8, 5)
    y = norm(x)
    assert y.shape == (3, 8, 5)


@pytest.mark.parametrize("norm_type", get_args(NormType))
def test_norm_2d(norm_type: NormType) -> None:
    norm = get_norm_2d(norm_type, dim=8, groups=2)
    x = torch.randn(3, 8, 5, 3)
    y = norm(x)
    assert y.shape == (3, 8, 5, 3)


@pytest.mark.parametrize("norm_type", get_args(NormType))
def test_norm_3d(norm_type: NormType) -> None:
    norm = get_norm_3d(norm_type, dim=8, groups=2)
    x = torch.randn(3, 8, 5, 3, 2)
    y = norm(x)
    assert y.shape == (3, 8, 5, 3, 2)


linear_norm_types: list[NormType] = ["no_norm", "batch", "batch_affine", "layer", "layer_affine"]


@pytest.mark.parametrize("norm_type", linear_norm_types)
def test_norm_linear(norm_type: NormType) -> None:
    norm = get_norm_linear(norm_type, dim=8)
    x = torch.randn(3, 5, 8)
    y = norm(x)
    assert y.shape == (3, 5, 8)
