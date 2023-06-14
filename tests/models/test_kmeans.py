"""Tests the K-Means PyTorch module.

This tests that the forward pass is computing the centroid IDs correctly.
"""

import pytest
import torch

from ml.models.kmeans import KMeans, kmeans_fn


def test_kmeans() -> None:
    centers = torch.randn(4, 12)
    kmeans = KMeans(centers.clone(), use_triton_if_available=False)
    vals = centers[None].repeat(3, 2, 1)

    # Checks that clusters are closest to themselves.
    clusters = kmeans(vals)
    assert (clusters == torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])).all()


@pytest.mark.has_triton()
def test_kmeans_against_triton() -> None:
    centers = torch.randn(4, 12, device="cuda")
    vals = centers[None].repeat(3, 2, 1)

    vanilla_fn = kmeans_fn(use_triton=False)
    triton_fn = kmeans_fn(use_triton=True)

    vanilla_clusters = vanilla_fn(vals, centers, (centers**2).sum(-1))
    triton_clusters = triton_fn(vals, centers, (centers**2).sum(-1))

    assert (vanilla_clusters == triton_clusters).all()


if __name__ == "__main__":
    # python -m tests.models.test_kmeans
    # test_kmeans()
    test_kmeans_against_triton()
