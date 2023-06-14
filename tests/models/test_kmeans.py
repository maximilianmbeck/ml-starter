"""Tests the K-Means PyTorch module.

This tests that the forward pass is computing the centroid IDs correctly.
"""

import torch

from ml.models.kmeans import KMeans


def test_kmeans() -> None:
    centers = torch.randn(4, 8)
    kmeans = KMeans(centers.clone())
    vals = centers[None].repeat(3, 1, 1)

    # Checks that clusters are closest to themselves.
    clusters = kmeans(vals)
    assert (clusters == torch.tensor([0, 1, 2, 3])).all()


if __name__ == "__main__":
    test_kmeans()
