"""Tests the K-Means PyTorch module.

This tests that the forward pass is computing the centroid IDs correctly, and
that updating the centroids works as expected as well.
"""

import torch

from ml.models.kmeans import KMeans


def test_kmeans() -> None:
    kmeans = KMeans(n_clusters=4, n_features=8)
    init_centers = kmeans.centers.clone()
    vals = init_centers[None].repeat(3, 1, 1)

    # Checks that clusters are closest to themselves.
    clusters = kmeans(vals)
    assert (clusters == torch.tensor([0, 1, 2, 3])).all()

    # Updates clusters with self.
    kmeans.update_(vals, clusters)
    assert torch.allclose(init_centers, kmeans.centers, atol=1e-4)

    # Tests only updating some of the clusters.
    kmeans.update_(torch.randn_like(vals), clusters % 2)
    assert torch.allclose(init_centers[2:], kmeans.centers[2:], atol=1e-4)


if __name__ == "__main__":
    test_kmeans()
