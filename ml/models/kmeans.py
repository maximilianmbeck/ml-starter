"""Defines a distributed K-Means module.

This is used to apply K-Means clusters to a tensor. This module can be used
with cluster centers found via Scikit-Learn, Faiss, or other libraries.
"""

import numpy as np
import torch
from torch import Tensor, nn


class KMeans(nn.Module):
    __constants__ = ["n_clusters", "n_features"]

    centers: Tensor

    def __init__(self, centers: Tensor | np.ndarray) -> None:
        super().__init__()

        n_clusters, n_features = centers.shape
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.register_buffer("centers", torch.empty(n_clusters, n_features))
        self.load_centers(centers)

    def load_centers(self, centers: Tensor | np.ndarray) -> None:
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers)
        assert centers.shape == self.centers.shape, f"Expected shape {self.centers.shape}, got {centers.shape}"
        self.centers.copy_(centers.to(self.centers))

    def forward(self, x: Tensor) -> Tensor:
        """Applies K-Means to get cluster IDs.

        We compute ``(x - centers) ^ 2`` by rewriting as
        ``x ^ 2 - 2 * x * centers + centers ^ 2`` which avoids expanding the
        tensor when doing the norm.

        Args:
            x: The input tensor, with shape ``(*, n_features)``

        Returns:
            The cluster IDs, with shape ``(*)``
        """
        # Equivalent code:
        # dist = torch.norm(x[..., None, :] - self.centers, p=2, dim=-1)
        # return dist.argmin(dim=-1)
        x_norm, centers_norm = (x**2).sum(-1), (self.centers**2).sum(-1)
        dist = x_norm[..., None] - (2 * (x @ self.centers.transpose(0, 1))) + centers_norm[..., None]
        # Absolute value is required here because sometimes the distance
        # can be negative due to numerical instability.
        return dist.abs().argmin(dim=-1)
