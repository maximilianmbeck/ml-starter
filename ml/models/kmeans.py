"""Defines a distributed K-Means module.

This is used to apply K-Means clusters to a tensor. This module can be used
with cluster centers found via Scikit-Learn, Faiss, or other libraries.
"""

import logging
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _vanilla_kmeans(x: Tensor, centers: Tensor, centers_norm: Tensor) -> Tensor:
    # Equivalent code:
    # dist = torch.norm(x[..., None, :] - centers, p=2, dim=-1)
    # return dist.argmin(dim=-1)
    x_norm = (x**2).sum(-1)
    dist = x_norm[..., None] - (2 * (x @ centers.transpose(0, 1))) + centers_norm
    # Absolute value is required here because sometimes the distance
    # can be negative due to numerical instability.
    return dist.abs().argmin(dim=-1)


def kmeans_fn(use_triton: bool) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    if torch.cuda.is_available() and use_triton:
        try:
            from ml.utils.triton.kmeans import kmeans as triton_kmeans_fn

            return triton_kmeans_fn

        except ImportError:
            logger.warning("Triton is not installed. Using vanilla Lion update function.")

    return _vanilla_kmeans


class KMeans(nn.Module):
    __constants__ = ["n_clusters", "n_features"]

    centers: Tensor
    centers_norm: Tensor

    def __init__(self, centers: Tensor | np.ndarray, *, use_triton_if_available: bool = True) -> None:
        super().__init__()

        n_clusters, n_features = centers.shape
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.register_buffer("centers", torch.empty(n_clusters, n_features))
        self.register_buffer("centers_norm", torch.empty(n_clusters))
        self.load_centers(centers)
        self.kmeans_fn = kmeans_fn(use_triton_if_available)

    def load_centers(self, centers: Tensor | np.ndarray) -> None:
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers)
        assert centers.shape == self.centers.shape, f"Expected shape {self.centers.shape}, got {centers.shape}"
        self.centers.copy_(centers.to(self.centers))
        self.centers_norm.copy_((self.centers**2).sum(-1))

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
        return self.kmeans_fn(x, self.centers, self.centers_norm)
