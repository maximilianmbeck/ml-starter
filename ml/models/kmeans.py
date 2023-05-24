"""Defines a distributed K-Means module.

This is used to apply K-Means clusters to a tensor. This can also be used for
rudimentary learning of clusters.
"""

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ReduceOp

from ml.utils.parallel import _GroupInfo, default_group_info


class KMeans(nn.Module):
    __constants__ = ["n_clusters", "n_features"]

    centers: Tensor
    counts: Tensor

    def __init__(self, n_clusters: int, n_features: int, *, group: _GroupInfo | None = None) -> None:
        super().__init__()

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.register_buffer("centers", torch.randn(n_clusters, n_features))
        self.register_buffer("counts", torch.zeros(n_clusters, dtype=torch.long))

        self._group_info = default_group_info() if group is None else group

    def load_weight_(self, weight: Tensor | np.ndarray) -> None:
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight)
        assert weight.shape == self.centers.shape, f"Expected shape {self.centers.shape}, got {weight.shape}"
        self.centers.copy_(weight.to(self.centers))

    def zero_counts_(self) -> None:
        self.counts.zero_()

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

    def update_(self, x: Tensor, cluster_ids: Tensor) -> None:
        """Updates the K-Means cluster centers using the cluster IDs.

        Args:
            x: The input tensor, with shape ``(*, n_features)``
            cluster_ids: The cluster IDs, with shape ``(*)``
        """
        x, cluster_ids = x.flatten(0, -2), cluster_ids.flatten()
        cluster_ids_x = cluster_ids[..., None].repeat(1, x.shape[-1])
        clusters, counts = torch.zeros_like(self.centers), torch.zeros_like(self.counts)
        clusters = torch.scatter_reduce(clusters, 0, cluster_ids_x, x, "sum")
        counts = torch.scatter_reduce(counts, 0, cluster_ids, torch.ones_like(cluster_ids), "sum")

        # Handles data-parallel updates, to keep clusters in sync everywhere.
        if self._group_info is not None:
            clusters_work = self._group_info.reduce(clusters, op=ReduceOp.SUM, async_op=True)
            counts_work = self._group_info.reduce(counts, op=ReduceOp.SUM, async_op=True)
            clusters_work.wait()
            counts_work.wait()

        new_counts = self.counts + counts
        sigma = (counts / new_counts)[..., None]
        counts = counts[..., None]

        new_centers = clusters / counts.clamp_min_(1)
        self.centers.copy_(new_centers * sigma + self.centers * (1 - sigma))
        self.counts.copy_(new_counts)
