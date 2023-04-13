import multiprocessing as mp
import time

import pytest
import torch

from ml.trainers.mixins.gpu_stats import GPUStatsMonitor


@pytest.mark.slow
@pytest.mark.has_gpu
def test_gpu_stats_monitor() -> None:
    """Starts a monitor, waits until it reads something, then kills it."""

    cuda_tensor = torch.randn(1, 2, 3, device="cuda")
    manager = mp.Manager()
    monitor = GPUStatsMonitor(1, manager)
    while len(stats := monitor.get()) == 0:
        time.sleep(1)
    monitor.stop()
    manager.shutdown()
    assert any(v.memory_used > 0 for v in stats.values())
    del cuda_tensor
