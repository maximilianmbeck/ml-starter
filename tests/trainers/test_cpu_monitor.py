import multiprocessing as mp
import time

import pytest

from ml.trainers.mixins.cpu_stats import CPUStatsMonitor


@pytest.mark.slow
def test_cpu_stats_monitor() -> None:
    """Starts a monitor, waits until it reads something, then kills it."""
    manager = mp.Manager()
    monitor = CPUStatsMonitor(0.01, manager)
    for _ in range(3):
        monitor.start(wait=True)
        while (stats := monitor.get()) is None:
            time.sleep(0.01)
        assert stats.mem_rss_total > 0
        monitor.stop()
    manager.shutdown()
