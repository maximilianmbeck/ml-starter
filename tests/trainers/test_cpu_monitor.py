"""Tests running a CPU monitoring subprocess.

This just starts a monitoring process, waits for it to read something, then
kills it and checks that it actually did read something.
"""

import multiprocessing as mp
import time

import pytest

from ml.trainers.mixins.cpu_stats import CPUStatsMonitor


@pytest.mark.slow
def test_cpu_stats_monitor() -> None:
    manager = mp.Manager()
    monitor = CPUStatsMonitor(0.01, manager)
    for _ in range(3):
        monitor.start(wait=True)
        while (stats := monitor.get()) is None:
            time.sleep(0.01)
        assert stats.mem_rss_total > 0
        monitor.stop()
    manager.shutdown()
