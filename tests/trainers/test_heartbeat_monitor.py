import functools
import multiprocessing as mp
import time
from multiprocessing.synchronize import Event

import pytest

from ml.trainers.mixins.heartbeat import HeartbeatMonitor


def set_event(pid: int, heartbeat_event: Event, test_event: Event) -> None:
    test_event.set()


@pytest.mark.slow
def test_cpu_stats_monitor() -> None:
    """Starts a monitor, waits until it reads something, then kills it."""
    manager = mp.Manager()
    test_event = manager.Event()
    monitor = HeartbeatMonitor(0.01, manager, functools.partial(set_event, test_event=test_event))
    for _ in range(3):
        monitor.start(wait=True)
        time.sleep(0.2)
        assert test_event.is_set()
        monitor.stop()
    manager.shutdown()


if __name__ == "__main__":
    test_cpu_stats_monitor()
