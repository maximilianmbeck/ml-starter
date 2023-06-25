"""Defines I/O utility functions.

This module contains a global read lock that can be used to prevent multiple
processes from doing disk reads at the same time. This is useful when reading
from a shared filesystem, such as NFS, where concurrent reads can cause
performance issues.

.. code-block:: python

    from ml.utils.io import read_lock, set_read_lock

    set_read_lock(lock)

    with read_lock:
        # Do some disk reads.
"""

from multiprocessing.synchronize import Lock
from types import TracebackType
from typing import ContextManager, Iterator, TypeVar

T = TypeVar("T")


class _ReadLock(ContextManager):
    def __init__(self) -> None:
        self.lock: Lock | None = None

    def set(self, lock: Lock) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        if self.lock is not None:
            self.lock.acquire()

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self.lock is not None:
            self.lock.release()


read_lock = _ReadLock()
set_read_lock = read_lock.set


def prefetch_samples(iterable: Iterator[T], pre_load_n: int) -> Iterator[T]:
    while True:
        items = []
        with read_lock:
            for _ in range(pre_load_n):
                try:
                    items.append(next(iterable))
                except StopIteration:
                    break

        if not items:
            break

        yield from items
