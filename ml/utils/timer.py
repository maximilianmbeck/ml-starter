import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Timer:
    """Defines a simple timer for logging an event."""

    def __init__(self, description: str, min_seconds_to_print: float = 1.0) -> None:
        self.description = description
        self.min_seconds_to_print = min_seconds_to_print
        self._start_time: Optional[float] = None
        self._elapsed_time: Optional[float] = None

    @property
    def elapsed_time(self) -> float:
        assert (elapsed_time := self._elapsed_time) is not None
        return elapsed_time

    def __enter__(self) -> "Timer":
        self._start_time = time.time()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        assert self._start_time is not None
        self._elapsed_time = time.time() - self._start_time
        if self._elapsed_time > self.min_seconds_to_print:
            logger.warning("Finished %s in %.3g seconds", self.description, self._elapsed_time)
