import errno
import functools
import logging
import os
import signal
import sys
import threading
import time
import warnings
from threading import Thread
from typing import Any, Callable, TypeVar

from ml.utils.colors import colorize
from ml.utils.distributed import is_master

timer_logger: logging.Logger = logging.getLogger(__name__)

TimeoutFunc = TypeVar("TimeoutFunc", bound=Callable[..., Any])


@functools.lru_cache
def allow_spinners() -> bool:
    return (
        "PYTEST_CURRENT_TEST" not in os.environ
        and "pytest" not in sys.modules
        and sys.stdout.isatty()
        and os.environ.get("TERM") != "dumb"
        and is_master()
    )


class Spinner:
    def __init__(self, text: str | None = None) -> None:
        self._text = "" if text is None else text
        self._spinner_stop = False
        self._spinner_close = False
        self._flag = threading.Event()
        self._thread = Thread(target=self._spinner, daemon=True)
        self._thread.start()

        # If we're in a breakpoint, we want to close the spinner when we exit
        # the breakpoint.
        self._original_breakpointhook = sys.breakpointhook

    def _breakpointhook(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("Breakpoint hit inside spinner; run `up 1` to see where it was hit")
        self.stop()
        sys.breakpointhook(*args, **kwargs)

    def set_text(self, text: str) -> "Spinner":
        self._text = colorize(text, "grey")
        return self

    def start(self) -> None:
        self._spinner_stop = False
        self._flag.set()
        sys.breakpointhook = self._breakpointhook

    def stop(self) -> None:
        self._spinner_stop = True
        sys.breakpointhook = self._original_breakpointhook

    def close(self) -> None:
        self.stop()
        self._spinner_close = True
        self._thread.join()

    def _spinner(self) -> None:
        chars = [colorize(c, "light-yellow") for c in ("|", "/", "-", "\\")]
        max_line_len = 0
        while not self._spinner_close:
            self._flag.wait()
            start_time = time.time()
            while not self._spinner_stop:
                for char in chars:
                    elapsed_secs = time.time() - start_time
                    line = f"[ {char} {elapsed_secs:.1f} ] {self._text}\r"
                    max_line_len = max(max_line_len, len(line))
                    sys.stderr.write(line)
                    sys.stderr.flush()
                    time.sleep(0.05)
            sys.stderr.write(" " * max_line_len + "\r")
            self._flag.clear()


@functools.lru_cache
def spinner() -> Spinner:
    return Spinner()


class Timer:
    """Defines a simple timer for logging an event."""

    def __init__(
        self,
        description: str,
        min_seconds_to_print: float = 5.0,
        logger: logging.Logger | None = None,
        spinner: bool = False,
    ) -> None:
        self.description = description
        self.min_seconds_to_print = min_seconds_to_print
        self._start_time: float | None = None
        self._elapsed_time: float | None = None
        self._logger = timer_logger if logger is None else logger
        self._use_spinner = spinner and allow_spinners()

    @property
    def elapsed_time(self) -> float:
        assert (elapsed_time := self._elapsed_time) is not None
        return elapsed_time

    def __enter__(self) -> "Timer":
        self._start_time = time.time()
        if self._use_spinner:
            spinner().set_text(self.description).start()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        assert self._start_time is not None
        self._elapsed_time = time.time() - self._start_time
        if self._elapsed_time > self.min_seconds_to_print:
            self._logger.warning("Finished %s in %.3g seconds", self.description, self._elapsed_time)
        spinner().stop()


def timeout(seconds: int, error_message: str = os.strerror(errno.ETIME)) -> Callable[[TimeoutFunc], TimeoutFunc]:
    """Decorator for timing out long-running functions.

    Note that this function won't work on Windows.

    Args:
        seconds: Timeout after this many seconds
        error_message: Error message to pass to TimeoutError

    Returns:
        Decorator function
    """

    def decorator(func: TimeoutFunc) -> TimeoutFunc:
        def _handle_timeout(*_: Any) -> None:
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper  # type: ignore

    return decorator
