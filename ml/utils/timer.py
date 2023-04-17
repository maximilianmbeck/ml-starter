import errno
import functools
import logging
import os
import signal
import sys
import time
from threading import Thread
from typing import Any, Callable, TypeVar

from ml.utils.colors import colorize

timer_logger: logging.Logger = logging.getLogger(__name__)

TimeoutFunc = TypeVar("TimeoutFunc", bound=Callable[..., Any])


@functools.lru_cache
def allow_spinners() -> bool:
    return (
        "PYTEST_CURRENT_TEST" not in os.environ
        and "pytest" not in sys.modules
        and sys.stdout.isatty()
        and os.environ.get("TERM") != "dumb"
    )


class Spinner:
    def __init__(self, text: str) -> None:
        self._text = colorize(text, "grey")
        self._spinner_thread: Thread | None = None
        self._spinner_running = False
        self._spinner_stop = False

    def start(self) -> None:
        if self._spinner_thread is not None:
            raise RuntimeError("Spinner already started")

        self._spinner_thread = Thread(target=self._spinner)
        self._spinner_thread.start()

    def stop(self) -> None:
        if self._spinner_thread is None:
            raise RuntimeError("Spinner not started")

        self._spinner_stop = True
        self._spinner_thread.join()
        self._spinner_thread = None

    def _spinner(self) -> None:
        self._spinner_running = True
        chars = [colorize(c, "light-yellow") for c in ("|", "/", "-", "\\")]
        start_time = time.time()
        max_line_len = 0
        while not self._spinner_stop:
            for char in chars:
                elapsed_secs = time.time() - start_time
                line = f"[ {char} {elapsed_secs:.1f} ] {self._text}\r"
                max_line_len = max(max_line_len, len(line))
                sys.stderr.write(line)
                sys.stderr.flush()
                time.sleep(0.1)
        sys.stderr.write(" " * max_line_len + "\r")
        self._spinner_running = False


class Timer:
    """Defines a simple timer for logging an event."""

    def __init__(
        self,
        description: str,
        min_seconds_to_print: float = 1.0,
        logger: logging.Logger | None = None,
        spinner: bool = False,
    ) -> None:
        self.description = description
        self.min_seconds_to_print = min_seconds_to_print
        self._start_time: float | None = None
        self._elapsed_time: float | None = None
        self._logger = timer_logger if logger is None else logger
        self._spinner = Spinner(description) if spinner and allow_spinners() else None

    @property
    def elapsed_time(self) -> float:
        assert (elapsed_time := self._elapsed_time) is not None
        return elapsed_time

    def __enter__(self) -> "Timer":
        self._start_time = time.time()
        if self._spinner is not None:
            self._spinner.start()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        assert self._start_time is not None
        self._elapsed_time = time.time() - self._start_time
        if self._elapsed_time > self.min_seconds_to_print:
            self._logger.warning("Finished %s in %.3g seconds", self.description, self._elapsed_time)
        if self._spinner is not None:
            self._spinner.stop()


def timeout(seconds: int, error_message: str = os.strerror(errno.ETIME)) -> Callable[[TimeoutFunc], TimeoutFunc]:
    """Decorator for timing out long-running functions.

    Note that this function won't work on Windows.

    Usage:
        try:
            @timeout(5)
            def long_running_function():
                ...
        except TimeoutError:
            handle_timeout()

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
