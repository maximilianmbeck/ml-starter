"""Defines error handling wrappers for datasets.

The worst feeling in the world is when you're training a model and it crashes
after 10 hours of training. This module defines some error handling wrappers
for datasets which will catch errors and log them (in batches).
"""

import bdb
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, TypeVar

from torch.utils.data.dataset import Dataset, IterableDataset

from ml.core.config import conf_field
from ml.utils.data import get_worker_info

logger: logging.Logger = logging.getLogger(__name__)

BatchT = TypeVar("BatchT")


def get_loc(num_excs: int = 1) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None or (exc_tb := exc_tb.tb_next) is None:
        return "unknown"
    exc_strs: list[str] = []
    for _ in range(num_excs):
        exc_strs += [f"{exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno}"]
        if (exc_tb := exc_tb.tb_next) is None:
            break
    return "\n".join(exc_strs)


@dataclass
class ErrorHandlingConfig:
    enabled: bool = conf_field(True, help="Is error handling enabled?")
    maximum_exceptions: int = conf_field(10, help="Maximum number of errors to encounter")
    backoff_after: int = conf_field(5, help="Start to do a sleeping backoff after this many exceptions")
    sleep_backoff: float = conf_field(0.1, help="Sleep backoff amount")
    sleep_backoff_power: float = conf_field(2.0, help="How much to multiply backoff for each successive exception")
    log_full_exception: bool = conf_field(False, help="Log the full exception message for each exception")
    flush_exception_summary_every: int = conf_field(500, help="How often to flush exception summary")
    report_top_n_exception_types: int = conf_field(5, help="Number of exceptions to summarize")
    exception_location_traceback_depth: int = conf_field(3, help="Traceback length for the exception location")


class ExceptionSummary:
    def __init__(self, flush_every: int, summary_length: int = 5) -> None:
        self.steps = 0
        self.total_exceptions = 0
        self.flush_every = flush_every
        self.summary_length = summary_length
        self.exceptions: Counter[str] = Counter()
        self.exception_classes: Counter[str] = Counter()
        self.exception_locs: Counter[str] = Counter()
        self.last_exception: Exception | None = None

    def add_exception(self, exc: Exception, loc: str) -> None:
        self.last_exception = exc
        self.exceptions[f"{exc.__class__.__name__}: {exc}"] += 1
        self.exception_classes[exc.__class__.__name__] += 1
        self.exception_locs[loc] += 1
        self.total_exceptions += 1

    def step(self) -> None:
        if self.steps >= self.flush_every:
            self.flush()
        self.steps += 1

    def summary(self) -> str:
        lines: list[str] = []

        def get_segment_header(header: str) -> list[str]:
            return [
                f"| {header:60s} | {'Count':10s} | {'Percent':10s} |",
                f"| {'-' * 60} | {'-' * 10} | {'-' * 10} |",
            ]

        def get_log_line(ks: str, v: int) -> str:
            chunks = [k[i : i + 60] for k in ks.split("\n") for i in range(0, len(k), 60)]
            v_int, v_prct = f"{v}", f"{int(v * 100 / self.steps)} %"
            log_lines = [f"| {chunks[0]:60s} | {v_int:10s} | {v_prct:10s} |"]
            for chunk in chunks[1:]:
                log_lines += [f"| {chunk:60s} | {'':10s} | {'':10s} |"]
            return "\n".join(log_lines)

        def get_line_break() -> str:
            return f"=={'=' * 60}==={'=' * 10}==={'=' * 10}=="

        # Logs the unique exception strings.
        lines += [get_line_break()]
        lines += get_segment_header("Exception (by message)")
        for k, v in self.exceptions.most_common(self.summary_length):
            lines += [get_log_line(k, v)]

        # Logs the individual exception classes.
        lines += [get_line_break()]
        lines += get_segment_header("Exception (by class)")
        for k, v in self.exception_classes.most_common(self.summary_length):
            lines += [get_log_line(k, v)]

        # Logs by line number.
        lines += [get_line_break()]
        lines += get_segment_header("Exception (by location)")
        for k, v in self.exception_locs.most_common(self.summary_length):
            lines += [get_log_line(k, v)]

        # Logs the total number of exceptions.
        lines += [get_line_break()]
        lines += [get_log_line("Total", self.total_exceptions)]
        lines += [get_log_line("Steps", self.steps)]
        lines += [get_line_break()]

        return "\n".join(lines)

    def flush(self) -> None:
        worker_info = get_worker_info()
        if worker_info.worker_id and self.total_exceptions > 0:
            logger.info("Exception summary:\n\n%s\n", self.summary())
        self.exceptions.clear()
        self.exception_classes.clear()
        self.exception_locs.clear()
        self.steps = 0
        self.total_exceptions = 0


class ErrorHandlingDataset(Dataset[BatchT]):
    """Defines a wrapper for safely handling errors."""

    dataset: Dataset[BatchT]
    config: ErrorHandlingConfig

    def __init__(self, dataset: Dataset[BatchT], config: ErrorHandlingConfig) -> None:
        super().__init__()

        self.dataset = dataset
        self.config = config
        self.exc_summary = ExceptionSummary(
            flush_every=config.flush_exception_summary_every,
            summary_length=config.report_top_n_exception_types,
        )

    def __getitem__(self, index: int) -> BatchT:
        num_exceptions = 0
        backoff_time = self.config.sleep_backoff
        self.exc_summary.step()
        while num_exceptions < self.config.maximum_exceptions:
            try:
                return self.dataset[index]
            except bdb.BdbQuit as e:
                logger.info("User interrupted debugging session; aborting")
                raise e
            except Exception as e:
                if self.config.log_full_exception:
                    logger.exception("Caught exception on index %d", index)
                self.exc_summary.add_exception(e, get_loc(self.config.exception_location_traceback_depth))
                index = random.randint(0, len(self) - 1)
            num_exceptions += 1
            if num_exceptions > self.config.backoff_after:
                logger.error(
                    "Encountered %d exceptions for a single index, backing off for %f seconds",
                    num_exceptions,
                    backoff_time,
                )
                time.sleep(backoff_time)
                backoff_time *= self.config.sleep_backoff_power
        exc_message = f"Reached max exceptions {self.config.maximum_exceptions}\n{self.exc_summary.summary()}"
        if self.exc_summary.last_exception is None:
            raise RuntimeError(exc_message)
        raise RuntimeError(exc_message) from self.exc_summary.last_exception

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__"):
            return self.dataset.__len__()
        raise NotImplementedError("Base dataset doesn't implemenet `__len__`")


class ErrorHandlingIterableDataset(IterableDataset[BatchT]):
    """Defines a wrapper for safely handling errors in iterable datasets."""

    dataset: IterableDataset[BatchT]
    iter: Iterator[BatchT]

    def __init__(self, dataset: IterableDataset[BatchT], config: ErrorHandlingConfig) -> None:
        super().__init__()

        self.iteration = 0
        self.dataset = dataset
        self.config = config
        self.exc_summary = ExceptionSummary(
            flush_every=config.flush_exception_summary_every,
            summary_length=config.report_top_n_exception_types,
        )

        self._configured_logging = False

    def __iter__(self) -> Iterator[BatchT]:
        self.iter = self.dataset.__iter__()
        self.iteration = 0
        return self

    def __next__(self) -> BatchT:
        num_exceptions = 0
        backoff_time = self.config.sleep_backoff
        self.exc_summary.step()
        self.iteration += 1
        while num_exceptions < self.config.maximum_exceptions:
            try:
                return self.iter.__next__()
            except bdb.BdbQuit as e:
                logger.info("User interrupted debugging session; aborting")
                raise e
            except StopIteration as e:
                raise e
            except Exception as e:
                if self.config.log_full_exception:
                    logger.exception("Caught exception on iteration %d", self.iteration)
                self.exc_summary.add_exception(e, get_loc(self.config.exception_location_traceback_depth))
            num_exceptions += 1
            if num_exceptions > self.config.backoff_after:
                logger.error(
                    "Encountered %d exceptions for a single index, backing off for %f seconds",
                    num_exceptions,
                    backoff_time,
                )
                time.sleep(backoff_time)
                backoff_time *= self.config.sleep_backoff_power
        raise RuntimeError(f"Reached max exceptions {self.config.maximum_exceptions}\n{self.exc_summary.summary()}")


def get_error_handling_dataset(dataset: Dataset[BatchT], config: ErrorHandlingConfig) -> Dataset[BatchT]:
    """Returns a dataset which wraps the base dataset and handles errors.

    Args:
        dataset: The dataset to handle errors for
        config: An associated config, describing which errors to handle

    Returns:
        The wrapped dataset, which catches some errors

    Raises:
        NotImplementedError: If the dataset type is not supported
    """
    if isinstance(dataset, IterableDataset):
        return ErrorHandlingIterableDataset(dataset, config)
    elif isinstance(dataset, Dataset):
        return ErrorHandlingDataset(dataset, config)
    raise NotImplementedError(f"Unexpected type: {dataset}")
