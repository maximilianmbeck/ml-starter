"""Runs the training script."""

import datetime
import logging
from contextlib import contextmanager
from typing import Iterator

from omegaconf import DictConfig

from ml.core.registry import Objects
from ml.utils.colors import colorize
from ml.utils.datetime import format_datetime, format_timedelta
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)


def train_main(config: DictConfig, objs: Objects | None = None) -> None:
    """Runs the training loop.

    Args:
        config: The configuration object.
        objs: Objects which have already been parsed from the config.
    """
    with Timer("setting random seed", spinner=True):
        from ml.utils.random import set_random_seed

        set_random_seed()

    objs = Objects.parse_raw_config(config, objs)

    # Checks that the config has the right keys for training.
    assert (model := objs.model) is not None
    assert (task := objs.task) is not None
    assert (optimizer := objs.optimizer) is not None
    assert (lr_scheduler := objs.lr_scheduler) is not None
    assert (trainer := objs.trainer) is not None

    @contextmanager
    def log_info_wrapper() -> Iterator[None]:
        start_time = datetime.datetime.now()
        try:
            yield
        finally:
            end_time = datetime.datetime.now()
            delta = end_time - start_time
            stats: dict[str, str] = {
                "Start Time": format_datetime(start_time),
                "End Time": format_datetime(end_time),
                "Duration": format_timedelta(delta),
            }
            if trainer is not None:
                stats["Experiment Directory"] = str(trainer.exp_dir)
            stats_str = "".join(f"\n ↪ {colorize(k, 'magenta')}: {colorize(v, 'cyan')}" for k, v in stats.items())
            logger.info("Finished training. Stats:%s", stats_str)

    # Runs the training loop.
    with log_info_wrapper():
        trainer.train(model, task, optimizer, lr_scheduler)
