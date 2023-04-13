import atexit
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Generic, TypeVar

from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MonitorProcessConfig(BaseTrainerConfig):
    pass


MonitorProcessConfigT = TypeVar("MonitorProcessConfigT", bound=MonitorProcessConfig)


class MonitorProcessMixin(
    BaseTrainer[MonitorProcessConfigT, ModelT, TaskT],
    Generic[MonitorProcessConfigT, ModelT, TaskT],
):
    """Defines a trainer mixin for getting CPU statistics."""

    def __init__(self, config: MonitorProcessConfigT) -> None:
        super().__init__(config)

        self._mp_manager = mp.Manager()
        atexit.register(self._mp_manager.shutdown)
