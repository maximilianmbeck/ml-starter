import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch import Tensor

from ml.core.config import conf_field
from ml.core.registry import register_logger
from ml.core.state import Phase, State
from ml.loggers.base import BaseLogger, BaseLoggerConfig
from ml.utils.datetime import format_timedelta


@dataclass
class StdoutLoggerConfig(BaseLoggerConfig):
    precision: int = conf_field(4, help="Scalar precision to log")


def format_number(value: int | float, precision: int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}g}"


def as_str(value: Any, precision: int) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Tensor):
        value = value.detach().float().cpu().item()
    if isinstance(value, (int, float)):
        return format_number(value, precision)
    raise TypeError(f"Unexpected log type: {type(value)}")


@register_logger("stdout", StdoutLoggerConfig)
class StdoutLogger(BaseLogger[StdoutLoggerConfig]):
    def __init__(self, config: StdoutLoggerConfig) -> None:
        super().__init__(config)

        self.log_values: dict[Phase, dict[str, dict[str, Callable[[], Any]]]] = {}
        self.logger = logging.getLogger("stdout")

    def initialize(self, log_directory: Path) -> None:
        super().initialize(log_directory)

        log_directory.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_directory / "stdout.log")
        self.logger.addHandler(file_handler)
        self.logger.debug("Finished initializing logger")

    def get_log_dict(self, state: State, namespace: str | None) -> dict[str, Callable[[], Any]]:
        if namespace is None:
            namespace = "default"
        if state.phase not in self.log_values:
            self.log_values[state.phase] = {}
        if namespace not in self.log_values[state.phase]:
            self.log_values[state.phase][namespace] = {}
        return self.log_values[state.phase][namespace]

    def log_scalar(self, key: str, value: Callable[[], int | float | Tensor], state: State, namespace: str) -> None:
        self.get_log_dict(state, namespace)[key] = value

    def log_string(self, key: str, value: Callable[[], str], state: State, namespace: str) -> None:
        self.get_log_dict(state, namespace)[key] = value

    def log_config(self, config: dict[str, int | float | str | bool], metrics: dict[str, int | float]) -> None:
        for k, v in sorted(metrics.items()):
            self.logger.info("%s: %s", k, as_str(v, self.config.precision))

    def write(self, state: State) -> None:
        if not (phase_log_values := self.log_values.get(state.phase)):
            return

        # Gets elapsed time since last write.
        elapsed_time = datetime.datetime.now() - self.start_time
        elapsed_time_str = format_timedelta(elapsed_time)

        def get_section_string(name: str, section: dict[str, Any]) -> str:
            inner_str = ", ".join(f'"{k}": {as_str(v(), self.config.precision)}' for k, v in sorted(section.items()))
            return '"' + name + '": {' + inner_str + "}"

        # Writes a log string to stdout.
        log_string = ", ".join(get_section_string(k, v) for k, v in sorted(phase_log_values.items()))
        self.logger.info("%s [%s] {%s}", state.phase, elapsed_time_str, log_string)

    def clear(self, state: State) -> None:
        if state.phase in self.log_values:
            self.log_values[state.phase].clear()
