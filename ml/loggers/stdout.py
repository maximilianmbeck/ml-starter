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
from ml.utils.colors import colorize
from ml.utils.datetime import format_timedelta
from ml.utils.distributed import is_distributed


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

    def write(self, state: State) -> None:
        if not (phase_log_values := self.log_values.get(state.phase)):
            return

        # Gets elapsed time since last write.
        elapsed_time = datetime.datetime.now() - self.start_time
        elapsed_time_str = format_timedelta(elapsed_time)

        def get_section_string(name: str, section: dict[str, Any]) -> str:
            p1, p2 = colorize('"', "magenta"), colorize('"', "blue")
            get_line = lambda kv: f"{p1}{kv[0]}{p2}: {as_str(kv[1](), self.config.precision)}"
            inner_str = ", ".join(map(get_line, sorted(section.items())))
            return '"' + colorize(name, "cyan") + '": {' + inner_str + "}"

        def colorize_phase(phase: Phase) -> str:
            match phase:
                case "train":
                    return colorize(phase, "green", bold=True)
                case "valid":
                    return colorize(phase, "yellow", bold=True)
                case "test":
                    return colorize(phase, "red", bold=True)
                case _:
                    return colorize(phase, "white", bold=True)

        def colorize_time(time: str) -> str:
            return colorize(time, "light-magenta")

        # Writes a log string to stdout.
        log_string = ", ".join(get_section_string(k, v) for k, v in sorted(phase_log_values.items()))
        self.logger.info("%s [%s] {%s}", colorize_phase(state.phase), colorize_time(elapsed_time_str), log_string)

    def clear(self, state: State) -> None:
        if state.phase in self.log_values:
            self.log_values[state.phase].clear()

    def default_write_every_n_seconds(self, phase: Phase) -> float:
        return 10.0 if is_distributed() else 1.0
