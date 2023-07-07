"""Defines a metered logger.

This logger keeps track of statistics of logged values. It is useful for
getting global statistics during evaluation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from torch import Tensor

from ml.core.registry import register_logger
from ml.core.state import Phase, State
from ml.loggers.base import BaseLogger, BaseLoggerConfig
from ml.utils.meter import Meter


def get_value(value: int | float | Tensor) -> int | float:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Tensor):
        return value.detach().float().cpu().item()
    raise TypeError(f"Unexpected log type: {type(value)}")


@dataclass
class MeterLoggerConfig(BaseLoggerConfig):
    pass


@register_logger("meter", MeterLoggerConfig)
class MeterLogger(BaseLogger[MeterLoggerConfig]):
    def __init__(self, config: MeterLoggerConfig) -> None:
        super().__init__(config)

        self.meters: dict[Phase, dict[str, dict[str, Meter]]] = {}

    def get_meter(self, state: State, key: str, namespace: str | None) -> Meter:
        if namespace is None:
            namespace = "default"
        if state.phase not in self.meters:
            self.meters[state.phase] = {}
        if namespace not in self.meters[state.phase]:
            self.meters[state.phase][namespace] = {}
        return self.meters[state.phase][namespace][key]

    def log_scalar(self, key: str, value: Callable[[], int | float | Tensor], state: State, namespace: str) -> None:
        self.get_meter(state, key, namespace).add(get_value(value()))

    def iter_meters(self) -> Iterable[Meter]:
        for v in self.meters.values():
            for vv in v.values():
                for vvv in vv.values():
                    yield vvv

    def get_value_dict(self) -> dict[str, int | float]:
        # First, reduces the meters.
        works: list[Any] = []
        for meter in self.iter_meters():
            works.extend(meter.reduce())
        for work in works:
            work.wait()

        # Next, builds the output dictionaries.
        out_dict: dict[str, int | float] = {}
        for phase, phase_meters in self.meters.items():
            for namespace, namespace_meters in phase_meters.items():
                for key, meter in namespace_meters.items():
                    abs_key = f"{phase}/{namespace}/{key}"
                    if meter.min_val is not None:
                        out_dict[f"{abs_key}/min"] = meter.min_val
                    if meter.max_val is not None:
                        out_dict[f"{abs_key}/max"] = meter.max_val
                    if meter.mean_val is not None:
                        out_dict[f"{abs_key}/mean"] = meter.mean_val
        return out_dict

    def write(self, state: State) -> None:
        pass

    def default_write_every_n_seconds(self, state: State) -> float:
        return 0.0
