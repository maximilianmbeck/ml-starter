"""Holds some common types which are used in various places."""

from typing import Any, Callable, Iterable

from torch import Tensor

Params = Iterable[Tensor] | Iterable[dict[str, Any]]

LossClosure = Callable[[], float]
OptLossClosure = LossClosure | None
Betas2 = tuple[float, float]
State = dict[str, Any]
OptFloat = float | None
Nus2 = tuple[float, float]
