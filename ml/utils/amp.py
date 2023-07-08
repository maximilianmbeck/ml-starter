"""Helper functions for mixed-precision training."""

import contextlib
import functools
from types import TracebackType
from typing import Any, Callable, ContextManager, Generic, ParamSpec, TypeVar

import torch

from ml.utils.containers import recursive_apply
from ml.utils.device.auto import AutoDevice

T = TypeVar("T")
P = ParamSpec("P")

SCRIPT_UNSUPPORTED_STR = "@autocast() decorator is not supported in script mode"


def _autocast_device_types() -> list[str]:
    device_types = ["cpu"]
    if torch.cuda.is_available():
        device_types.append("cuda")
    if hasattr(torch, "xpu"):
        device_types.append("xpu")
    if hasattr(torch, "hpu"):
        device_types.append("hpu")
    return device_types


class autocast_all:  # noqa: N801
    def __init__(
        self,
        device_types: list[str] | None = None,
        enabled: bool = True,
        cache_enabled: bool | None = None,
    ) -> None:
        self.enabled = enabled

        self.device_types = _autocast_device_types() if device_types is None else device_types
        self.autocast_ctxs = [
            torch.autocast(device_type, enabled=enabled, cache_enabled=cache_enabled)
            for device_type in self.device_types
        ]

    def __enter__(self) -> None:
        for autocast_ctx in self.autocast_ctxs:
            autocast_ctx.__enter__()

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        for autocast_ctx in self.autocast_ctxs:
            autocast_ctx.__exit__(_t, _e, _tr)

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def decorate_autocast(*args: P.args, **kwargs: P.kwargs) -> T:
            with self:
                return func(*args, **kwargs)

        decorate_autocast.__script_unsupported = SCRIPT_UNSUPPORTED_STR
        return decorate_autocast


@functools.lru_cache()
def default_device() -> str:
    type = AutoDevice.detect_device().get_device().type
    if type not in ("cpu", "cuda"):
        type = "cpu"
    return type


@functools.lru_cache
def default_dtype(enabled: bool) -> torch.dtype:
    default_dtype = AutoDevice.detect_device().get_floating_point_type()
    if not enabled:
        return torch.float32 if default_dtype in (torch.float16, torch.bfloat16) else default_dtype
    return default_dtype


class autocast_tensors(Generic[T]):  # noqa: N801
    """Defines a context manager for enabling or disabling autocasting.

    This context manager simultaneously converts a tensor or container of
    tensors to the dtype that the device expects. For example, if enabling
    autocast, it will convert the tensor or tensors to whatever the default
    floating point type is for the device (typically FP16 or BF16). If
    disabling, it will convert the tensor or tensors to FP32.

    Parameters:
        xs: The tensor or container of tensors to autocast.
        device_type: The device type to use for autocasting. If not specified,
            the default device type will be used.
        dtype: The dtype to use for autocasting. If not specified, the default
            dtype will be used.
        enabled: Whether to enable or disable autocasting.
        cache_enabled: Whether to enable or disable the cache for autocasting.
            If not specified, the default cache setting will be used.
    """

    def __init__(
        self,
        xs: T | None = None,
        device_type: str | None = None,
        dtype: torch.dtype | None = None,
        enabled: bool = True,
        cache_enabled: bool | None = None,
    ) -> None:
        if device_type is None:
            device_type = default_device()
        if dtype is None:
            dtype = default_dtype(enabled)
        self.device_type = device_type
        self.dtype = dtype
        self.xs = xs
        self.enabled = enabled
        self.autocast_ctx: ContextManager
        if device_type == "cpu" and dtype != torch.bfloat16:
            self.autocast_ctx = contextlib.nullcontext()
        else:
            self.autocast_ctx = torch.autocast(device_type, dtype, enabled=enabled, cache_enabled=cache_enabled)

    def apply(self, xs: Any) -> Any:  # noqa: ANN401
        return recursive_apply(xs, lambda t: t.to(self.dtype))

    def __enter__(self) -> T:
        self.autocast_ctx.__enter__()
        return self.apply(self.xs)

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        self.autocast_ctx.__exit__(_t, _e, _tr)

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def decorate_autocast(*args: P.args, **kwargs: P.kwargs) -> T:
            with self:
                return func(*self.apply(args), **self.apply(kwargs))

        decorate_autocast.__script_unsupported = SCRIPT_UNSUPPORTED_STR
        return decorate_autocast
