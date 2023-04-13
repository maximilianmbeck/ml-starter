import contextlib
import functools
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)

from ml.core.common_types import Batch
from ml.utils.timer import Timer


def allow_nonblocking(device_a: torch.device, device_b: torch.device) -> bool:
    return device_a.type in ("cpu", "cuda") and device_b.type in ("cpu", "cuda")


def get_tasks_outstanding(dataloader_iter: _BaseDataLoaderIter) -> int:
    if isinstance(dataloader_iter, _MultiProcessingDataLoaderIter):
        try:
            return dataloader_iter._worker_result_queue.qsize()
        except NotImplementedError:
            return -2
    return -1


def recursive_apply(item: Any, func: Callable[[Tensor], Tensor]) -> Any:
    """Applies a function recursively to tensors in an item.

    Args:
        item: The item to apply the function to
        func: The function to apply (for the tensor)

    Returns:
        The same item, with the function applied
    """

    if isinstance(item, (str, int, float)):
        return item
    if isinstance(item, np.ndarray):
        item = torch.from_numpy(item)
    if isinstance(item, Tensor):
        return func(item)
    if is_dataclass(item):
        return item.__class__(**{k: recursive_apply(v, func) for k, v in item.__dict__.items()})
    if isinstance(item, Mapping):
        return {k: recursive_apply(v, func) for k, v in item.items()}
    if isinstance(item, Sequence):
        return [recursive_apply(i, func) for i in item]
    return item


class Prefetcher(Iterable[Batch]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(
        self,
        to_device_func: Callable[[Any], Any],
        dataloader: DataLoader,
        raise_stop_iter: bool = False,
    ) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.raise_stop_iter = raise_stop_iter
        self.dataloader_iter = iter(self.dataloader)
        self.next_sample = None
        self.get_batch_time = 0.0
        self.to_device_time = 0.0
        self.num_queued_samples = -1

    def prefetch(self) -> None:
        try:
            with Timer("getting sample from dataloader") as timer:
                next_sample = next(self.dataloader_iter)
            self.get_batch_time = timer.elapsed_time
            with Timer("moving sample to device") as timer:
                self.next_sample = self.to_device_func(next_sample)
            self.to_device_time = timer.elapsed_time
            self.num_queued_samples = get_tasks_outstanding(self.dataloader_iter)
        except StopIteration:
            self.next_sample = None

    def recursive_chunk(self, item: Any, chunks: int) -> list[Any]:
        """Applies a function recursively to tensors in an item.

        Args:
            item: The item to apply the function to
            chunks: The number of output chunks

        Returns:
            The item, split into the requested number of chunks
        """

        if isinstance(item, (str, int, float)):
            return [item] * chunks
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        if isinstance(item, Tensor):
            item_chunk_list = list(item.chunk(chunks, dim=0))
            assert len(item_chunk_list) == chunks, f"{len(item_chunk_list)=} != {chunks=}"
            return item_chunk_list
        if is_dataclass(item):
            item_chunk_dict = {k: self.recursive_chunk(v, chunks) for k, v in item.__dict__.items()}
            return [item.__class__(**{k: v[i] for k, v in item_chunk_dict.items()}) for i in range(chunks)]
        if isinstance(item, Mapping):
            item_chunk_dict = {k: self.recursive_chunk(v, chunks) for k, v in item.items()}
            return [{k: v[i] for k, v in item_chunk_dict.items()} for i in range(chunks)]
        if isinstance(item, Sequence):
            item_chunk_lists = [self.recursive_chunk(i, chunks) for i in item]
            return [[k[i] for k in item_chunk_lists] for i in range(chunks)]
        return item

    @classmethod
    def recursive_apply(cls, item: Any, func: Callable[[Tensor], Tensor]) -> Any:
        return recursive_apply(item, func)

    def __iter__(self) -> Iterator[Batch]:
        self.prefetch()

        try:
            while True:
                if self.next_sample is None:
                    raise StopIteration
                sample = self.next_sample
                self.prefetch()
                yield sample

        except StopIteration:
            # Resets the dataloader if the iteration has completed.
            self.dataloader_iter = iter(self.dataloader)
            if self.raise_stop_iter:
                raise


class InfinitePrefetcher(Iterable[Batch]):
    def __init__(self, prefetcher: Prefetcher[Batch]) -> None:
        self.prefetcher = prefetcher

    def __iter__(self) -> Iterator[Batch]:
        while True:
            for batch in self.prefetcher:
                yield batch


class BaseDevice(ABC):
    """Base mixin for different trainer device types."""

    @classmethod
    @abstractmethod
    def has_device(cls) -> bool:
        """Detects whether or not the device is available.

        Returns:
            If the device is available
        """

    @classmethod
    @abstractmethod
    def get_device(cls) -> torch.device:
        """Returns the device, for instantiating new tensors.

        Returns:
            The device
        """

    @classmethod
    @abstractmethod
    def get_floating_point_type(cls) -> torch.dtype:
        """Returns the default floating point type to use.

        Returns:
            The dtype
        """

    @classmethod
    @abstractmethod
    def get_torch_compile_backend(cls) -> str | Callable:
        """Returns the backend to use for Torch compile.

        Returns:
            The backend
        """

    @classmethod
    def sample_to_device(cls, sample: Any) -> Any:
        device = cls.get_device()
        dtype_fp = cls.get_floating_point_type()
        return Prefetcher.recursive_apply(
            sample,
            lambda t: t.to(
                device,
                dtype_fp if t.is_floating_point() else t.dtype,
                non_blocking=allow_nonblocking(t.device, device),
            ),
        )

    @classmethod
    def get_prefetcher(cls, dataloader: DataLoader) -> Prefetcher:
        return Prefetcher(functools.partial(cls.sample_to_device), dataloader)

    @classmethod
    def module_to(cls, module: nn.Module) -> None:
        module.to(cls.get_device(), cls.get_floating_point_type())

    @classmethod
    def tensor_to(cls, tensor: Tensor) -> Tensor:
        device = cls.get_device()
        if tensor.is_floating_point():
            return tensor.to(device, cls.get_floating_point_type())
        return tensor.to(device)

    @classmethod
    def recursive_apply(cls, item: Any) -> Any:
        def func(i: Any) -> Any:
            if isinstance(i, Tensor):
                return cls.tensor_to(i)
            if isinstance(i, nn.Module):
                cls.module_to(i)
                return i
            return i

        return recursive_apply(item, func)

    @classmethod
    def autocast_context(cls, enabled: bool = True) -> ContextManager:
        device_type = cls.get_device().type
        if device_type == "mps":
            device_type = "cpu"
        if device_type not in ("cpu", "cuda"):
            return contextlib.nullcontext()
        return torch.autocast(device_type, enabled=enabled)
