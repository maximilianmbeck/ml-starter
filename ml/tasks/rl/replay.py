"""Defines the replay buffer classes for storing sessions from an environment."""

import random
from typing import Callable, Generic, Iterator, TypeVar

from torch.utils.data.dataset import IterableDataset

from ml.tasks.datasets.collate import collate
from ml.utils.data import get_worker_info

T = TypeVar("T")


class ReplaySamples(Generic[T]):
    def __init__(self, samples: list[T]) -> None:
        super().__init__()

        self.samples = samples

    def sample(self, clip_size: int, stride: int = 1, only_last: bool = False) -> list[T]:
        # clip_size = 3, stride = 2: [0, 2, 4]; clip_size = 4, stride = 3: [0, 3, 6, 9]
        length, qsize = (clip_size - 1) * stride + 1, len(self)
        assert length <= qsize, f"{length=} is greater than {qsize=}"
        start = qsize - length if only_last else random.randint(0, qsize - length)
        items = [self[i] for i in range(start, start + length, stride)]
        if len(items) != clip_size:
            raise ValueError(f"Got {len(items)} item(s), expected {clip_size}")
        return items

    def __getitem__(self, index: int) -> T:
        if isinstance(self.samples, list):
            return self.samples[index]
        raise NotImplementedError(f"Unsupported sample type: {type(self.samples)}")

    def __len__(self) -> int:
        if isinstance(self.samples, list):
            return len(self.samples)
        raise NotImplementedError(f"Unsupported sample type: {type(self.samples)}")

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]


class MultiReplaySamples(Generic[T]):
    def __init__(self, samples: list[ReplaySamples[T]]) -> None:
        super().__init__()

        self.samples = samples

    def partition(self, rank: int, world_size: int) -> "MultiReplaySamples":
        return MultiReplaySamples(self.samples[rank::world_size])

    def sample(self, clip_size: int, stride: int = 1, only_last: bool = False) -> list[list[T]]:
        return [s.sample(clip_size, stride=stride, only_last=only_last) for s in self.samples]

    def __getitem__(self, index: int) -> ReplaySamples[T]:
        if isinstance(self.samples, list):
            return self.samples[index]
        raise NotImplementedError(f"Unsupported sample type: {type(self.samples)}")

    def __len__(self) -> int:
        if isinstance(self.samples, list):
            return len(self.samples)
        raise NotImplementedError(f"Unsupported sample type: {type(self.samples)}")

    def __iter__(self) -> Iterator[ReplaySamples[T]]:
        for i in range(len(self)):
            yield self[i]


class ReplayDataset(IterableDataset[T], Generic[T]):
    def __init__(
        self,
        buffer: MultiReplaySamples[T],
        clip_size: int,
        stride: int = 1,
        collate_fn: Callable[[list[T]], T] = collate,  # type: ignore[assignment]
    ) -> None:
        super().__init__()

        self.buffer = buffer
        self._buffer_partitioned: MultiReplaySamples[T] | None = None
        self.clip_size = clip_size
        self.stride = stride
        self.collate_fn = collate_fn

    @property
    def buffer_partitioned(self) -> MultiReplaySamples[T]:
        if self._buffer_partitioned is None:
            raise ValueError("Cannot access partitioned buffer before `__iter__` is called")
        return self._buffer_partitioned

    def __iter__(self) -> Iterator[T]:
        if self._buffer_partitioned is None:
            worker_info = get_worker_info()
            self._buffer_partitioned = self.buffer.partition(worker_info.worker_id, worker_info.num_workers)
        return self

    def __next__(self) -> T:
        samples = random.choice(self.buffer.samples)
        sample = samples.sample(self.clip_size, stride=self.stride, only_last=False)
        return self.collate_fn(sample)
