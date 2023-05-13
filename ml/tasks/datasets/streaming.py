"""Defines a dataset which combines many streaming datasets.

This dataset takes a set of child iterable datasets and iterates from
them infinitely. When a child dataset is exhausted, it is returned to
the reservoir and restarted, while another dataset is chosen.
"""

import logging
import random
from typing import Collection, Generic, Iterator, TypeVar

from torch.utils.data.dataset import IterableDataset

from ml.utils.data import get_worker_info

logger: logging.Logger = logging.getLogger(__name__)

Batch = TypeVar("Batch")


class StreamingDataset(IterableDataset[tuple[int, Batch]], Generic[Batch]):
    """Defines a dataset which combines many streaming datasets.

    This dataset takes a set of child iterable datasets and iterates from
    them infinitely. When a child dataset is exhausted, it is returned to
    the reservoir and restarted, while another dataset is chosen.

    An example usage for this dataset is to get samples from many videos,
    where each sub-dataset yields video samples. This way the child dataset
    can be used to run inference on a single video, while the parent
    streaming dataset can be used to train on a mixture of videos. The
    child dataset can then be optimized to make video loading times fast.
    """

    def __init__(self, datasets: Collection[IterableDataset[Batch]], max_simultaneous: int) -> None:
        """Initializes a new streaming dataset.

        Args:
            datasets: The sub-datasets to iterate from
            max_simultaneous: The maximum number of simultaneous datasets to
                iterate from. Increasing this number increases the dataset
                diversity but also increases memory usage as samples need to be
                stored in memory

        Raises:
            ValueError: If no datasets are provided
        """
        super().__init__()

        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")

        self.datasets = list(datasets)
        self.max_simultaneous = max_simultaneous

    worker_datasets: dict[int, IterableDataset[Batch]]
    iterators: dict[int, Iterator[Batch]]
    reservoir: list[int]
    reservoir_pointer: int

    def __iter__(self) -> Iterator[tuple[int, Batch]]:
        worker_info = get_worker_info()
        dataset_ids = list(range(len(self.datasets)))
        dataset_ids = dataset_ids[worker_info.worker_id :: worker_info.num_workers]

        # Gets the subset of worker dataset for this iterator.
        self.worker_datasets = {i: self.datasets[i] for i in dataset_ids}
        if len(self.worker_datasets) == 0:
            raise ValueError(
                f"Worker {worker_info.worker_id} doesn't have any datasets; "
                f"consider reducing the worker count to {len(self.datasets)}"
            )

        # Creates a reservoir of available IDs, and a dict of active iterators.
        self.iterators = {}
        self.reservoir = list(dataset_ids)
        random.shuffle(self.reservoir)
        self.reservoir_pointer = 0

        return self

    def swap_reservoir(self, a: int, b: int) -> None:
        self.reservoir[a], self.reservoir[b] = self.reservoir[b], self.reservoir[a]

    def fill_reservoir(self) -> None:
        while self.reservoir_pointer < min(self.max_simultaneous, len(self.reservoir)):
            new_iter_id = random.randint(self.reservoir_pointer, len(self.reservoir) - 1)
            self.swap_reservoir(new_iter_id, self.reservoir_pointer)
            self.reservoir_pointer += 1

    def sample_reservoir_id(self) -> int:
        return random.randint(0, self.reservoir_pointer - 1)

    def return_dataset(self, reservoir_id: int) -> None:
        assert reservoir_id < self.reservoir_pointer
        dataset_id = self.reservoir[reservoir_id]
        if dataset_id in self.iterators:
            self.iterators.pop(dataset_id)
        self.swap_reservoir(reservoir_id, self.reservoir_pointer - 1)
        self.reservoir_pointer -= 1

    def __next__(self) -> tuple[int, Batch]:
        dataset_id: int | None = None
        sample: Batch | None = None
        while dataset_id is None or sample is None:
            self.fill_reservoir()
            reservoir_id = self.sample_reservoir_id()
            dataset_id = self.reservoir[reservoir_id]
            if dataset_id not in self.iterators:
                self.iterators[dataset_id] = iter(self.worker_datasets[dataset_id])
            try:
                sample = next(self.iterators[dataset_id])
            except StopIteration:
                logger.debug("Finished one iteration for dataset %d", dataset_id)
                self.return_dataset(reservoir_id)
        return dataset_id, sample


class StreamingDatasetNoIndex(StreamingDataset[Batch], IterableDataset[tuple[int, Batch]], Generic[Batch]):
    """Defines a streaming dataset which only yields the batch.

    This dataset is identical to the StreamingDataset, except that it
    cuts off the dataset index and only yields the batch.
    """

    def __next__(self) -> Batch:  # type: ignore[override]
        return super().__next__()[1]
