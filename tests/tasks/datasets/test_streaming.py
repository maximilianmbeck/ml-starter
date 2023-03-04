import itertools
import unittest
from typing import Iterator

from torch.utils.data.dataset import IterableDataset

from ml.tasks.datasets.streaming import StreamingDataset


class TestStreamingDataset(unittest.TestCase):
    """Tests basic functionality of streaming dataset."""

    def test_streaming_dataset(self) -> None:
        class TestDataset(IterableDataset[str]):
            def __init__(self, values: list[str]) -> None:
                super().__init__()

                self.vals = values

            def __iter__(self) -> Iterator[str]:
                def iter_func() -> Iterator[str]:
                    yield from self.vals

                return iter_func()

        dataset: StreamingDataset = StreamingDataset(
            datasets=[TestDataset(["a", "b", "c", "d", "e"]) for _ in range(20)],
            max_simultaneous=3,
        )
        unique_datasets, unique_samples = set(), set()
        for dataset_id, sample in itertools.islice(dataset, 15):
            unique_datasets.add(dataset_id)
            unique_samples.add(sample)
        assert len(unique_datasets) < 15
        assert len(unique_samples) == 5


if __name__ == "__main__":
    unittest.main()
