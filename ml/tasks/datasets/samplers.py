"""Custom samplers for datasets."""

import random
from typing import Iterator, Sized

from torch.utils.data.sampler import Sampler


class ChunkSampler(Sampler[int]):
    def __init__(self, dataset: Sized, batch_size: int, shuffle: bool = False) -> None:
        """Sampler which yields chunks of adjacent IDs.

        This sampler is useful for cases like seq2seq models with variable
        output length sequences and padding; it is more efficient to put
        similar-length sequences next to each other so that the average
        collated tensor is smaller and has less padding. In such cases, simply
        sorting the underlying dataset by caption length and using this sampler
        yields the desired behavior.

        Args:
            dataset: The dataset to sample from
            batch_size: The size of each chunk
            shuffle: Yield chunks in random order or from first to last
        """
        super().__init__(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.dataset)))
        random_offset = random.randint(0, self.batch_size - 1) if self.shuffle else 0
        ind_chunks = [indices[i : i + self.batch_size] for i in range(random_offset, len(indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(ind_chunks)
        for ind_chunk in ind_chunks:
            yield from ind_chunk

    def __len__(self) -> int:
        return self.num_samples
