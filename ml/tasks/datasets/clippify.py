"""Defines a dataset for converting frame streams to clips."""

import logging
import random
from collections import deque
from typing import Deque, Generic, Iterator, TypeVar

import torch
from torch import Tensor
from torch.utils.data.dataset import IterableDataset

from ml.tasks.datasets.collate import collate

logger: logging.Logger = logging.getLogger(__name__)

Batch = TypeVar("Batch")


class ClippifyDataset(IterableDataset[tuple[Tensor, Batch]], Generic[Batch]):
    def __init__(
        self,
        image_dataset: IterableDataset[Batch],
        num_images: int,
        *,
        stride: int = 1,
        jump_size: int = 1,
        sample_first: bool = False,
        use_last: bool = False,
    ) -> None:
        """Defines a dataset which efficiently yields sequences of images.

        The underlying image dataset just needs to iterate through a sequence
        of images, in order. This wrapper dataset collates the images into clips
        with some striding between adjacent images.

        Images are inserted into a deque and routinely popped. The underlying
        dataset should do necessary error handling, since this dataset will simply
        throw an error on failure.

        Args:
            image_dataset: The child dataset which yields the images
            num_images: The number of images in each clip
            stride: The stride between adjacent images
            jump_size: How many frames to jump in the future
            sample_first: If set, don't always start on the first item; instead,
                sample the first item within `jump_size`
            use_last: If set, always use the last item in the dataset
        """
        super().__init__()

        self.image_dataset = image_dataset
        self.num_images = num_images
        self.stride = stride
        self.jump_size = jump_size
        self.sample_first = sample_first
        self.use_last = use_last

    image_iter: Iterator[Batch]
    inds: list[int]
    image_queue: Deque[tuple[int, Batch]]
    image_ptr: int
    image_queue_ptr: int
    hit_last: bool

    def __iter__(self) -> Iterator[tuple[Tensor, Batch]]:
        self.image_iter = self.image_dataset.__iter__()
        self.inds = [i * self.stride for i in range(self.num_images)]
        self.image_queue = deque()
        self.image_ptr = random.randint(0, self.jump_size - 1) if self.sample_first else 0
        self.image_queue_ptr = 0
        self.hit_last = False
        return self

    def __next__(self) -> tuple[Tensor, Batch]:
        if self.hit_last:
            raise StopIteration

        inds = self.inds

        # Pushes images up to the last index.
        while self.image_queue_ptr <= inds[-1] + self.image_ptr:
            try:
                image = next(self.image_iter)
                self.image_queue.append((self.image_queue_ptr, image))
                self.image_queue_ptr += 1
            except StopIteration:
                if not self.use_last:
                    raise

                # Points `image_ptr` at the first index in the last sequence.
                # If this index isn't in the queue, raise anyway.
                self.image_ptr = self.image_queue_ptr - inds[-1] - 1
                if self.image_queue[0][0] > inds[0] + self.image_ptr:
                    raise

                # Flag to avoid another iteration.
                self.hit_last = True

                break

        # Pops all of the images which are below the first index.
        while self.image_queue[0][0] < inds[0] + self.image_ptr:
            self.image_queue.popleft()

        # Collates the images to return.
        item = collate([self.image_queue[i][1] for i in inds])
        assert item is not None, "Indices are empty!"
        self.image_ptr += self.jump_size
        return torch.IntTensor(inds), item
