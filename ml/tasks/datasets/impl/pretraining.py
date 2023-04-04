import logging
from pathlib import Path
from typing import Iterator, Literal

import torch
import torchvision
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

from ml.core.state import Phase
from ml.tasks.datasets.clippify import ClippifyDataset
from ml.tasks.datasets.impl.kinetics import get_kinetics_clips
from ml.tasks.datasets.impl.types import Clip, Frame
from ml.tasks.datasets.streaming import StreamingDataset
from ml.tasks.datasets.video_file import VideoFileDataset
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)

DatasetKey = Literal["kinetics"]


def get_clips(key: DatasetKey, phase: Phase) -> list[Path]:
    match key:
        case "kinetics":
            return get_kinetics_clips(phase)
        case _:
            raise ValueError(f"Invalid dataset key: {key}")


def get_frame_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.2, 1.0),
                antialias=True,
            ),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989],
            ),
        ],
    )


class VideoFrameDataset(IterableDataset[Frame]):
    """Unified PyTorch dataset for loading video frames."""

    def __init__(
        self,
        key: DatasetKey,
        phase: Phase,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> None:
        super().__init__()

        with Timer(f"getting file paths for dataset '{key}'", logger=logger):
            file_paths = get_clips(key, phase)

        # Gets file paths for just this shard.
        file_paths = file_paths[shard_id::num_shards]

        transform = get_frame_transform()

        # Gets a streaming dataset from all the videos.
        with Timer(f"initializing {phase} dataset", logger=logger):
            self._datasets = [VideoFileDataset(file_path=file_path, transform=transform) for file_path in file_paths]

    def __iter__(self) -> Iterator[Frame]:
        self.ds_id = 0
        self.clip_id = 0
        self.frame_id = 0
        self.ds_iter = iter(self._datasets[self.ds_id])
        return self

    def __next__(self) -> Frame:
        try:
            self.frame_id += 1
            frame = next(self.ds_iter)
        except StopIteration:
            self.ds_id += 1
            if self.ds_id >= len(self._datasets):
                raise
            self.clip_id += 1
            self.frame_id = 0
            self.ds_iter = iter(self._datasets[self.ds_id])
            frame = next(self.ds_iter)

        return Frame(frame=frame, identifier=f"{self.clip_id}_{self.frame_id}")


class VideoClipDataset(IterableDataset[Clip]):
    """Unified PyTorch dataset for loading video clips."""

    def __init__(
        self,
        key: DatasetKey,
        phase: Phase,
        max_simultaneous: int,
        num_images: int,
        *,
        stride: int = 1,
        jump_size: int = 1,
        sample_first: bool = False,
        use_last: bool = False,
    ) -> None:
        super().__init__()

        with Timer(f"getting file paths for dataset '{key}'", logger=logger):
            file_paths = get_clips(key, phase)

        transform = get_frame_transform()

        # Gets a streaming dataset from all the videos.
        with Timer(f"initializing {phase} dataset", logger=logger):
            self._dataset = StreamingDataset(
                datasets=[
                    ClippifyDataset(
                        image_dataset=VideoFileDataset(
                            # file_path=self.root / f"{self.phase_name}_288px" / concept_video,
                            file_path=file_path,
                            transform=transform,
                        ),
                        num_images=num_images,
                        stride=stride,
                        jump_size=jump_size,
                        sample_first=sample_first,
                        use_last=use_last,
                    )
                    for file_path in file_paths
                ],
                max_simultaneous=max_simultaneous,
            )

    def __iter__(self) -> Iterator[Clip]:
        self.ds_iter = iter(self._dataset)
        return self

    def __next__(self) -> Clip:
        clip_id, (frame_ids, frames) = next(self.ds_iter)
        frames = frames.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
        identifiers = [f"{clip_id}_{frame_id}" for frame_id in frame_ids.tolist()]
        return Clip(frames=frames, identifiers=identifiers)


def run_video_clip_dataset_adhoc_test() -> None:
    """Runs an adhoc test for the Kinetics dataset.

    Usage:
        python -m ml.tasks.datasets.impl.pretraining
    """

    configure_logging(use_tqdm=True)

    ds = VideoClipDataset("kinetics", "train", 10, 5)

    for i, item in enumerate(tqdm(ds)):
        logger.info(f"Item {i}: {item.identifiers}, {item.frames.shape}")
        if i == 10:
            break


if __name__ == "__main__":
    run_video_clip_dataset_adhoc_test()
