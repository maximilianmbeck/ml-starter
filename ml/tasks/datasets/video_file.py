"""Defines a dataset which iterates through frames in a video file."""

from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from torch import Tensor
from torch.utils.data.dataset import IterableDataset

from ml.utils.numpy import as_cpu_tensor
from ml.utils.video import Reader, VideoProps, read_video


class VideoFileDataset(IterableDataset[Tensor]):
    def __init__(
        self,
        file_path: str | Path,
        reader: Reader = "ffmpeg",
        transform: None | Callable[[Tensor], Tensor] = None,
    ) -> None:
        """Defines a dataset which iterates through frames in a video file.

        Args:
            file_path: The path to the video file to iterate through
            reader: The video reader to use
            transform: An optional transform to apply to each frame
        """
        super().__init__()

        self.file_path = str(file_path)
        self.reader = reader
        self.transform = transform

    video_props: VideoProps
    video_stream: Iterator[np.ndarray | Tensor]

    def __iter__(self) -> Iterator[Tensor]:
        self.video_props = VideoProps.from_file_ffmpeg(self.file_path)
        self.video_stream = read_video(self.file_path, reader=self.reader)
        return self

    def __next__(self) -> Tensor:
        buffer = next(self.video_stream)
        image = as_cpu_tensor(buffer).permute(2, 0, 1)  # HWC -> CHW
        if self.transform is not None:
            image = self.transform(image)
        return image
