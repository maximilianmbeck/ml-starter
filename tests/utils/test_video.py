"""Tests video serialization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.utils.video import Reader, Writer, read_video, write_video


@pytest.mark.parametrize(
    "reader, writer",
    [
        ("ffmpeg", "ffmpeg"),
        ("ffmpeg", "matplotlib"),
        ("av", "av"),
        ("opencv", "opencv"),
    ],
)
def test_video_read_write(reader: Reader, writer: Writer, tmpdir: Path) -> None:
    write_video(iter((np.random.rand(10, 10, 3) for _ in range(10))), str(tmpdir / "a.mp4"), writer=writer)
    write_video(iter((torch.rand(10, 10, 3) for _ in range(10))), str(tmpdir / "a_tensor.mp4"), writer=writer)
    frames = list(read_video(tmpdir / "a.mp4", reader=reader))
    assert len(frames) > 0
