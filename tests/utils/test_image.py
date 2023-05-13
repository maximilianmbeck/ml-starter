"""Tests image utility functions."""

from pathlib import Path

import numpy as np

from ml.utils.image import read_gif, write_gif


def test_write_gif(tmpdir: Path) -> None:
    shape = (32, 32, 1)

    def get_arr(i: int) -> np.ndarray:
        # Using random values for frames doesn't work because the compression
        # algorithm is lossy; using a constant value for each frame works.
        return np.full(shape, i + 1, dtype=np.uint8) + np.array([0, 1, 2], dtype=np.uint8).reshape(1, 1, 3)

    frames = [get_arr(i) for i in range(10)]
    write_gif(iter(frames), tmpdir / "test.gif", keep_resolution=True)
    recovered_frames = list(read_gif(tmpdir / "test.gif"))
    assert (np.stack(recovered_frames) == np.stack(frames)).all()
