"""Tests audio serialization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.utils.audio import Reader, Writer, get_audio_props, read_audio, write_audio


@pytest.mark.parametrize(
    "reader, writer",
    [
        # ("ffmpeg", "ffmpeg"),
        ("av", "av"),
        ("sf", "sf"),
    ],
)
def test_video_read_write(reader: Reader, writer: Writer, tmpdir: Path) -> None:
    mono_fpath, stereo_fpath = str(tmpdir / "a_mono.wav"), str(tmpdir / "a_ster.wav")
    mono_iter = [np.linspace(-1, 1, 3_200) for _ in range(5)]
    stereo_iter = [np.linspace(-1, 1, 3_200)[None, :].repeat(2, axis=0) for _ in range(5)]

    assert np.concatenate(mono_iter, -1).shape[-1] == 16_000
    assert np.concatenate(stereo_iter, -1).shape[-1] == 16_000

    write_audio(iter(mono_iter), mono_fpath, 16_000, writer=writer)
    write_audio(iter(stereo_iter), stereo_fpath, 16_000, writer=writer)

    mono_chunks = list(read_audio(mono_fpath, reader=reader, chunk_length=2048, sampling_rate=8_000))
    stereo_chunks = list(read_audio(stereo_fpath, reader=reader, chunk_length=2048, sampling_rate=8_000))

    assert np.concatenate(mono_chunks, -1).shape == (1, 8_000)
    assert np.concatenate(stereo_chunks, -1).shape == (2, 8_000)

    mono_props = get_audio_props(mono_fpath, reader=reader)
    stereo_props = get_audio_props(stereo_fpath, reader=reader)

    assert mono_props.channels == 1
    assert mono_props.sample_rate == 16_000
    assert mono_props.num_frames == 16_000

    assert stereo_props.channels == 2
    assert stereo_props.sample_rate == 16_000
    assert stereo_props.num_frames == 16_000

    mono_chunk, stereo_chunk = mono_chunks[0], stereo_chunks[0]
    assert mono_chunk.shape == (1, 2048)
    assert stereo_chunk.shape == (2, 2048)

    # Checks that the audio is monotonically increasing, mostly.
    # This calculation is thrown off because of the sampling rate change.
    assert (np.diff(mono_chunk) <= 0).sum() <= 100
    assert (np.diff(stereo_chunk) <= 0).sum() <= 100


if __name__ == "__main__":
    # python -m tests.utils.test_audio
    test_video_read_write("av", "av", Path(tempfile.mkdtemp()))
