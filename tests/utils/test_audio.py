"""Tests audio serialization."""

from pathlib import Path

import numpy as np
import pytest

from ml.utils.audio import Reader, Writer, get_audio_props, read_audio, write_audio


@pytest.mark.parametrize(
    "reader, writer",
    [
        ("ffmpeg", "ffmpeg"),
        ("av", "av"),
    ],
)
def test_video_read_write(reader: Reader, writer: Writer, tmpdir: Path) -> None:
    write_audio(iter((np.random.rand(1, 10) for _ in range(10))), str(tmpdir / "a_mono.wav"), 16_000, writer=writer)
    write_audio(iter((np.random.rand(2, 10) for _ in range(10))), str(tmpdir / "a_stereo.wav"), 16_000, writer=writer)
    mono_props = get_audio_props(tmpdir / "a_mono.wav", reader=reader)
    stereo_props = get_audio_props(tmpdir / "a_stereo.wav", reader=reader)
    assert mono_props.channels == 1
    assert mono_props.sample_rate == 16_000
    assert stereo_props.channels == 2
    assert stereo_props.sample_rate == 16_000
    assert len(list(read_audio(tmpdir / "a_mono.wav", reader=reader))) > 0
    assert len(list(read_audio(tmpdir / "a_stereo.wav", reader=reader))) > 0
