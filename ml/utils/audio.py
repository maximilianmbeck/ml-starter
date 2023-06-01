"""Defines utilites for saving and loading audio streams.

The main API for using this module is:

.. code-blocks:: python

    from ml.utils.audio import read_audio, write_audio

This just uses FFMPEG so it should be rasonably quick.
"""

import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import av
import ffmpeg
import numpy as np
from torch import Tensor

from ml.utils.numpy import as_numpy_array


@dataclass
class AudioProps:
    sample_rate: int
    channels: int
    duration: float

    @classmethod
    def from_file_av(cls, fpath: str | Path) -> "AudioProps":
        container = av.open(str(fpath))
        stream = container.streams.audio[0]

        return cls(
            sample_rate=stream.rate,
            channels=stream.channels,
            duration=stream.duration,
        )

    @classmethod
    def from_file_ffmpeg(cls, fpath: str | Path) -> "AudioProps":
        probe = ffmpeg.probe(str(fpath))

        for stream in probe["streams"]:
            if stream["codec_type"] == "audio":
                return cls(
                    sample_rate=int(stream["sample_rate"]),
                    channels=int(stream["channels"]),
                    duration=float(stream["duration"]),
                )

        raise ValueError(f"Could not find audio stream in {fpath}")


def read_audio_av(in_file: str | Path) -> Iterator[np.ndarray]:
    """Function that reads an audio file to a stream of numpy arrays using PyAV.

    Args:
        in_file: Path to the input file.
    """
    container = av.open(str(in_file))
    stream = container.streams.audio[0]

    for frame in container.decode(stream):
        yield frame.to_ndarray()


def read_audio_ffmpeg(
    in_file: str | Path,
    *,
    output_fmt: str = "f32le",
    chunk_size: int = 16_000,
) -> Iterator[np.ndarray]:
    """Function that reads an audio file to a stream of numpy arrays using FFMPEG.

    Args:
        in_file: Path to the input file.
        output_fmt: Format of the output audio. See `ffmpeg` docs for more info.
        chunk_size: Size of the chunks to read.

    Yields:
        Audio chunks as numpy arrays.
    """
    stream = ffmpeg.input(str(in_file))
    stream = ffmpeg.output(stream, "pipe:", format=output_fmt, acodec="pcm_f32le")
    stream = ffmpeg.run_async(stream, pipe_stdout=True)

    while True:
        chunk = stream.stdout.read(chunk_size)
        if not chunk:
            break
        yield np.frombuffer(chunk, dtype=np.float32)

    stream.stdout.close()
    stream.wait()


def write_audio_av(itr: Iterator[np.ndarray | Tensor], out_file: str | Path, sampling_rate: int) -> None:
    """Function that writes a stream of audio to a file using PyAV.

    Args:
        itr: Iterator of audio chunks, with shape (n_channels, n_samples).
        out_file: Path to the output file.
        sampling_rate: Sampling rate of the audio.
    """
    container = av.open(str(out_file), mode="w")
    stream = container.add_stream("pcm_f32le", rate=sampling_rate)

    is_first_frame = True
    is_mono = True
    for frame in itr:
        frame_np_float = as_numpy_array(frame)
        assert frame_np_float.ndim == 2, f"Expected 2D array, got {frame_np_float.shape}D"

        if is_first_frame:
            assert (channels := frame_np_float.shape[0]) in (1, 2), f"Expected 1 or 2 channels, got {channels}"
            is_mono = channels == 1
            stream.channels = channels
            stream.layout = "mono" if is_mono else "stereo"
            is_first_frame = False

        frame_np = (frame_np_float * 2**15).astype(np.int16)
        if is_mono:
            frame_av = av.AudioFrame.from_ndarray(frame_np, format="s16")
        else:
            frame_av = av.AudioFrame.from_ndarray(frame_np, format="s16p")
        frame_av.rate = sampling_rate
        frame_av.time_base = stream.time_base
        container.mux(stream.encode(frame_av))

    container.close()


def write_audio_ffmpeg(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    sampling_rate: int,
    *,
    output_fmt: str = "f32le",
) -> None:
    """Function that writes a stream of audio to a file using FFMPEG.

    Args:
        itr: Iterator of audio chunks.
        out_file: Path to the output file.
        sampling_rate: Sampling rate of the audio.
        output_fmt: Format of the output audio. See `ffmpeg` docs for more info.
    """
    first_frame = as_numpy_array(next(itr))
    assert first_frame.ndim == 2, f"Expected 2D array, got {first_frame.shape}D"
    assert (channels := first_frame.shape[0]) in (1, 2), f"Expected 1 or 2 channels, got {channels}"

    stream = ffmpeg.input("pipe:", format=output_fmt, acodec="pcm_f32le", ar=sampling_rate, ac=channels)
    stream = ffmpeg.output(stream, str(out_file))
    stream = ffmpeg.overwrite_output(stream)
    stream = ffmpeg.run_async(stream, pipe_stdin=True)

    stream.stdin.write(first_frame.tobytes())
    for frame in itr:
        assert frame.ndim == 2, f"Expected 2D array, got {frame.shape}D"
        assert frame.shape[0] == channels, f"Expected {channels} channels, got {frame.shape[0]}"
        stream.stdin.write(as_numpy_array(frame).tobytes())

    stream.stdin.close()
    stream.wait()


Reader = Literal["ffmpeg", "av"]
Writer = Literal["ffmpeg", "av"]


def get_audio_props(in_file: str | Path, *, reader: Reader = "ffmpeg") -> AudioProps:
    if reader == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in this system.")
            reader = "av"
        else:
            return AudioProps.from_file_ffmpeg(in_file)

    if reader == "av":
        return AudioProps.from_file_av(in_file)

    raise ValueError(f"Unknown reader {reader}")


def read_audio(in_file: str | Path, *, reader: Reader = "ffmpeg") -> Iterator[np.ndarray]:
    if reader == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in this system.")
            reader = "av"
        else:
            return read_audio_ffmpeg(in_file)

    if reader == "av":
        return read_audio_av(in_file)

    raise ValueError(f"Unknown reader {reader}")


def write_audio(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    sample_rate: int,
    *,
    writer: Writer = "ffmpeg",
) -> None:
    """Function that writes a stream of audio to a file.

    Args:
        itr: Iterator of audio chunks.
        out_file: Path to the output file.
        sample_rate: Sample rate of the audio.
        writer: Writer to use. Can be either `ffmpeg` or `av`.
    """
    if writer == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in this system.")
            writer = "av"
        else:
            write_audio_ffmpeg(itr, out_file, sample_rate)
            return

    if writer == "av":
        write_audio_av(itr, out_file, sample_rate)
        return

    raise ValueError(f"Unknown writer {writer}")
