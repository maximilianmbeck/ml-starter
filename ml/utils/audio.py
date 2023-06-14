# mypy: disable-error-code="import"
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
import numpy as np
import torch
import torchaudio.functional as A
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
        try:
            import ffmpeg
        except ImportError as e:
            raise ImportError("Please install matplotlib to use this function: `pip install ffmpeg-python`") from e

        probe = ffmpeg.probe(str(fpath))

        for stream in probe["streams"]:
            if stream["codec_type"] == "audio":
                return cls(
                    sample_rate=int(stream["sample_rate"]),
                    channels=int(stream["channels"]),
                    duration=float(stream["duration"]),
                )

        raise ValueError(f"Could not find audio stream in {fpath}")


def _cleanup_wav_chunk(wav: np.ndarray, channels: int | None = None) -> np.ndarray:
    if wav.ndim == 1:
        wav = wav.reshape(-1, 1 if channels is None else channels).T
    return wav


def read_audio_av(in_file: str | Path) -> Iterator[np.ndarray]:
    """Function that reads an audio file to a stream of numpy arrays using PyAV.

    Args:
        in_file: Path to the input file.

    Yields:
        Audio chunks as numpy arrays, with shape (n_channels, n_samples).
    """
    props = AudioProps.from_file_av(in_file)

    container = av.open(str(in_file))
    stream = container.streams.audio[0]

    for frame in container.decode(stream):
        frame_np = frame.to_ndarray().reshape(-1, props.channels).T
        if frame_np.dtype == np.int16:
            frame_np = frame_np.astype(np.float32) / 2**15
        yield frame_np


def read_audio_ffmpeg(in_file: str | Path, *, chunk_size: int = 16_000) -> Iterator[np.ndarray]:
    """Function that reads an audio file to a stream of numpy arrays using FFMPEG.

    Args:
        in_file: Path to the input file.
        chunk_size: Size of the chunks to read.

    Yields:
        Audio chunks as numpy arrays, with shape (n_channels, n_samples).
    """
    props = AudioProps.from_file_ffmpeg(in_file)

    try:
        import ffmpeg
    except ImportError as e:
        raise ImportError("Please install matplotlib to use this function: `pip install ffmpeg-python`") from e

    stream = ffmpeg.input(str(in_file))
    stream = ffmpeg.output(stream, "pipe:", format="f32le", acodec="pcm_f32le")
    stream = ffmpeg.run_async(stream, pipe_stdout=True, quiet=True)

    while True:
        chunk = stream.stdout.read(chunk_size)
        if not chunk:
            break
        yield np.frombuffer(chunk, dtype=np.float32).reshape(props.channels, -1)

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
        frame_np_float = _cleanup_wav_chunk(as_numpy_array(frame))
        assert frame_np_float.ndim == 2, f"Expected 2D array, got {frame_np_float.shape}D"

        if is_first_frame:
            assert (channels := frame_np_float.shape[0]) in (1, 2), f"Expected 1 or 2 channels, got {channels}"
            is_mono = channels == 1
            stream.channels = channels
            stream.layout = "mono" if is_mono else "stereo"
            is_first_frame = False

        frame_np = (frame_np_float * 2**15).astype(np.int16)
        output_fmt = "s16" if is_mono else "s16p"
        frame_av = av.AudioFrame.from_ndarray(frame_np, format=output_fmt)
        frame_av.rate = sampling_rate
        frame_av.time_base = stream.time_base
        container.mux(stream.encode(frame_av))

    container.close()


def write_audio_ffmpeg(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    sampling_rate: int,
) -> None:
    """Function that writes a stream of audio to a file using FFMPEG.

    Args:
        itr: Iterator of audio chunks.
        out_file: Path to the output file.
        sampling_rate: Sampling rate of the audio.
    """
    first_frame = _cleanup_wav_chunk(as_numpy_array(next(itr)))
    assert (channels := first_frame.shape[0]) in (1, 2), f"Expected 1 or 2 channels, got {channels}"

    try:
        import ffmpeg
    except ImportError as e:
        raise ImportError("Please install matplotlib to use this function: `pip install ffmpeg-python`") from e

    stream = ffmpeg.input("pipe:", format="f32le", acodec="pcm_f32le", ar=sampling_rate, ac=channels)
    stream = ffmpeg.output(stream, str(out_file))
    stream = ffmpeg.overwrite_output(stream)
    stream = ffmpeg.run_async(stream, pipe_stdin=True, quiet=True)

    def get_bytes(frame: np.ndarray) -> bytes:
        return frame.tobytes()

    stream.stdin.write(get_bytes(first_frame))
    for frame in itr:
        frame = _cleanup_wav_chunk(as_numpy_array(frame))
        stream.stdin.write(get_bytes(frame))

    stream.stdin.close()
    stream.wait()


Reader = Literal["ffmpeg", "av", "sf"]
Writer = Literal["ffmpeg", "av", "sf"]


def get_audio_props(in_file: str | Path, *, reader: Reader = "av") -> AudioProps:
    if reader == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in this system.")
            reader = "av"
        else:
            return AudioProps.from_file_ffmpeg(in_file)

    if reader == "av":
        return AudioProps.from_file_av(in_file)

    raise ValueError(f"Unknown reader {reader}")


def _resample_audio(
    audio_chunks: Iterator[np.ndarray],
    *,
    chunk_length: int | None = None,
    sampling_rate: tuple[int, int] | None = None,
) -> Iterator[np.ndarray]:
    if chunk_length is None:
        yield from audio_chunks
        return

    audio_chunk_list: list[np.ndarray] = []
    total_length: int = 0
    for chunk in audio_chunks:
        if sampling_rate is not None:
            chunk = A.resample(torch.from_numpy(chunk), sampling_rate[0], sampling_rate[1]).numpy()
        cur_chunk_length = chunk.shape[-1]
        while total_length + cur_chunk_length >= chunk_length:
            yield np.concatenate(audio_chunk_list + [chunk[..., : chunk_length - total_length]], axis=-1)
            chunk = chunk[..., chunk_length - total_length :]
            audio_chunk_list = []
            total_length = 0
            cur_chunk_length = chunk.shape[-1]
        if cur_chunk_length > 0:
            audio_chunk_list.append(chunk)
            total_length += cur_chunk_length

    if audio_chunk_list:
        yield np.concatenate(audio_chunk_list, axis=-1)


def read_audio(
    in_file: str | Path,
    *,
    chunk_length: int | None = None,
    sampling_rate: int | None = None,
    reader: Reader = "av",
) -> Iterator[np.ndarray]:
    """Function that reads a stream of audio from a file.

    The output audio is in ``float32`` format.

    Args:
        in_file: Path to the input file.
        chunk_length: Size of the chunks to read. If ``None``, will iterate
            whatever chunk size the underlying reader uses. Otherwise, samples
            will be rechunked to the desired size.
        sampling_rate: Sampling rate to resample the audio to. If ``None``,
            will use the sampling rate of the input audio.
        reader: Reader to use. Can be either ``ffmpeg`` or ``av``.

    Returns:
        Iterator over numpy arrays, with shape ``(n_channels, n_samples)``.
    """
    if reader == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in this system.")
            reader = "av"
        else:
            sr = None if sampling_rate is None else (AudioProps.from_file_ffmpeg(in_file).sample_rate, sampling_rate)
            return _resample_audio(read_audio_ffmpeg(in_file), chunk_length=chunk_length, sampling_rate=sr)

    if reader == "av":
        sr = None if sampling_rate is None else (AudioProps.from_file_av(in_file).sample_rate, sampling_rate)
        return _resample_audio(read_audio_av(in_file), chunk_length=chunk_length, sampling_rate=sr)

    raise ValueError(f"Unknown reader {reader}")


def write_audio(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    sample_rate: int,
    *,
    writer: Writer = "av",
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
