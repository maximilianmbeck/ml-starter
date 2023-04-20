import asyncio
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import AsyncGenerator, Iterator, Literal

import cv2
import ffmpeg
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as V
from torch import Tensor
from torchvision.transforms import InterpolationMode

VALID_CHANNEL_COUNTS = {1, 3}


def as_uint8(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr * 255).round().astype(np.uint8)
    raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")


@dataclass
class VideoProps:
    frame_width: int
    frame_height: int
    frame_count: int
    fps: Fraction

    @classmethod
    def from_file_opencv(cls, fpath: str | Path) -> "VideoProps":
        cap = cv2.VideoCapture(str(fpath))

        return cls(
            frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=Fraction(cap.get(cv2.CAP_PROP_FPS)),
        )

    @classmethod
    def from_file_ffmpeg(cls, fpath: str | Path) -> "VideoProps":
        probe = ffmpeg.probe(str(fpath))

        for stream in probe["streams"]:
            if stream["codec_type"] == "video":
                width, height, count = stream["width"], stream["height"], int(stream["nb_frames"])
                fps_num, fps_denom = stream["r_frame_rate"].split("/")
                return cls(
                    frame_width=width,
                    frame_height=height,
                    frame_count=count,
                    fps=Fraction(int(fps_num), int(fps_denom)),
                )

        raise ValueError(f"Could not parse video properties from video in {fpath}")


def _aminmax(t: Tensor) -> tuple[Tensor, Tensor]:
    # `aminmax` isn't supported for MPS tensors, fall back to separate calls.
    minv, maxv = (t.min(), t.max()) if t.is_mps else tuple(t.aminmax())
    return minv, maxv


def make_human_viewable_resolution(
    image: Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    trg_res: tuple[int, int] = (250, 250),
) -> Tensor:
    """Resizes image to human-viewable resolution.

    Args:
        image: The image to resize, with shape (C, H, W)
        interpolation: Interpolation mode to use for image resizing
        trg_res: The target image resolution; the image will be reshaped to
            have approximately the same area as an image with this resolution

    Returns:
        The resized image
    """

    width, height = V.get_image_size(image)
    trg_height, trg_width = trg_res
    factor = math.sqrt((trg_height * trg_width) / (height * width))
    new_height, new_width = int(height * factor), int(width * factor)
    return V.resize(image, [new_height, new_width], interpolation)


def standardize_image(
    image: np.ndarray | Tensor,
    *,
    log_key: str | None = None,
    normalize: bool = True,
    keep_resolution: bool = False,
) -> np.ndarray:
    """Converts an arbitrary image to shape (C, H, W).

    Args:
        image: The image tensor to log
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)
        keep_resolution: If set, preserve original image resolution, otherwise
            change image resolution to human-viewable

    Returns:
        The normalized image, with shape (H, W, C)

    Raises:
        ValueError: If the image shape is invalid
    """

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if normalize and image.is_floating_point():
        minv, maxv = _aminmax(image)
        maxv.clamp_min_(1.0)
        minv.clamp_max_(0.0)
        image = torch.clamp((image.detach() - minv) / (maxv - minv), 0.0, 1.0)

    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] in VALID_CHANNEL_COUNTS:
            pass
        elif image.shape[2] in VALID_CHANNEL_COUNTS:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid channel count{'' if log_key is None else f' for {log_key}'}: {image.shape}")
    else:
        raise ValueError(f"Invalid image shape{'' if log_key is None else f' for {log_key}'}: {image.shape}")

    if not keep_resolution:
        image = make_human_viewable_resolution(image)

    return image.permute(1, 2, 0).detach().cpu().numpy()


def read_video_ffmpeg(
    in_file: str | Path,
    *,
    output_fmt: str = "rgb24",
    channels: int = 3,
) -> Iterator[np.ndarray]:
    """Function that reads a video to a stream of numpy arrays using FFMPEG.

    Args:
        in_file: The input video to read
        output_fmt: The output image format
        channels: Number of output channels for each video frame

    Yields:
        Frames from the video as numpy arrays with shape (H, W, C)
    """

    props = VideoProps.from_file_ffmpeg(in_file)

    stream = ffmpeg.input(str(in_file))
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt=output_fmt, r=float(props.fps))
    stream = ffmpeg.run_async(stream, pipe_stdout=True)

    while True:
        in_bytes = stream.stdout.read(props.frame_width * props.frame_height * channels)
        if not in_bytes:
            break
        yield np.frombuffer(in_bytes, np.uint8).reshape((props.frame_height, props.frame_width, channels))

    stream.stdout.close()
    stream.wait()


async def read_video_with_timestamps_ffmpeg(
    in_file: str | Path,
    *,
    output_fmt: str = "rgb24",
    channels: int = 3,
    target_dims: tuple[int | None, int | None] | None = None,
) -> AsyncGenerator[tuple[np.ndarray, float], None]:
    """Like `read_video_ffmpeg` but also returns timestamps.

    Args:
        in_file: The input video to read
        output_fmt: The output image format
        channels: Number of output channels for each video frame
        target_dims: (width, height) dimensions for images being loaded, with
            None meaning that the aspect ratio should be kept the same

    Yields:
        Frames from the video as numpy arrays with shape (H, W, C), along with
        the frame timestamps
    """

    props = VideoProps.from_file_ffmpeg(in_file)

    def aspect_ratio(x: int, a: int, b: int) -> int:
        return (x * a + b - 1) // b

    vf: list[str] = []
    if target_dims is not None:
        width_opt, height_opt = target_dims
        if width_opt is None:
            assert height_opt is not None, "If width is None, height must not be None"
            width_opt = aspect_ratio(height_opt, props.frame_width, props.frame_height)
        if height_opt is None:
            assert width_opt is not None, "If height is None, width must not be None"
            height_opt = aspect_ratio(width_opt, props.frame_height, props.frame_width)
        assert (width := width_opt) is not None and (height := height_opt) is not None
        vf.append(f"scale={width}:{height}")
    else:
        width, height = props.frame_width, props.frame_height
    vf.append("showinfo")

    stream = ffmpeg.input(str(in_file))
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt=output_fmt, r=float(props.fps), vf=",".join(vf))
    stream = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=True)

    async def gen_frames() -> AsyncGenerator[np.ndarray, None]:
        while True:
            in_bytes = stream.stdout.read(height * width * channels)
            if not in_bytes:
                await asyncio.sleep(10)
                raise StopAsyncIteration
            frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, channels))
            yield frame

    async def gen_timestamps() -> AsyncGenerator[float, None]:
        exp = re.compile(rb"n:\s*(\d+)\s*pts:\s*(\d+)\s*pts_time:\s*([\d\.]+)")
        while True:
            in_line = stream.stderr.readline()
            if not in_line:
                raise StopAsyncIteration
            exp_match = exp.search(in_line)
            if exp_match is None:
                continue
            _, _, pts_time = exp_match.groups()
            yield float(pts_time.decode("utf-8"))

    frame_iter, ts_iter = gen_frames(), gen_timestamps()

    try:
        while True:
            frame, ts = await asyncio.gather(frame_iter.__anext__(), ts_iter.__anext__())
            yield frame, ts

    except StopAsyncIteration:
        stream.stdout.close()
        stream.stderr.close()
        stream.wait()


def read_video_opencv(in_file: str | Path) -> Iterator[np.ndarray]:
    """Reads a video as a stream using OpenCV.

    Args:
        in_file: The input video to read

    Yields:
        Frames from the video as numpy arrays with shape (H, W, C)
    """

    cap = cv2.VideoCapture(str(in_file))

    while True:
        ret, buffer = cap.read()
        if not ret:
            cap.release()
            return
        yield buffer


def write_video_opencv(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    *,
    fps: int | Fraction = 30,
    codec: str = "MP4V",
) -> None:
    """Function that writes a video from a stream of numpy arrays using OpenCV.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        fps: Frames per second for the video.
        codec: FourCC code specifying OpenCV video codec type. Examples are
            MPEG, MP4V, DIVX, AVC1, H236.
    """

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)

    first_img = standardize_image(next(itr))
    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    stream = cv2.VideoWriter(str(out_file), fourcc, fps if isinstance(fps, int) else round(fps), (width, height))

    def write_frame(img: np.ndarray) -> None:
        stream.write(as_uint8(img))

    write_frame(first_img)
    for img in itr:
        write_frame(standardize_image(img))

    stream.release()
    cv2.destroyAllWindows()


def write_video_ffmpeg(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    *,
    fps: int | Fraction = 30,
    out_fps: int | Fraction = 30,
    vcodec: str = "libx264",
    input_fmt: str = "rgb24",
    output_fmt: str = "yuv420p",
) -> None:
    """Function that writes an video from a stream of numpy arrays using FFMPEG.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        fps: Frames per second for the video.
        out_fps: Frames per second for the saved video.
        vcodec: The video codec to use for the output video
        input_fmt: The input image format
        output_fmt: The output image format
    """

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)

    first_img = standardize_image(next(itr))
    height, width, _ = first_img.shape

    stream = ffmpeg.input("pipe:", format="rawvideo", pix_fmt=input_fmt, s=f"{width}x{height}", r=float(fps))
    stream = ffmpeg.output(stream, str(out_file), pix_fmt=output_fmt, vcodec=vcodec, r=float(out_fps))
    stream = ffmpeg.overwrite_output(stream)
    stream = ffmpeg.run_async(stream, pipe_stdin=True)

    def write_frame(img: np.ndarray) -> None:
        stream.stdin.write(as_uint8(img).tobytes())

    # Writes all the video frames to the file.
    write_frame(first_img)
    for img in itr:
        write_frame(standardize_image(img))

    stream.stdin.close()
    stream.wait()


def write_video_matplotlib(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    *,
    dpi: int = 50,
    fps: int | Fraction = 30,
    title: str = "Video",
    comment: str | None = None,
    writer: str = "ffmpeg",
) -> None:
    """Function that writes an video from a stream of input tensors.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        dpi: Dots per inch for output image.
        fps: Frames per second for the video.
        title: Title for the video metadata.
        comment: Comment for the video metadata.
        writer: The Matplotlib video writer to use (if you use the
            default one, make sure you have `ffmpeg` installed on your
            system).
    """

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)

    first_img = standardize_image(next(itr))
    height, width, _ = first_img.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))

    # Ensures that there's no extra space around the image.
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None,
    )

    # Creates the writer with the given metadata.
    writer_obj = ani.writers[writer]
    metadata = {
        "title": title,
        "artist": __name__,
        "comment": comment,
    }
    mpl_writer = writer_obj(
        fps=fps if isinstance(fps, int) else round(fps),
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    with mpl_writer.saving(fig, out_file, dpi=dpi):
        im = ax.imshow(as_uint8(first_img), interpolation="nearest")
        mpl_writer.grab_frame()

        for img in itr:
            im.set_data(as_uint8(standardize_image(img)))
            mpl_writer.grab_frame()


Reader = Literal["ffmpeg", "opencv"]
Writer = Literal["ffmpeg", "matplotlib", "opencv"]


def read_video(reader: Reader, in_file: str | Path) -> Iterator[np.ndarray]:
    if reader == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in the system. Falling back to OpenCV.")
            reader = "opencv"
        else:
            return read_video_ffmpeg(in_file)

    if reader == "opencv":
        return read_video_opencv(in_file)

    raise ValueError(f"Invalid reader: {reader}")


def write_video(
    writer: Writer,
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    *,
    fps: int | Fraction = 30,
) -> None:
    if writer == "ffmpeg":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in the system. Falling back to OpenCV.")
            writer = "opencv"
        else:
            return write_video_ffmpeg(itr, out_file, fps=fps)

    if writer == "matplotlib":
        if not shutil.which("ffmpeg"):
            warnings.warn("FFMPEG is not available in the system. Falling back to OpenCV.")
            writer = "opencv"
        else:
            return write_video_matplotlib(itr, out_file, fps=fps)

    if writer == "opencv":
        return write_video_opencv(itr, out_file, fps=fps)

    raise ValueError(f"Invalid writer: {writer}")
