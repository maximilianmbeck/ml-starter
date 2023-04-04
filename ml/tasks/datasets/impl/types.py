from typing import NamedTuple

from PIL.Image import Image
from torch import Tensor


class RawFrame(NamedTuple):
    image: Image
    identifier: str | None = None


class Clip(NamedTuple):
    frames: Tensor
    identifiers: list[str]


class Frame(NamedTuple):
    frame: Tensor
    identifier: str
