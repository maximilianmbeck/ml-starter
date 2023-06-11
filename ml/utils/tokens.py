"""Defines utility functions for dealing with tokens and token datasets.

This file provides helper methods for reading and writing compressed datasets
of tokens. This compresses the tokens into ``ceil(log2(num_tokens))`` bits per
token, with padding at the end of each line to ensure that each line is a
multiple of 8 bits. This optimizes for making the file size as small as
possible while still being efficient to read from.

Here's an example of how to use the API:

.. highlight:: python
.. code-block:: python

    from ml.utils.tokens import TokenReader, TokenWriter

    num_tokens = 6
    file_path = "/path/to/dataset.bin"

    # Write the tokens to the dataset.
    with TokenWriter(file_path, num_tokens, compressed=True) as writer:
        for _ in range(10):
            writer.write([1, 2, 3, 4, 5])

    # Read the tokens from the dataset.
    reader = TokenReader(file_path)
    num_samples = len(reader)
    for i in range(num_samples):
        print(reader[i])

Additionally, you can use an offsets file to cache the offsets of each line
in the dataset:

.. highlight:: python
.. code-block:: python

    reader = TokenReader(file_path, offsets_path="/path/to/offsets.bin")
"""

import gzip
import logging
import struct
from pathlib import Path
from typing import BinaryIO, Literal, Sequence

logger = logging.getLogger(__name__)

NumberFormat = Literal["Q", "I", "H", "B"]

MAGIC = b"MLTK"  # Magic number for the token file format.
OFFSET_MAGIC = b"MLTO"  # Magic number for the offsets file format.


def _arr_to_bytes(tokens: Sequence[int], num_tokens: int) -> bytes:
    num_bits = (num_tokens - 1).bit_length()
    assert all(0 <= token < num_tokens for token in tokens)
    byte_arr = bytearray()
    cur_token = ""
    for token in tokens:
        cur_token += f"{token:0{num_bits}b}"
        byte_arr.extend(int(cur_token[i * 8 : (i + 1) * 8], 2) for i in range(len(cur_token) // 8))
        cur_token = cur_token[len(cur_token) // 8 * 8 :]
    if cur_token:
        byte_arr.append(int(cur_token.ljust(8, "0"), 2))
    # Visualize byte array: f'{int(byte_arr.hex(), base=16):b}'
    return bytes(byte_arr)


def _bytes_to_arr(data: bytes, num_tokens: int) -> list[int]:
    num_bits = (num_tokens - 1).bit_length()
    arr: list[int] = []
    cur_token = ""
    for byte in data:
        cur_token += f"{byte:08b}"
        while len(cur_token) >= num_bits:
            arr.append(int(cur_token[:num_bits], 2))
            cur_token = cur_token[num_bits:]
    return arr


class TokenWriter:
    """Helper class for writing a dataset of tokens to a file.

    This class can be used in conjunction with :class:`TokenReader` to write
    and read datasets of tokens. The default numerical formats are chosen to
    work well with typical ranges of token datasets. At the upper end, this
    supports ``2 ^ 32`` tokens, ``2 ^ 32`` tokens per line, and ``2 ^ 64``
    tokens per file.

    Parameters:
        path: The path to the file to write to.
        num_tokens: The number of tokens in the dataset.
        compressed: Whether to compress each line independently using zlib.
        overwrite_if_exists: Whether to overwrite the file if it already exists.
        num_tokens_fmt: The format string for the number of tokens.
        lengths_fmt: The format string for the lengths of each line.
        offset_fmt: The format string for the offsets of each line.
    """

    def __init__(
        self,
        path: str | Path,
        num_tokens: int,
        compressed: bool = False,
        overwrite_if_exists: bool = False,
        *,
        num_tokens_fmt: NumberFormat = "I",
        lengths_fmt: NumberFormat = "I",
        offset_fmt: NumberFormat = "Q",
    ) -> None:
        self._path = Path(path)
        self._fp: gzip.GzipFile | BinaryIO | None = None
        self._offsets: list[int] = []
        self._num_tokens = num_tokens
        self._compressed = compressed
        self._overwrite_if_exists = overwrite_if_exists
        self._num_tokens_fmt = num_tokens_fmt
        self._lengths_fmt = lengths_fmt
        self._offset_fmt = offset_fmt

    def __enter__(self) -> "TokenWriter":
        if self._path.exists():
            if self._overwrite_if_exists:
                logger.warning("Token file already exists and will be overwritten")
            else:
                raise FileExistsError(f"Token file already exists at {self._path}")
        self._fp = gzip.open(self._path, "wb") if self._compressed else open(self._path, "wb")

        # Writes the file header.
        self._fp.write(MAGIC)
        self._fp.write((self._num_tokens_fmt + self._lengths_fmt + self._offset_fmt).encode("ascii"))
        self._fp.write(struct.pack(self._num_tokens_fmt, self._num_tokens))

        return self

    def __exit__(self, _t: type[Exception] | None, _v: Exception | None, _tb: object | None) -> None:
        assert self._fp is not None

        self._fp.close()

    def write(self, tokens: Sequence[int]) -> None:
        assert self._fp is not None, "TokenWriter must be opened with a context manager"

        # Converts the tokens to a binary array.
        byte_data = _arr_to_bytes(tokens, self._num_tokens)

        # Writes the binary data
        num_bytes = len(byte_data)
        self._fp.write(struct.pack(self._lengths_fmt, num_bytes))
        self._fp.write(byte_data)


class TokenReader:
    """Helper class for reading a dataset of tokens from a file.

    This class can be used in conjunction with :class:`TokenWriter` to write
    and read datasets of tokens.

    Parameters:
        path: The path to the file to read from.
        offsets_path: The path to the file containing the offsets of each line.
            If this is not provided, the offsets will be read from the token
            file itself. If the file does not exist, it will be created.
    """

    def __init__(self, path: str | Path, offsets_path: str | Path | None) -> None:
        self._path = Path(path)
        self._offests_path = Path(offsets_path) if offsets_path is not None else None

        # Check the magic number against GZIP magic number to determine if
        # the file is compressed.
        with open(self._path, "rb") as f:
            self._compressed = f.read(2) == b"\x1f\x8b"

        with gzip.open(self._path, "rb") if self._compressed else open(self._path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError("Invalid token file")
            fmt_strings = f.read(3).decode("ascii")
            self._num_tokens_fmt = fmt_strings[0]
            self._lengths_fmt = fmt_strings[1]
            self._offset_fmt = fmt_strings[2]
            self._num_tokens = struct.unpack(self._num_tokens_fmt, f.read(struct.calcsize(self._num_tokens_fmt)))[0]

            def read_offsets() -> list[int]:
                offsets: list[int] = []
                while True:
                    offset = f.tell()
                    if (len_bytes := f.read(struct.calcsize(self._lengths_fmt))) is None or len(len_bytes) == 0:
                        break
                    offsets.append(offset)
                    len_int = struct.unpack(self._lengths_fmt, len_bytes)[0]
                    f.seek(len_int, 1)
                return offsets

            if self._offests_path is not None:
                if self._offests_path.exists():
                    with open(self._offests_path, "rb") as ofr:
                        magic = ofr.read(len(OFFSET_MAGIC))
                        if magic != OFFSET_MAGIC:
                            raise ValueError("Invalid offsets file")
                        num_offsets_bytes = ofr.read(struct.calcsize(self._num_tokens_fmt))
                        num_offsets = struct.unpack(self._num_tokens_fmt, num_offsets_bytes)[0]
                        of_bytes = ofr.read(num_offsets * struct.calcsize(self._offset_fmt))
                        self._offsets = list(struct.unpack(f"{num_offsets}{self._offset_fmt}", of_bytes))
                else:
                    self._offsets = read_offsets()
                    with open(self._offests_path, "wb") as ofw:
                        ofw.write(OFFSET_MAGIC)
                        ofw.write(struct.pack(self._num_tokens_fmt, len(self._offsets)))
                        ofw.write(struct.pack(f"{len(self._offsets)}{self._offset_fmt}", *self._offsets))
            else:
                self._offsets = read_offsets()

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> list[int]:
        offset = self._offsets[index]
        with gzip.open(self._path, "rb") if self._compressed else open(self._path, "rb") as f:
            f.seek(offset)
            num_bytes = struct.unpack(self._lengths_fmt, f.read(struct.calcsize(self._lengths_fmt)))[0]
            byte_data = f.read(num_bytes)
        return _bytes_to_arr(byte_data, self._num_tokens)
