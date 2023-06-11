"""Tests that methods for reading and writing token datasets are correct."""

import random
from pathlib import Path

import pytest

from ml.utils.tokens import TokenReader, TokenWriter, _arr_to_bytes, _bytes_to_arr


def test_arr_to_bytes_to_arr() -> None:
    values = [1, 2, 3, 4, 5, 253, 254]
    assert _bytes_to_arr(_arr_to_bytes(values, 255), 255) == values
    assert _bytes_to_arr(_arr_to_bytes(values, 512), 512) == values
    assert _bytes_to_arr(_arr_to_bytes(values, 100_000), 100_000) == values


@pytest.mark.parametrize("num_tokens", [255, 512, 100_000])
@pytest.mark.parametrize("compressed", [True, False])
@pytest.mark.parametrize("use_offsets_file", [True, False])
def test_read_write(num_tokens: int, compressed: bool, use_offsets_file: bool, tmpdir: Path) -> None:
    file_path = tmpdir / "dataset.bin"
    offsets_path = tmpdir / ".offsets.bin" if use_offsets_file else None

    # Write the tokens to the dataset.
    all_tokens = []
    with TokenWriter(file_path, num_tokens, compressed=compressed) as writer:
        for _ in range(10):
            token_length = random.randint(1, 100)
            tokens = [random.randint(0, num_tokens - 1) for _ in range(token_length)]
            all_tokens.append(tokens)
            writer.write(tokens)

    # Read the tokens from the dataset.
    reader = TokenReader(file_path, offsets_path=offsets_path)
    num_samples = len(reader)
    assert num_samples == len(all_tokens)
    for i in range(num_samples):
        assert reader[i] == all_tokens[i]

    # Reads again, to test that the offsets file is used.
    reader = TokenReader(file_path, offsets_path=offsets_path)
    num_samples = len(reader)
    assert num_samples == len(all_tokens)
    for i in range(num_samples):
        assert reader[i] == all_tokens[i]
