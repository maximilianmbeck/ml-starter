"""Tests that methods for reading and writing token datasets are correct."""

import random
from pathlib import Path

import pytest

from ml.utils.tokens import TokenReader, TokenWriter, _arr_to_bytes, _bytes_to_arr


def test_arr_to_bytes_to_arr() -> None:
    values = [1, 2, 3, 4, 5, 6, 7]
    assert _bytes_to_arr(_arr_to_bytes(values, 100), len(values), 100) == values

    values += [253, 254]
    assert _bytes_to_arr(_arr_to_bytes(values, 255), len(values), 255) == values

    values += [510, 511]
    assert _bytes_to_arr(_arr_to_bytes(values, 512), len(values), 512) == values

    values += [50_000, 99_999]
    assert _bytes_to_arr(_arr_to_bytes(values, 100_000), len(values), 100_000) == values


@pytest.mark.parametrize("num_tokens", [255, 512, 100_000])
@pytest.mark.parametrize("compressed", [True, False])
@pytest.mark.parametrize("use_offsets_file", [True, False])
def test_read_write(num_tokens: int, compressed: bool, use_offsets_file: bool, tmpdir: Path) -> None:
    file_path = tmpdir / "dataset.bin"
    offsets_path = tmpdir / ".offsets.bin" if use_offsets_file else None

    # Write the tokens to the dataset.
    all_tokens = []
    all_token_lengths = []
    with TokenWriter(file_path, num_tokens, compressed=compressed) as writer:
        for _ in range(10):
            token_length = random.randint(10, 100)
            all_token_lengths.append(token_length)
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

    # Checks reader properties.
    assert reader.lengths == all_token_lengths

    # Checks reading a subset of an index.
    assert reader[1, 3:101] == all_tokens[1][3:]
    assert reader[2, :5] == all_tokens[2][:5]
    assert reader[3, 5:] == all_tokens[3][5:]
    assert reader[4, :-5] == all_tokens[4][:-5]
    assert reader[5, -5:] == all_tokens[5][-5:]

    # Checks reading entirely into memory.
    reader = TokenReader(file_path, offsets_path=offsets_path, in_memory=True)
    num_samples = len(reader)
    assert num_samples == len(all_tokens)
    for i in range(num_samples):
        assert reader[i] == all_tokens[i]
