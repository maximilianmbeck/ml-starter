"""Simple utility functions for doing unit test checks."""

from torch import Tensor


def assert_no_nans(t: Tensor) -> None:
    assert not t.isnan().any()
    assert not t.isinf().any()
    assert not t.isneginf().any()
