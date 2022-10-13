from typing import Dict

from torch import Tensor


def assert_dicts_equal(a: Dict[str, Tensor], b: Dict[str, Tensor], *, max_err: float = 1e-3) -> None:
    keys = set(a.keys()) | set(b.keys())
    assert set(a.keys()) == keys
    assert set(b.keys()) == keys
    for k in keys:
        assert a[k].shape == b[k].shape
        assert (a[k] - b[k]).abs().max().item() < max_err
