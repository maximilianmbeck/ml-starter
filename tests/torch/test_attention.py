"""Tests that the ``scaled_dot_product_attention`` function works correctly.

This just checks that the built-in attention function in newer versions of
PyTorch matches a reference implementation.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_scaled_dot_product_attention(is_causal: bool, dtype: torch.dtype, device: torch.device) -> None:
    if device.type in ("cpu", "mps") and dtype in (torch.bfloat16, torch.float16):
        pytest.skip("CPU does not support bfloat16 and float16")

    query, key, value = torch.randn(2, 4, 3, 16 * 3, dtype=dtype, device=device).tensor_split(3, dim=-1)

    def ref_attention(query: Tensor, key: Tensor, value: Tensor, is_causal: bool) -> Tensor:
        qlen, klen = query.shape[-2], key.shape[-2]
        attn_mask = query.new_ones(qlen, klen, dtype=torch.bool)
        attn_mask = attn_mask.tril(diagonal=0) if is_causal else attn_mask
        attn_mask_fl = query.new_zeros(qlen, klen).masked_fill(~attn_mask, -float("inf"))
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask_fl, dim=-1)
        return attn_weight @ value

    func_out = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    ref_out = ref_attention(query, key, value, is_causal=is_causal)

    # These algorithms are not very exact in lower-precision modes so we set
    # atol to be pretty high.
    assert torch.allclose(func_out, ref_out, atol=0.05)
