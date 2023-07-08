"""Tests that the ``scaled_dot_product_attention`` function works correctly.

This just checks that the built-in attention function in newer versions of
PyTorch matches a reference implementation.

Also contains some tests for other attention modules.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from ml.utils.attention import NextTokenDiscriminator


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


def test_next_token_discriminator() -> None:
    dtype = torch.float64

    a = torch.tensor([1, 2, 3, 4], dtype=dtype)[None, :, None]
    b = torch.tensor([5, 6, 7, 8], dtype=dtype)[None, :, None]

    mod = NextTokenDiscriminator(1, 4)
    mod.init_emb.data.zero_()

    c, mask = mod.forward(a, b)
    attn_weights = (~mask).to(c)
    d = (attn_weights @ c).squeeze()

    # First, we shift the A vector over by one timestep to get [0, 1, 2, 3].
    # The first half of the output vector is the cumsum of these values, like
    # what would happen when doing regular attention. The second half of the
    # matrix is the cumsum of the shifted A vector plus the value of the
    # associated timestep of the B vector, i.e., [5 + 0, 6 + 1, 7 + 3, 8 + 6].
    assert torch.allclose(d, torch.tensor([0, 1, 3, 6, 5, 7, 10, 14], dtype=dtype))
