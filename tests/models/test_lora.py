from typing import Type, cast, get_args

import pytest
import torch
from torch import Tensor, nn

from ml.models.lora import SupportedModule, lora


@pytest.mark.parametrize("mod_type", get_args(SupportedModule))
def test_lora_modules(mod_type: Type[nn.Module]) -> None:
    """Tests loading weights from a non-LoRA model into a LoRA model.

    Args:
        mod_type: The type of the model to test.

    Raises:
        NotImplementedError: If the model type is not supported.
    """

    model: nn.Module
    lora_model: nn.Module
    in_tensors: tuple[Tensor, ...]

    if mod_type in (nn.Embedding, nn.Linear, nn.LSTM, nn.GRU):
        model = mod_type(10, 20)
        if mod_type == nn.Embedding:
            in_tensors = (torch.randint(0, 10, (5, 10)),)
        elif mod_type == nn.Linear:
            in_tensors = (torch.randn(5, 10),)
        else:
            in_tensors = (torch.randn(5, 10),)

    elif mod_type in (nn.Conv1d, nn.Conv2d):
        model = mod_type(3, 5, 3)
        if mod_type == nn.Conv1d:
            in_tensors = (torch.randn(5, 3, 10),)
        else:
            in_tensors = (torch.randn(5, 3, 10, 10),)

    else:
        raise NotImplementedError(f"Unsupported model type: {mod_type}")

    lora_model = lora(cast(SupportedModule, model), r=2)

    # Loads the weights from the reference model into the LoRA model.
    lora_model.load_state_dict(model.state_dict())

    ref_out, lora_out = model(*in_tensors), lora_model(*in_tensors)
    if isinstance(ref_out, tuple):
        ref_out, lora_out = ref_out[0], lora_out[0]

    # Checks that the outputs are initially the same for the same input.
    assert torch.allclose(ref_out, lora_out)

    # Checks that the gradients for the LoRA parameters are non-null.
    lora_out.sum().backward()
    for n, p in lora_model.named_parameters():
        if n.startswith("lora_") or n.startswith("bias"):
            assert p.grad is not None
        else:
            assert p.grad is None
