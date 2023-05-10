from typing import Type, cast

import pytest
import torch
from torch import Tensor, nn

from ml.models.lora import lora


@pytest.mark.parametrize("mod_type", [nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d])
def test_lora_load_weights(mod_type: Type[nn.Module]) -> None:
    """Tests loading weights from a non-LoRA model into a LoRA model.

    Args:
        mod_type: The type of the model to test.

    Raises:
        NotImplementedError: If the model type is not supported.
    """

    model: nn.Module
    lora_model: nn.Module
    in_tensor: Tensor
    if mod_type in (nn.Embedding, nn.Linear):
        model = mod_type(10, 20)
        in_tensor = torch.randint(0, 10, (5, 10)) if mod_type == nn.Embedding else torch.randn(5, 10)
    elif mod_type in (nn.Conv1d, nn.Conv2d):
        model = mod_type(3, 5, 3)
        in_tensor = torch.randn(5, 3, 10) if mod_type == nn.Conv1d else torch.randn(5, 3, 10, 10)
    else:
        raise NotImplementedError(f"Unsupported model type: {mod_type}")

    lora_model = lora(cast(nn.Embedding | nn.Linear | nn.Conv1d | nn.Conv2d, model), r=2)

    # Loads the weights from the reference model into the LoRA model.
    lora_model.load_state_dict(model.state_dict())

    # Checks that the outputs are initially the same for the same input.
    ref_out, lora_out = model(in_tensor), lora_model(in_tensor)
    assert torch.allclose(ref_out, lora_out)

    # Checks that the gradients for the LoRA parameters are non-null.
    lora_out.sum().backward()
    assert any(getattr(lora_model, p).grad is not None for p in ("lora_a", "lora_b"))
