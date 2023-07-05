"""Common optimizer utilities."""

from typing import Any, Iterable

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def separate_decayable_params(model: nn.Module, default_decay: bool, weight_decay: float) -> Iterable[dict[str, Any]]:
    """Don't weight decay biases.

    This is mostly taken from nanoGPT.

    Args:
        model: The model to get the parameters for
        default_decay: Whether to decay by default (for modules which aren't
            explicitly specified)
        weight_decay: The weight decay to use

    Returns:
        The dictionary to pass to the optimizer
    """
    wd_params: set[str] = set()
    no_wd_params: set[str] = set()
    seen: set[str] = set()

    always_decay = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.MultiheadAttention,
    )

    never_decay = (
        _BatchNorm,
        nn.LocalResponseNorm,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.Embedding,
        nn.EmbeddingBag,
    )

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if fpn in seen:
                continue
            seen.add(fpn)
            if p.ndim < 2:
                no_wd_params.add(fpn)
            elif isinstance(m, never_decay):
                no_wd_params.add(fpn)
            elif isinstance(m, always_decay):
                wd_params.add(fpn)
            else:
                (wd_params if default_decay else no_wd_params).add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = wd_params & no_wd_params
    union_params = wd_params | no_wd_params
    assert len(inter_params) == 0, "Parameters made it into both decay and no-decay sets!"
    assert len(param_dict.keys() - union_params) == 0, "Parameters were not separated into decay or no-decay set!"

    return [
        {"params": [param_dict[pn] for pn in sorted(list(wd_params))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_wd_params))], "weight_decay": 0.0},
    ]


def can_use_fused(model: nn.Module) -> bool:
    return all(p.is_cuda and p.is_floating_point() for p in model.parameters())


def can_use_foreach(model: nn.Module) -> bool:
    return all(p.device.type in ("cpu", "cuda") and p.is_floating_point() for p in model.parameters())
