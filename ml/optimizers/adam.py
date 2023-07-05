"""Wrapper around the PyTorch Adam / AdamW optimizer.

With weight decay greater than 0 (which is the default), uses the AdamW variant
of the optimizer.
"""

from dataclasses import dataclass

from torch import nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
from ml.optimizers.common import can_use_foreach, can_use_fused, separate_decayable_params


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    betas: tuple[float, float] = conf_field((0.9, 0.999), help="Beta coefficients")
    eps: float = conf_field(1e-8, help="Epsilon term to add to the denominator for stability")
    weight_decay: float = conf_field(1e-5, help="Weight decay regularization to use")
    amsgrad: bool = conf_field(False, help="Whether to use the AMSGrad variant of the algorithm")
    default_decay: bool = conf_field(True, help="Whether to decay module params which aren't explicitly specified")
    foreach: bool | None = conf_field(None, help="Whether to use the foreach variant of the optimizer")
    capturable: bool = conf_field(False, help="Whether to use capturable AdamW pathway")
    differentiable: bool = conf_field(False, help="Whether to use differentiable AdamW")
    fused: bool | None = conf_field(None, help="Whether to use the fused optimizer")

    @classmethod
    def get_defaults(cls) -> dict[str, "AdamOptimizerConfig"]:
        return {
            "gpt-3-small": AdamOptimizerConfig(
                lr=6e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "gpt-3-medium": AdamOptimizerConfig(
                lr=3e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "gpt-3-large": AdamOptimizerConfig(
                lr=2.5e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "roberta-base": AdamOptimizerConfig(
                lr=6e-4,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            ),
            "roberta-large": AdamOptimizerConfig(
                lr=4e-4,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            ),
        }


@register_optimizer("adam", AdamOptimizerConfig)
class AdamOptimizer(BaseOptimizer[AdamOptimizerConfig, Adam | AdamW]):
    def get(self, model: nn.Module) -> Adam | AdamW:
        b1, b2 = self.config.betas

        # Chooses reasonable defaults for foreach and fused variants.
        fused, foreach = self.config.fused, self.config.foreach
        if foreach is not None and fused is not None:
            assert not (foreach and fused), "Cannot use both foreach and fused variants of Adam"
        if foreach is None and fused is None:
            if not self.config.differentiable and can_use_fused(model):
                fused = True
            elif can_use_foreach(model):
                foreach = True
        if fused is None:
            fused = False
        if foreach is None:
            foreach = False

        if self.config.weight_decay > 0.0:
            return AdamW(
                separate_decayable_params(model, self.config.default_decay, self.config.weight_decay),
                lr=self.config.lr,
                betas=(b1, b2),
                eps=self.config.eps,
                amsgrad=self.config.amsgrad,
                foreach=foreach,
                capturable=self.config.capturable,
                differentiable=self.config.differentiable,
                fused=fused,
                **self.common_kwargs,
            )

        return Adam(
            model.parameters(),
            lr=self.config.lr,
            betas=(b1, b2),
            eps=self.config.eps,
            amsgrad=self.config.amsgrad,
            foreach=foreach,
            capturable=self.config.capturable,
            differentiable=self.config.differentiable,
            fused=fused,
            **self.common_kwargs,
        )
