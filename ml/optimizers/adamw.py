"""Wrapper around the PyTorch Adamw optimizer."""

from dataclasses import dataclass

from torch import nn
from torch.optim.adamw import AdamW

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
from ml.optimizers.common import separate_decayable_params


@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    betas: tuple[float, float] = conf_field((0.9, 0.999), help="Beta coefficients")
    eps: float = conf_field(1e-8, help="Epsilon term to add to the denominator for stability")
    weight_decay: float = conf_field(1e-5, help="Weight decay regularization to use")
    amsgrad: bool = conf_field(False, help="Whether to use the AMSGrad variant of the algorithm")
    default_decay: bool = conf_field(True, help="Whether to decay module params which aren't explicitly specified")
    foreach: bool = conf_field(False, help="Whether to use the foreach variant of the optimizer")
    capturable: bool = conf_field(False, help="Whether to use capturable AdamW pathway")
    differentiable: bool = conf_field(False, help="Whether to use differentiable AdamW")
    fused: bool = conf_field(False, help="Whether to use the fused optimizer")

    @classmethod
    def get_defaults(cls) -> dict[str, "AdamWOptimizerConfig"]:
        return {
            "gpt-3-small": AdamWOptimizerConfig(
                lr=6e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "gpt-3-medium": AdamWOptimizerConfig(
                lr=3e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "gpt-3-large": AdamWOptimizerConfig(
                lr=2.5e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "roberta-base": AdamWOptimizerConfig(
                lr=6e-4,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            ),
            "roberta-large": AdamWOptimizerConfig(
                lr=4e-4,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            ),
        }


@register_optimizer("adamw", AdamWOptimizerConfig)
class AdamWOptimizer(BaseOptimizer[AdamWOptimizerConfig, AdamW]):
    def get(self, model: nn.Module) -> AdamW:
        b1, b2 = self.config.betas

        return AdamW(
            separate_decayable_params(model, self.config.default_decay, self.config.weight_decay),
            lr=self.config.lr,
            betas=(b1, b2),
            eps=self.config.eps,
            amsgrad=self.config.amsgrad,
            foreach=self.config.foreach,
            capturable=self.config.capturable,
            differentiable=self.config.differentiable,
            fused=self.config.fused,
            **self.common_kwargs,
        )
