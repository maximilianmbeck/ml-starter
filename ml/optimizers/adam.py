from dataclasses import dataclass

from torch import nn
from torch.optim.adam import Adam

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    betas: tuple[float, float] = conf_field((0.9, 0.999), help="Beta coefficients")
    eps: float = conf_field(1e-4, help="Epsilon term to add to the denominator for stability")
    weight_decay: float = conf_field(0.0, help="Weight decay regularization to use")
    amsgrad: bool = conf_field(False, help="Whether to use the AMSGrad variant of the algorithm")
    foreach: bool = conf_field(False, help="Whether to use the foreach variant of the optimizer")
    capturable: bool = conf_field(False, help="Whether to use capturable AdamW pathway")
    differentiable: bool = conf_field(False, help="Whether to use differentiable AdamW")
    fused: bool = conf_field(False, help="Whether to use the fused optimizer")


@register_optimizer("adam", AdamOptimizerConfig)
class AdamOptimizer(BaseOptimizer[AdamOptimizerConfig]):
    def get(self, model: nn.Module) -> Adam:
        b1, b2 = self.config.betas

        return Adam(
            model.parameters(),
            lr=self.config.lr,
            betas=(b1, b2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad,
            foreach=self.config.foreach,
            capturable=self.config.capturable,
            differentiable=self.config.differentiable,
            fused=self.config.fused,
            **self.common_kwargs,
        )
