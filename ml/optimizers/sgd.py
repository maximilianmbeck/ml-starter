from dataclasses import dataclass

from torch import nn
from torch.optim.sgd import SGD

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig


@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    momentum: float = conf_field(0.0, help="Momentum term for all parameters")
    dampening: float = conf_field(0.0, help="Dampening for momentum")
    nesterov: bool = conf_field(False, help="Enable Nesterov momentum")
    weight_decay: float = conf_field(1e-5, help="Weight decay regularization to use")


@register_optimizer("sgd", SGDOptimizerConfig)
class SGDOptimizer(BaseOptimizer[SGDOptimizerConfig]):
    def get(self, model: nn.Module) -> SGD:
        return SGD(
            model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            dampening=self.config.dampening,
            nesterov=self.config.nesterov,
            weight_decay=self.config.weight_decay,
            **self.common_kwargs,
        )
