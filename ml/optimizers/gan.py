"""GAN model optimizer wrapper.

This wrapper allows downstream users to set different optimizers or optimizer
parameters for the generator and discriminator of a GAN.

This class is used by the GAN trainer interface and shouldn't be used elsewhere.
"""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, DictConfig
from torch import nn
from torch.optim.optimizer import Optimizer

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig


@dataclass
class GenerativeAdversarialNetworkOptimizerConfig(BaseOptimizerConfig):
    generator: Any = conf_field(MISSING, help="The generator optimizer to use")
    discriminator: Any = conf_field(MISSING, help="The discriminator optimizer to use")

    @classmethod
    def update(cls: type["GenerativeAdversarialNetworkOptimizerConfig"], config: DictConfig) -> DictConfig:
        config = super().update(config)
        assert (gen_name := config.generator.get("name")) is not None, "The generator name must be specified"
        assert (dis_name := config.discriminator.get("name")) is not None, "The discriminator name must be specified"
        _, gen_cfg_cls = register_optimizer.lookup(gen_name)
        config.generator = gen_cfg_cls.update(config.generator)
        _, dis_cfg_cls = register_optimizer.lookup(dis_name)
        config.discriminator = dis_cfg_cls.update(config.discriminator)
        return config

    @classmethod
    def resolve(
        cls: type["GenerativeAdversarialNetworkOptimizerConfig"],
        config: "GenerativeAdversarialNetworkOptimizerConfig",
    ) -> None:
        _, gen_cfg_cls = register_optimizer.lookup(config.generator.name)
        gen_cfg_cls.resolve(config.generator)
        _, dis_cfg_cls = register_optimizer.lookup(config.discriminator.name)
        dis_cfg_cls.resolve(config.discriminator)


@register_optimizer("gan", GenerativeAdversarialNetworkOptimizerConfig)
class GenerativeAdversarialNetworkOptimizer(BaseOptimizer[GenerativeAdversarialNetworkOptimizerConfig, Optimizer]):
    def __init__(self, config: GenerativeAdversarialNetworkOptimizerConfig) -> None:
        super().__init__(config)

        gen_cls, _ = register_optimizer.lookup(config.generator.name)
        self.generator = gen_cls(config.generator)

        dis_cls, _ = register_optimizer.lookup(config.discriminator.name)
        self.discriminator = dis_cls(config.discriminator)

    def get(self, model: nn.Module) -> Optimizer:
        raise NotImplementedError("This method shouldn't be called directly.")
