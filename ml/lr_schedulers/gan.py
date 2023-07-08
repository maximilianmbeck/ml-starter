"""GAN learning rate scheduler wrapper.

This wrapper allows for downstream users to set different learning rate
schedules for the generator and discriminator of a GAN.

This class is used by the GAN trainer interface and shouldn't be used elsewhere.
"""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, DictConfig

from ml.core.config import conf_field
from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class GenerativeAdversarialNetworkLRSchedulerConfig(BaseLRSchedulerConfig):
    generator: Any = conf_field(MISSING, help="The generator optimizer to use")
    discriminator: Any = conf_field(MISSING, help="The discriminator optimizer to use")

    @classmethod
    def update(cls: type["GenerativeAdversarialNetworkLRSchedulerConfig"], config: DictConfig) -> DictConfig:
        config = super().update(config)
        assert (gen_name := config.generator.get("name")) is not None, "The generator name must be specified"
        assert (dis_name := config.discriminator.get("name")) is not None, "The discriminator name must be specified"
        _, gen_cfg_cls = register_lr_scheduler.lookup(gen_name)
        config.generator = gen_cfg_cls.update(config.generator)
        _, dis_cfg_cls = register_lr_scheduler.lookup(dis_name)
        config.discriminator = dis_cfg_cls.update(config.discriminator)
        return config

    @classmethod
    def resolve(
        cls: type["GenerativeAdversarialNetworkLRSchedulerConfig"],
        config: "GenerativeAdversarialNetworkLRSchedulerConfig",
    ) -> None:
        _, gen_cfg_cls = register_lr_scheduler.lookup(config.generator.name)
        gen_cfg_cls.resolve(config.generator)
        _, dis_cfg_cls = register_lr_scheduler.lookup(config.discriminator.name)
        dis_cfg_cls.resolve(config.discriminator)


@register_lr_scheduler("gan", GenerativeAdversarialNetworkLRSchedulerConfig)
class GenerativeAdversarialNetworkLRScheduler(BaseLRScheduler[GenerativeAdversarialNetworkLRSchedulerConfig]):
    def __init__(self, config: GenerativeAdversarialNetworkLRSchedulerConfig) -> None:
        super().__init__(config)

        gen_cls, _ = register_lr_scheduler.lookup(config.generator.name)
        self.generator: BaseLRScheduler = gen_cls(config.generator)

        dis_cls, _ = register_lr_scheduler.lookup(config.discriminator.name)
        self.discriminator: BaseLRScheduler = dis_cls(config.discriminator)

    def get_lr_scale(self, state: State) -> float:
        raise NotImplementedError("This method shouldn't be called directly")
