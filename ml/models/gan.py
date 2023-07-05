"""Defines the wrapper model for the generator and discriminator of a GAN."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from omegaconf import MISSING, DictConfig

from ml.core.config import conf_field
from ml.core.registry import register_model
from ml.models.base import BaseModel, BaseModelConfig


@dataclass
class GenerativeAdversarialNetworkModelConfig(BaseModelConfig):
    generator: Any = conf_field(MISSING, help="The generator model to use")
    discriminator: Any = conf_field(MISSING, help="The discriminator model to use")

    @classmethod
    def update(
        cls: type["GenerativeAdversarialNetworkModelConfig"],
        config: DictConfig,
    ) -> DictConfig:
        config = super().update(config)
        assert (gen_name := config.generator.get("name")) is not None, "The generator name must be specified"
        assert (dis_name := config.discriminator.get("name")) is not None, "The discriminator name must be specified"
        _, gen_cfg_cls = register_model.lookup(gen_name)
        config.generator = gen_cfg_cls.update(config.generator)
        _, dis_cfg_cls = register_model.lookup(dis_name)
        config.discriminator = dis_cfg_cls.update(config.discriminator)
        return config

    @classmethod
    def resolve(
        cls: type["GenerativeAdversarialNetworkModelConfig"],
        config: "GenerativeAdversarialNetworkModelConfig",
    ) -> None:
        _, gen_cfg_cls = register_model.lookup(config.generator.name)
        gen_cfg_cls.resolve(config.generator)
        _, dis_cfg_cls = register_model.lookup(config.discriminator.name)
        dis_cfg_cls.resolve(config.discriminator)


GeneratorT = TypeVar("GeneratorT", bound=BaseModel)
DiscriminatorT = TypeVar("DiscriminatorT", bound=BaseModel)


@register_model("gan", GenerativeAdversarialNetworkModelConfig)
class GenerativeAdversarialNetworkModel(
    BaseModel[GenerativeAdversarialNetworkModelConfig],
    Generic[GeneratorT, DiscriminatorT],
):
    def __init__(self, config: GenerativeAdversarialNetworkModelConfig) -> None:
        super().__init__(config)

        gen_cls, _ = register_model.lookup(config.generator.name)
        self.generator = cast(GeneratorT, gen_cls(config.generator))

        dis_cls, _ = register_model.lookup(config.discriminator.name)
        self.discriminator = cast(DiscriminatorT, dis_cls(config.discriminator))

    def requires_grads_(self, generator: bool, discriminator: bool) -> None:
        self.generator.requires_grad_(generator)
        self.discriminator.requires_grad_(discriminator)

    def forward(self, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError("The base GAN model should not implement the forward pass.")
