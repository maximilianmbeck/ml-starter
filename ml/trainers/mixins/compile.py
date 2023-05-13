"""A trainer mixin to support ``torch.compile``.

By default this is disabled, but can be enabled by setting the environment
variable ``TORCH_COMPILE=1`` or setting ``trainer.torch_compile.enabled=true``
in your configuration file.
"""

import logging
from dataclasses import dataclass
from typing import Callable, TypeVar, cast

import torch
from omegaconf import II

from ml.core.config import conf_field
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT

logger = logging.getLogger(__name__)


@dataclass
class TorchCompileConfig:
    enabled: bool = conf_field(II("oc.env:TORCH_COMPILE,0"), help="Enable Torch compilation")
    fullgraph: bool = conf_field(False, help="Whether it is OK to break the model into subgraphs")
    dynamic: bool = conf_field(False, help="Whether to use dynamic shape tracing")
    backend: str = conf_field("auto", help="The backend to use")
    mode: str | None = conf_field("max-autotune", help="Can be either 'default', 'reduce-overhead' or 'max-autotune'")


@dataclass
class CompileConfig(BaseTrainerConfig):
    torch_compile: TorchCompileConfig = conf_field(TorchCompileConfig(), help="Torch compile config")


CompileConfigT = TypeVar("CompileConfigT", bound=CompileConfig)


class CompileMixin(BaseTrainer[CompileConfigT, ModelT, TaskT]):
    """Defines a mixin for calling `torch.compile` on models."""

    def _compile_model(self, model: ModelT) -> ModelT:
        if self.config.torch_compile.enabled:
            backend: str | Callable = self.config.torch_compile.backend
            if backend == "auto":
                backend = self._device.get_torch_compile_backend()
                logger.info("Using torch-compile backend [%s]", backend)

            model = cast(
                ModelT,
                torch.compile(
                    model,
                    fullgraph=self.config.torch_compile.fullgraph,
                    dynamic=self.config.torch_compile.dynamic,
                    backend=backend,
                    mode=self.config.torch_compile.mode,
                    disable=not self.config.torch_compile.enabled,
                ),
            )

        return model
