"""Tests that a template config can be instantiated."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data.dataset import Dataset

from ml.core.registry import (
    Objects,
    register_base,
    register_lr_scheduler,
    register_model,
    register_optimizer,
    register_task,
)
from ml.core.state import Phase, State
from ml.models.base import BaseModel, BaseModelConfig
from ml.tasks.sl.base import SupervisedLearningTask, SupervisedLearningTaskConfig


@dataclass
class TemplateModelConfig(BaseModelConfig):
    pass


@register_model("template", TemplateModelConfig)
class TemplateModel(BaseModel[TemplateModelConfig]):
    def __init__(self, config: TemplateModelConfig) -> None:
        super().__init__(config)

        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


@dataclass
class TemplateTaskConfig(SupervisedLearningTaskConfig):
    pass


@register_task("template", TemplateTaskConfig)
class TemplateTask(SupervisedLearningTask[TemplateTaskConfig, TemplateModel, Tensor, Tensor, Tensor]):
    def run_model(self, model: TemplateModel, batch: Tensor, state: State) -> Tensor:
        raise NotImplementedError

    def compute_loss(self, model: TemplateModel, batch: Tensor, state: State, output: Tensor) -> Tensor:
        raise NotImplementedError

    def get_dataset(self, phase: Phase) -> Dataset:
        raise NotImplementedError


BASE_CONFIG_YAML = """
model:
  name: template

task:
  name: template
  max_steps: 100_000
  train_dl:
    batch_size: 16

trainer:
  name: sl
  exp_name: test
"""


def get_all_keys(reg: type[register_base]) -> list[str]:
    reg.populate_registry("THIS KEY DOES NOT EXIST")
    return list(sorted(reg.REGISTRY.keys()))


@pytest.mark.parametrize("lr_scheduler_key", get_all_keys(register_lr_scheduler))
@pytest.mark.slow
def test_instantiate_lr_schedulers(lr_scheduler_key: str, tmpdir: Path) -> None:
    """Tests that all LR schedulers can be instantiated.

    Args:
        lr_scheduler_key: The key of the LR scheduler to instantiate.
        tmpdir: The temporary directory to use.
    """
    base_config = OmegaConf.create(BASE_CONFIG_YAML)
    lr_scheduler_config = {"lr_scheduler": {"name": lr_scheduler_key}}
    config = cast(DictConfig, OmegaConf.merge(base_config, lr_scheduler_config))
    config.trainer.base_run_dir = str(tmpdir)
    Objects.update_config(config)
    Objects.resolve_config(config)
    lr_scheduler = register_lr_scheduler.build_entry(config)
    assert lr_scheduler is not None


@pytest.mark.parametrize("optimizer_key", get_all_keys(register_optimizer))
@pytest.mark.slow
def test_instantiate_optimizers(optimizer_key: str, tmpdir: Path) -> None:
    """Tests that all LR schedulers can be instantiated.

    Args:
        optimizer_key: The key of the LR scheduler to instantiate.
        tmpdir: The temporary directory to use.
    """
    base_config = OmegaConf.create(BASE_CONFIG_YAML)
    optimizer_config = {"optimizer": {"name": optimizer_key}}
    config = cast(DictConfig, OmegaConf.merge(base_config, optimizer_config))
    config.trainer.base_run_dir = str(tmpdir)
    Objects.update_config(config)
    Objects.resolve_config(config)
    optimizer = register_optimizer.build_entry(config)
    assert optimizer is not None
