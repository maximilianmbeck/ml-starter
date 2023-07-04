"""Runs end-to-end tests of GAN training.

This test is also useful for reasoning about and debugging the entire
training loop.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from omegaconf import MISSING
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset

from ml.core.config import conf_field
from ml.core.registry import register_model, register_task
from ml.core.state import Phase, State
from ml.models.base import BaseModel, BaseModelConfig
from ml.tasks.gan.base import GenerativeAdversarialNetworkTask, GenerativeAdversarialNetworkTaskConfig
from ml.utils.checkpoint import instantiate_config
from ml.utils.logging import configure_logging


@dataclass
class GeneratorModelConfig(BaseModelConfig):
    num_layers: int = conf_field(MISSING, help="The number of layers in the generator")


@register_model("dummy-generator", GeneratorModelConfig)
class GeneratorModel(BaseModel[GeneratorModelConfig]):
    def __init__(self, config: GeneratorModelConfig) -> None:
        super().__init__(config)

        self.linear = nn.Sequential(*(nn.Linear(8, 8) for _ in range(config.num_layers)))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


@dataclass
class DiscriminatorModelConfig(BaseModelConfig):
    num_layers: int = conf_field(MISSING, help="The number of layers in the discriminator")


@register_model("dummy-discriminator", DiscriminatorModelConfig)
class DiscriminatorModel(BaseModel[DiscriminatorModelConfig]):
    def __init__(self, config: DiscriminatorModelConfig) -> None:
        super().__init__(config)

        self.linear = nn.Sequential(*(nn.Linear(8, 8) for _ in range(config.num_layers)))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


@dataclass
class DummyTaskConfig(GenerativeAdversarialNetworkTaskConfig):
    pass


Batch = Tensor
GeneratorOutput = Tensor
DiscriminatorOutput = Tensor
Loss = dict[str, Tensor]


class DummyDataset(Dataset[Batch]):
    def __getitem__(self, index: int) -> Batch:
        return torch.randn(3, 8)

    def __len__(self) -> int:
        return 10


@register_task("dummy-gan-task", DummyTaskConfig)
class DummyTask(
    GenerativeAdversarialNetworkTask[
        DummyTaskConfig,
        GeneratorModel,
        DiscriminatorModel,
        Batch,
        GeneratorOutput,
        DiscriminatorOutput,
        Loss,
    ],
):
    def run_generator(
        self,
        model: GeneratorModel,
        batch: Batch,
        state: State,
    ) -> GeneratorOutput:
        return model(batch)

    def run_discriminator(
        self,
        model: DiscriminatorModel,
        batch: Batch,
        generator_outputs: GeneratorOutput,
        state: State,
    ) -> DiscriminatorOutput:
        return model(generator_outputs)

    def compute_discriminator_loss(
        self,
        generator: GeneratorModel,
        discriminator: DiscriminatorModel,
        batch: Batch,
        state: State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
    ) -> Loss:
        return {"loss": dis_output.sum()}

    def get_dataset(self, phase: Phase) -> Dataset:
        return DummyDataset()


@pytest.mark.slow
def test_gan_e2e_training(tmpdir: Path) -> None:
    configure_logging()

    config = {
        "model": {
            "name": "gan",
            "generator": {
                "name": "dummy-generator",
                "num_layers": 2,
            },
            "discriminator": {
                "name": "dummy-discriminator",
                "num_layers": 2,
            },
        },
        "task": {
            "name": "dummy-gan-task",
            "train_dl": {
                "batch_size": 2,
            },
            "max_steps": 10,
        },
        "optimizer": {
            "name": "gan",
            "generator": {
                "name": "adam",
                "lr": 3e-4,
                "weight_decay": 1e-2,
            },
            "discriminator": {
                "name": "adam",
                "lr": 1e-4,
                "weight_decay": 1e-2,
            },
        },
        "lr_scheduler": {
            "name": "linear",
        },
        "trainer": {
            "name": "gan",
            "clip_grad_norm": 1.0,
            "clip_grad_value": 1.0,
            "exp_name": "test",
            "log_dir_name": "test",
            "base_run_dir": str(tmpdir),
            "run_id": 0,
            "batches_per_step_schedule": [
                {
                    "num_steps": 5,
                    "num_batches": 2,
                },
            ],
        },
        "logger": [
            {"name": "stdout"},
        ],
    }

    objects = instantiate_config(config)

    assert (trainer := objects.trainer) is not None
    assert (model := objects.model) is not None
    assert (task := objects.task) is not None
    assert (optimizer := objects.optimizer) is not None
    assert (lr_scheduler := objects.lr_scheduler) is not None

    trainer.train(model, task, optimizer, lr_scheduler)


if __name__ == "__main__":
    # python -m tests.e2e.test_gan_e2e
    test_gan_e2e_training(Path(tempfile.mkdtemp()))
