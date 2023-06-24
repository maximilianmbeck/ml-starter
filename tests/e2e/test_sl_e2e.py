"""Runs end-to-end tests of supervised learning training.

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
from ml.tasks.sl.base import SupervisedLearningTask, SupervisedLearningTaskConfig
from ml.utils.checkpoint import instantiate_config
from ml.utils.logging import configure_logging


@dataclass
class ConvModelConfig(BaseModelConfig):
    num_layers: int = conf_field(MISSING, help="Number of layers to use")


@register_model("dummy-conv-model", ConvModelConfig)
class ConvModel(BaseModel[ConvModelConfig]):
    def __init__(self, config: ConvModelConfig) -> None:
        super().__init__(config)

        self.emb = nn.Embedding(10, 8)
        self.convs = nn.Sequential(*(nn.Conv1d(3, 3, 3, padding=1) for _ in range(config.num_layers)))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z = x + self.emb(y)
        return self.convs(z)


@dataclass
class DummyTaskConfig(SupervisedLearningTaskConfig):
    pass


Model = ConvModel
Batch = tuple[Tensor, Tensor]
Output = Tensor
Loss = dict[str, Tensor]


class DummyDataset(Dataset[Batch]):
    def __getitem__(self, index: int) -> Batch:
        return torch.randn(3, 8), torch.randint(0, 9, (3,))

    def __len__(self) -> int:
        return 10


@register_task("dummy-sl-task", DummyTaskConfig)
class DummyTask(SupervisedLearningTask[DummyTaskConfig, Model, Batch, Output, Loss]):
    def __init__(self, config: DummyTaskConfig) -> None:
        super().__init__(config)

    def run_model(self, model: Model, batch: Batch, state: State) -> Output:
        return model.forward(*batch)

    def compute_loss(self, model: ConvModel, batch: Batch, state: State, output: Output) -> Loss:
        return {"loss": output.sum()}

    def get_dataset(self, phase: Phase) -> Dataset:
        return DummyDataset()


@pytest.mark.slow
def test_sl_e2e_training(tmpdir: Path) -> None:
    configure_logging()

    config = {
        "model": {
            "name": "dummy-conv-model",
            "num_layers": 2,
        },
        "task": {
            "name": "dummy-sl-task",
            "train_dl": {
                "batch_size": 2,
            },
            "max_steps": 10,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 3e-4,
            "weight_decay": 1e-2,
        },
        "lr_scheduler": {
            "name": "linear",
        },
        "trainer": {
            "name": "sl",
            "clip_grad_norm": 1.0,
            "clip_grad_value": 1.0,
            "exp_name": "test",
            "log_dir_name": "test",
            "base_run_dir": str(tmpdir),
            "run_id": 0,
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
    # python -m tests.e2e.test_sl_e2e
    test_sl_e2e_training(Path(tempfile.mkdtemp()))
