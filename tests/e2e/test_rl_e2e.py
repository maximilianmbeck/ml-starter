"""Runs end-to-end tests of reinforcement learning training."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from omegaconf import MISSING
from torch import Tensor, nn

from ml.core.config import conf_field
from ml.core.registry import register_model, register_task
from ml.core.state import State
from ml.models.base import BaseModel, BaseModelConfig
from ml.tasks.environments.base import Environment
from ml.tasks.rl.base import ReinforcementLearningTask, ReinforcementLearningTaskConfig
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
        self.convs = nn.Sequential(*(nn.Conv1d(1, 1, 3, padding=1) for _ in range(config.num_layers)))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z = x + self.emb(y).squeeze(-2)
        return self.convs(z)


RLState = Tensor
RLAction = Tensor


class DummyEnvironment(Environment[RLState, RLAction]):
    def reset(self, seed: int | None = None) -> RLState:
        return torch.randn(8)

    def render(self, state: RLState) -> Tensor:
        return torch.randn(3, 16, 16)

    def sample_action(self) -> RLAction:
        return torch.randint(0, 3, (1,))

    def step(self, action: RLAction) -> RLState:
        return torch.randn(8)

    def terminated(self, state: RLState) -> bool:
        return False


@dataclass
class DummyTaskConfig(ReinforcementLearningTaskConfig):
    pass


Model = ConvModel
Batch = tuple[Tensor, Tensor]
Output = Tensor
Loss = dict[str, Tensor]


@register_task("dummy-rl-task", DummyTaskConfig)
class DummyTask(ReinforcementLearningTask[DummyTaskConfig, Model, RLState, RLAction, Output, Loss]):
    def __init__(self, config: DummyTaskConfig) -> None:
        super().__init__(config)

    def get_actions(self, model: Model, states: list[RLState], optimal: bool) -> list[RLAction]:
        return [torch.randint(0, 3, (1,)) for _ in states]

    def get_environment(self) -> Environment[RLState, RLAction]:
        return DummyEnvironment()

    def run_model(self, model: Model, batch: Batch, state: State) -> Output:
        return model.forward(*batch)

    def compute_loss(self, model: ConvModel, batch: Batch, state: State, output: Output) -> Loss:
        return {"loss": output.sum()}


@pytest.mark.slow
def test_rl_e2e_training(tmpdir: Path) -> None:
    configure_logging()

    config = {
        "model": {
            "name": "dummy-conv-model",
            "num_layers": 2,
        },
        "task": {
            "name": "dummy-rl-task",
            "train_dl": {
                "batch_size": 2,
                "num_workers": 0,
            },
            "max_steps": 10,
            "dataset": {
                "num_update_steps": 10,
            },
        },
        "optimizer": {
            "name": "adam",
            "lr": 3e-4,
            "weight_decay": 1e-2,
        },
        "lr_scheduler": {
            "name": "linear",
        },
        "trainer": {
            "name": "rl",
            "clip_grad_norm": 1.0,
            "clip_grad_value": 1.0,
            "exp_name": "test",
            "log_dir_name": "test",
            "base_run_dir": str(tmpdir),
            "sampling": {
                "num_epoch_samples": 10,
            },
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
    # python -m tests.e2e.test_rl_e2e
    test_rl_e2e_training(Path(tempfile.mkdtemp()))
