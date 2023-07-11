"""Tests for the checkpoint utilities."""

from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from ml.core.registry import register_model, register_task
from ml.core.state import State
from ml.models.base import BaseModel, BaseModelConfig
from ml.tasks.base import BaseTask, BaseTaskConfig
from ml.utils.checkpoint import instantiate_config, load_model_and_task


@pytest.mark.slow
def test_load_model_and_task(tmpdir: Path) -> None:
    """Tests saving and loading a model and task.

    Args:
        tmpdir: The temporary directory to use for the test.
    """

    @register_model("dummy", BaseModelConfig)
    class DummyModel(BaseModel):
        def __init__(self, config: BaseModelConfig) -> None:
            super().__init__(config)

            self.weight = nn.Parameter(torch.randn(1))

        def forward(self, x: Tensor) -> Tensor:
            return x * self.weight

    @register_task("dummy", BaseTaskConfig)
    class DummyTask(BaseTask):
        def run_model(self, model: DummyModel, batch: Tensor, state: State) -> Tensor:
            raise NotImplementedError

        def compute_loss(self, model: DummyModel, batch: Tensor, state: State, output: Tensor) -> Tensor:
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "model": {"name": "dummy"},
            "task": {"name": "dummy"},
            "trainer": {
                "name": "sl",
                "exp_name": "test",
                "log_dir_name": "test",
                "base_run_dir": str(tmpdir),
            },
        },
    )

    # Instantiates the model, task and trainer (the trainer is only used
    # to save the model).
    objs = instantiate_config(config)
    assert (model := objs.model) is not None
    assert (task := objs.task) is not None
    assert (trainer := objs.trainer) is not None

    cast(DummyModel, model).weight.data = torch.tensor([2.0])

    # Saves the model and task.
    state = State.init_state()
    trainer.save_config()
    trainer.save_checkpoint(state, task, model)

    # Load directly from checkpoint.
    model, task = load_model_and_task(ckpt_path=trainer.ckpt_path)
    assert isinstance(model, DummyModel)
    assert isinstance(task, DummyTask)

    # Loads the model and task.
    model, task = load_model_and_task(config_path=trainer.config_path)

    # Checks that the loaded model and task have the same parameters.
    assert isinstance(model, DummyModel)
    assert isinstance(task, DummyTask)
    assert model.weight.item() == 2.0

    # Removes the checkpoint and checks that an error is thrown.
    trainer.get_ckpt_path().unlink()
    trainer.get_ckpt_path(state).unlink()
    with pytest.raises(RuntimeError):
        load_model_and_task(config_path=trainer.config_path)

    # Allows missing checkpoints.
    model, task = load_model_and_task(config_path=trainer.config_path, missing_ckpt_okay=True)
    assert isinstance(model, DummyModel)
    assert isinstance(task, DummyTask)
