from dataclasses import dataclass
from typing import TypeVar

from torch import nn

from ml.core.registry import register_trainer
from ml.models.base import BaseModel
from ml.tasks.base import BaseTask
from ml.trainers.vanilla import VanillaTrainer, VanillaTrainerConfig


@dataclass
class DPTrainerConfig(VanillaTrainerConfig):
    pass


DPTrainerConfigType = TypeVar("DPTrainerConfigType", bound=DPTrainerConfig)  # pylint: disable=invalid-name


@register_trainer("dp", DPTrainerConfig)
class DPTrainer(VanillaTrainer[DPTrainerConfigType]):
    def get_task_model(self, task: BaseTask, model: BaseModel) -> nn.Module:
        task_model = super().get_task_model(task, model)
        devices = self.device.get_devices()
        if len(devices) > 1:
            task_model = nn.parallel.DataParallel(module=task_model, device_ids=devices)
        return task_model
