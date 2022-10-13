from dataclasses import dataclass
from enum import Enum
from typing import Tuple, TypeVar

from omegaconf import MISSING
from torch import nn

from ml.core.config import conf_field

Module = TypeVar("Module", bound=nn.Module)


class Phase(Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"

    @property
    def is_train(self) -> bool:
        return self == Phase.TRAIN

    @property
    def is_valid(self) -> bool:
        return self == Phase.VALID

    @property
    def is_test(self) -> bool:
        return self == Phase.TEST

    @staticmethod
    def set_phase(model: Module, phase: "Phase") -> Tuple[Module, "Phase"]:
        if phase == Phase.TRAIN:
            if not model.training:
                model = model.train()
            return model, phase
        else:
            if model.training:
                model = model.eval()
            return model, phase


@dataclass
class State:
    """Defines the state variables to track training."""

    num_epochs: int = conf_field(MISSING, help="Number of epochs so far")
    num_steps: int = conf_field(MISSING, help="Number of steps so far")
    num_samples: int = conf_field(MISSING, help="Number of sample so far")
    num_valid_steps: int = conf_field(MISSING, help="Number of validation steps so far")
    num_test_steps: int = conf_field(MISSING, help="Number of test steps so far")
    phase: Phase = conf_field(MISSING, help="Current training phase")

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            num_epochs=0,
            num_steps=0,
            num_samples=0,
            num_valid_steps=0,
            num_test_steps=0,
            phase=Phase.TRAIN,
        )

    @property
    def training(self) -> bool:
        return self.phase == Phase.TRAIN
