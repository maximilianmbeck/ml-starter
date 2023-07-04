"""Defines the base GAN task type.

This class expects you to implement the following functions:

.. code-block:: python

    class MyGanTask(ml.ReinforcementLearning[Config, Model, Batch, GeneratorOutput, DiscriminatorOutput, Loss]):
        def run_generator(self, model: Model, batch: Batch, state: ml.State) -> GeneratorOutput:
            ...

        def run_discriminator(self, model: Model, batch: Batch, state: ml.State) -> DiscriminatorOutput:
            ...

        def compute_discriminator_loss(
            self,
            model: Model,
            batch: Batch,
            state: ml.State,
            gen_output: GeneratorOutput,
            dis_output: DiscriminatorOutput,
        ) -> Loss:
            ...

        def get_dataset(self, phase: ml.Phase) -> Dataset:
            ...
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from torch import Tensor

from ml.core.common_types import Batch, Loss
from ml.core.config import conf_field
from ml.core.state import State
from ml.models.gan import DiscriminatorT, GenerativeAdversarialNetworkModel, GeneratorT
from ml.tasks.sl.base import SupervisedLearningTask, SupervisedLearningTaskConfig

logger: logging.Logger = logging.getLogger(__name__)

GeneratorOutput = TypeVar("GeneratorOutput")
DiscriminatorOutput = TypeVar("DiscriminatorOutput")


@dataclass
class RoundRobinConfig:
    generator_steps: int = conf_field(1, help="Number of generator steps per discriminator step")
    discriminator_steps: int = conf_field(1, help="Number of discriminator steps per generator step")
    step_discriminator_first: bool = conf_field(True, help="Whether to step the discriminator first")


@dataclass
class GenerativeAdversarialNetworkTaskConfig(SupervisedLearningTaskConfig):
    round_robin: RoundRobinConfig = conf_field(RoundRobinConfig())


GenerativeAdversarialNetworkTaskConfigT = TypeVar(
    "GenerativeAdversarialNetworkTaskConfigT",
    bound=GenerativeAdversarialNetworkTaskConfig,
)


class GenerativeAdversarialNetworkTask(
    SupervisedLearningTask[
        GenerativeAdversarialNetworkTaskConfigT,
        GenerativeAdversarialNetworkModel[GeneratorT, DiscriminatorT],
        Batch,
        tuple[GeneratorOutput, DiscriminatorOutput],
        Loss,
    ],
    Generic[
        GenerativeAdversarialNetworkTaskConfigT,
        GeneratorT,
        DiscriminatorT,
        Batch,
        GeneratorOutput,
        DiscriminatorOutput,
        Loss,
    ],
    ABC,
):
    @abstractmethod
    def run_generator(self, generator: GeneratorT, batch: Batch, state: State) -> GeneratorOutput:
        """Runs the generator model on the given batch.

        Args:
            generator: The generator module.
            batch: The batch to run the model on.
            state: The current training state.

        Returns:
            The output of the generator model
        """

    @abstractmethod
    def run_discriminator(
        self,
        discriminator: DiscriminatorT,
        batch: Batch,
        generator_outputs: GeneratorOutput,
        state: State,
    ) -> DiscriminatorOutput:
        """Runs the discriminator model on the given batch.

        Args:
            discriminator: The discriminator model.
            batch: The batch to run the model on.
            generator_outputs: The output of the generator model.
            state: The current training state.

        Returns:
            The output of the discriminator model
        """

    @abstractmethod
    def compute_discriminator_loss(
        self,
        generator: GeneratorT,
        discriminator: DiscriminatorT,
        batch: Batch,
        state: State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
    ) -> Loss:
        """Computes the discriminator loss for the given batch.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            batch: The batch to run the model on.
            state: The current training state.
            gen_output: The output of the generator model.
            dis_output: The output of the discriminator model.

        Returns:
            The discriminator loss.
        """

    def compute_generator_loss(
        self,
        generator: GeneratorT,
        discriminator: DiscriminatorT,
        batch: Batch,
        state: State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
    ) -> Loss:
        loss = self.compute_discriminator_loss(generator, discriminator, batch, state, gen_output, dis_output)
        if isinstance(loss, Tensor):
            return cast(Loss, -loss)
        if isinstance(loss, dict):
            assert all(isinstance(v, Tensor) for v in loss.values())
            return cast(Loss, {k: -v for k, v in loss.items()})
        raise TypeError(f"Expected discriminator loss to be a Tensor or Dict[str, Tensor], got {type(loss)}")

    def is_generator_step(self, state: State) -> bool:
        if state.training:
            return False
        gen_steps, dis_steps = self.config.round_robin.generator_steps, self.config.round_robin.discriminator_steps
        step_id = state.num_steps % (gen_steps + dis_steps)
        if self.config.round_robin.step_discriminator_first:
            return step_id >= dis_steps
        return step_id < dis_steps

    def run_model(
        self,
        model: GenerativeAdversarialNetworkModel[GeneratorT, DiscriminatorT],
        batch: Batch,
        state: State,
    ) -> tuple[GeneratorOutput, DiscriminatorOutput]:
        gen_model, dis_model = model.generator, model.discriminator
        if self.is_generator_step(state):
            gen_model.requires_grad_(False)
            dis_model.requires_grad_(True)
            generator_output = self.run_generator(gen_model, batch, state)
            discriminator_output = self.run_discriminator(dis_model, batch, generator_output, state)
        else:
            gen_model.requires_grad_(True)
            dis_model.requires_grad_(False)
            generator_output = self.run_generator(gen_model, batch, state)
            discriminator_output = self.run_discriminator(dis_model, batch, generator_output, state)
        return generator_output, discriminator_output

    def compute_loss(
        self,
        model: GenerativeAdversarialNetworkModel[GeneratorT, DiscriminatorT],
        batch: Batch,
        state: State,
        output: tuple[GeneratorOutput, DiscriminatorOutput],
    ) -> Loss:
        gen_model, dis_model = model.generator, model.discriminator
        gen_output, dis_output = output
        if self.is_generator_step(state):
            return self.compute_generator_loss(gen_model, dis_model, batch, state, gen_output, dis_output)
        return self.compute_discriminator_loss(gen_model, dis_model, batch, state, gen_output, dis_output)
