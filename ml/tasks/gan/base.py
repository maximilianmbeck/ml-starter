"""Defines the base GAN task type.

This class expects you to implement the following functions:

.. code-block:: python

    class MyGanTask(
        ml.GenerativeAdversarialNetworkTask[
            Config,
            Generator,
            Discriminator,
            Batch,
            GeneratorOutput,
            DiscriminatorOutput,
            Loss,
        ],
    ):
        def run_generator(self, model: Generator, batch: Batch, state: ml.State) -> GeneratorOutput:
            ...

        def run_discriminator(
            self,
            model: Discriminator,
            batch: Batch,
            gen_outputs: GeneratorOutput,
            state: ml.State,
        ) -> DiscriminatorOutput:
            ...

        def compute_discriminator_loss(
            self,
            generator: Generator,
            discriminator: Discriminator,
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
from typing import Generic, TypeVar

from torch import Tensor

from ml.core.common_types import Batch
from ml.core.config import conf_field
from ml.core.state import Phase, State
from ml.models.gan import DiscriminatorT, GenerativeAdversarialNetworkModel, GeneratorT
from ml.tasks.sl.base import SupervisedLearningTask, SupervisedLearningTaskConfig

logger: logging.Logger = logging.getLogger(__name__)

GeneratorOutput = TypeVar("GeneratorOutput")
DiscriminatorOutput = TypeVar("DiscriminatorOutput")


@dataclass
class GenerativeAdversarialNetworkTaskConfig(SupervisedLearningTaskConfig):
    generator_steps: int = conf_field(1, help="Number of generator steps per discriminator step")
    discriminator_steps: int = conf_field(1, help="Number of discriminator steps per generator step")


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
        dict[str, Tensor],
    ],
    Generic[
        GenerativeAdversarialNetworkTaskConfigT,
        GeneratorT,
        DiscriminatorT,
        Batch,
        GeneratorOutput,
        DiscriminatorOutput,
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
        gen_outputs: GeneratorOutput,
        state: State,
    ) -> DiscriminatorOutput:
        """Runs the discriminator model on the given batch.

        Args:
            discriminator: The discriminator model.
            batch: The batch to run the model on.
            gen_outputs: The output of the generator model.
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
    ) -> dict[str, Tensor]:
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
    ) -> dict[str, Tensor]:
        loss = self.compute_discriminator_loss(generator, discriminator, batch, state, gen_output, dis_output)
        return {k: -v for k, v in loss.items()}

    def is_generator_step(self, state: State, phase: Phase | None = None) -> bool:
        gen_steps, dis_steps = self.config.generator_steps, self.config.discriminator_steps
        step_id = state.num_phase_steps(state.phase if phase is None else phase) % (gen_steps + dis_steps)
        is_generator = step_id >= dis_steps
        return is_generator

    def run_model(
        self,
        model: GenerativeAdversarialNetworkModel[GeneratorT, DiscriminatorT],
        batch: Batch,
        state: State,
    ) -> tuple[GeneratorOutput, DiscriminatorOutput]:
        gen_model, dis_model = model.generator, model.discriminator
        generator_output = self.run_generator(gen_model, batch, state)
        discriminator_output = self.run_discriminator(dis_model, batch, generator_output, state)
        return generator_output, discriminator_output

    def compute_loss(
        self,
        model: GenerativeAdversarialNetworkModel[GeneratorT, DiscriminatorT],
        batch: Batch,
        state: State,
        output: tuple[GeneratorOutput, DiscriminatorOutput],
    ) -> dict[str, Tensor]:
        gen_model, dis_model = model.generator, model.discriminator
        gen_output, dis_output = output
        if self.is_generator_step(state):
            return self.compute_generator_loss(gen_model, dis_model, batch, state, gen_output, dis_output)
        return self.compute_discriminator_loss(gen_model, dis_model, batch, state, gen_output, dis_output)

    # -----
    # Hooks
    # -----

    def on_after_gan_forward_step(
        self,
        generator: GeneratorT,
        discriminator: DiscriminatorT,
        batch: Batch,
        state: State,
        gen_output: GeneratorOutput,
        dis_output: DiscriminatorOutput,
    ) -> None:
        """GAN-specific hook that is called after a forward step.

        This is useful for implementing the Wasserstein GAN gradient penalty.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            batch: The batch to run the model on.
            state: The current training state.
            gen_output: The output of the generator model.
            dis_output: The output of the discriminator model.
        """

    def on_after_forward_step(
        self,
        model: GenerativeAdversarialNetworkModel[GeneratorT, DiscriminatorT],
        batch: Batch,
        output: tuple[GeneratorOutput, DiscriminatorOutput],
        state: State,
    ) -> None:
        super().on_after_forward_step(model, batch, output, state)

        self.on_after_gan_forward_step(model.generator, model.discriminator, batch, state, output[0], output[1])
