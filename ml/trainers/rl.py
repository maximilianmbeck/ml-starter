"""Defines a trainer to use for reinforcement learning.

This trainer spawns a number of workers to collect experience from the
environment. The workers then send the experience to the model, which
learns from it. The model sends actions back to the workers, which
perform the actions in the environment and collect the next state.
"""

import contextlib
import logging
import signal
from dataclasses import dataclass
from types import FrameType
from typing import Generic, TypeVar

from omegaconf import MISSING

from ml.core.config import conf_field
from ml.core.registry import register_trainer
from ml.lr_schedulers.base import BaseLRScheduler
from ml.optimizers.base import BaseOptimizer
from ml.tasks.rl.base import ReinforcementLearningTask
from ml.trainers.base import ModelT
from ml.trainers.ddp import DDPTrainer
from ml.trainers.slurm import SlurmTrainer, SlurmTrainerConfig
from ml.trainers.vanilla import TrainingFinishedException, VanillaTrainer, VanillaTrainerConfig
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    num_epoch_samples: int = conf_field(MISSING, help="Number of samples to collect each epoch")
    min_batch_size: int = conf_field(1, help="Minimum batch size for doing inference on the model")
    max_batch_size: int | None = conf_field(None, help="Maximum batch size to infer through model")
    max_wait_time: float | None = conf_field(None, help="Maximum time to wait for inferring batches")
    min_trajectory_length: int = conf_field(1, help="Minimum length of trajectories to collect")
    max_trajectory_length: int | None = conf_field(None, help="Maximum length of trajectories to collect")
    force_sync: bool = conf_field(False, help="Force workers to run in sync mode rather than async mode")
    optimal: bool = conf_field(False, help="Whether to choose the optimal action or sample from the policy")


@dataclass
class ReinforcementLearningTrainerConfig(VanillaTrainerConfig):
    sampling: SamplingConfig = conf_field(SamplingConfig())


ReinforcementLearningTrainerConfigT = TypeVar(
    "ReinforcementLearningTrainerConfigT",
    bound=ReinforcementLearningTrainerConfig,
)
ReinforcementLearningTaskT = TypeVar("ReinforcementLearningTaskT", bound=ReinforcementLearningTask)


@register_trainer("vanilla_rl", ReinforcementLearningTrainerConfig)
class ReinforcementLearningVanillaTrainer(
    VanillaTrainer[ReinforcementLearningTrainerConfigT, ModelT, ReinforcementLearningTaskT],
    Generic[ReinforcementLearningTrainerConfigT, ModelT, ReinforcementLearningTaskT],
):
    def train(
        self,
        model: ModelT,
        task: ReinforcementLearningTaskT,
        optimizer: BaseOptimizer,
        lr_scheduler: BaseLRScheduler,
    ) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler

        Raises:
            ValueError: If the task is not a reinforcement learning task
        """

        if not isinstance(task, ReinforcementLearningTask):
            raise ValueError(f"Expected task to be a ReinforcementLearningTask, got {type(task)}")

        self._init_environment()
        model = self._compile_model(model)
        task_model = self.get_task_model(task, model)
        optim, lr_sched = self._get_optim_and_lr_sched(model, optimizer, lr_scheduler)
        state = self._get_state(task, model, optim, lr_sched)

        def on_exit(signum: int, _: FrameType | None) -> None:
            sig = signal.Signals(signum)
            self.on_exit(sig, state, task, model, optim, lr_sched)

        self.set_signal_handler(on_exit)

        def on_finish_training() -> None:
            self.save_checkpoint(state, task, model, optim, lr_sched)
            raise TrainingFinishedException

        # Handle user-defined interrupts.
        signal.signal(signal.SIGUSR1, on_exit)

        # Gets the environment workers.
        worker_pool = task.get_worker_pool(force_sync=self.config.sampling.force_sync)

        self.on_training_start(state, task, model, optim, lr_sched)

        try:
            with contextlib.ExitStack() as ctx:
                profile = self.get_profile()
                if profile is not None:
                    ctx.enter_context(profile)

                while True:
                    with self.step_context("on_epoch_start"):
                        self.on_epoch_start(state, task, model, optim, lr_sched)

                    state.num_epoch_steps = 0
                    state.num_epoch_samples = 0

                    with self.step_context("collect_rl_samples"), self.autocast_context():
                        samples = task.collect_samples(
                            model=model,
                            worker_pool=worker_pool,
                            total_samples=self.config.sampling.num_epoch_samples,
                            min_trajectory_length=self.config.sampling.min_trajectory_length,
                            max_trajectory_length=self.config.sampling.max_trajectory_length,
                            min_batch_size=self.config.sampling.min_batch_size,
                            max_batch_size=self.config.sampling.max_batch_size,
                            max_wait_time=self.config.sampling.max_wait_time,
                            optimal=self.config.sampling.optimal,
                        )

                    with self.step_context("build_rl_dataset"):
                        with Timer("building dataset", spinner=True):
                            train_ds = task.build_rl_dataset(samples)
                        with Timer("building dataloader", spinner=True):
                            train_dl = task.get_dataloader(train_ds, "train")
                        with Timer("getting prefetcher", spinner=True):
                            train_pf = self._device.get_prefetcher(train_dl)

                    for train_batch in train_pf:
                        self._log_prefetcher_stats(train_pf)

                        if task.is_training_over(state):
                            on_finish_training()

                        with self.step_context("on_step_start"):
                            self.on_step_start(state, train_batch, task, model, optim, lr_sched)

                        loss_dict = self.train_step(
                            task_model=task_model,
                            batch=train_batch,
                            state=state,
                            task=task,
                            model=model,
                            optim=optim,
                            lr_sched=lr_sched,
                        )

                        if self.should_checkpoint(state):
                            self.save_checkpoint(state, task, model, optim, lr_sched)

                        if profile is not None:
                            profile.step()

                        with self.step_context("on_step_end"):
                            self.on_step_end(state, train_batch, loss_dict, task, model, optim, lr_sched)

                        if task.epoch_is_over(state):
                            break

                    with self.step_context("on_epoch_end"):
                        self.on_epoch_end(state, task, model, optim, lr_sched)

                    state.num_epochs += 1

        except TrainingFinishedException:
            logger.info(
                "Finished training after %d epochs, %d steps, %d samples",
                state.num_epochs,
                state.num_steps,
                state.num_samples,
            )

        except Exception:
            logger.exception("Caught exception during training loop")

        finally:
            self.on_training_end(state, task, model, optim, lr_sched)


@register_trainer("ddp_rl", ReinforcementLearningTrainerConfig)
class ReinforcementLearningDDPTrainer(
    ReinforcementLearningVanillaTrainer[ReinforcementLearningTrainerConfig, ModelT, ReinforcementLearningTaskT],
    DDPTrainer[ReinforcementLearningTrainerConfig, ModelT, ReinforcementLearningTaskT],
):
    pass


@dataclass
class ReinforcementLearningSlurmTrainerConfig(
    ReinforcementLearningTrainerConfig,
    SlurmTrainerConfig,
):
    pass


@register_trainer("slurm_rl", ReinforcementLearningSlurmTrainerConfig)
class ReinforcementLearningSlurmTrainer(
    ReinforcementLearningVanillaTrainer[ReinforcementLearningSlurmTrainerConfig, ModelT, ReinforcementLearningTaskT],
    SlurmTrainer[ReinforcementLearningSlurmTrainerConfig, ModelT, ReinforcementLearningTaskT],
):
    pass
