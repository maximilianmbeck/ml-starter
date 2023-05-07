"""Defines a trainer to use for supervised learning.

This trainer is akin to PyTorch Lightning or Keras, in that it handles
the training loop, logging, and checkpointing. We get a dataset and dataloader
from the task, and then train the model on the dataset.
"""

import contextlib
import logging
import signal
from dataclasses import dataclass
from types import FrameType
from typing import Generic, Iterator, TypeVar

from ml.core.common_types import Batch
from ml.core.config import conf_field
from ml.core.registry import register_trainer
from ml.lr_schedulers.base import BaseLRScheduler
from ml.optimizers.base import BaseOptimizer
from ml.tasks.sl.base import SupervisedLearningTask
from ml.trainers.base import ModelT
from ml.trainers.vanilla import TrainingFinishedException, VanillaTrainer, VanillaTrainerConfig
from ml.utils.device.base import InfinitePrefetcher
from ml.utils.timer import Timer

logger: logging.Logger = logging.getLogger(__name__)


class EpochDoneException(Exception):
    """Raised when an epoch is done."""


@dataclass
class ValidationConfig:
    valid_every_n_steps: int | None = conf_field(100, help="Number of training steps to run per test step")
    num_init_valid_steps: int | None = conf_field(1, help="Number of initial validation steps")


@dataclass
class SupervisedLearningTrainerConfig(VanillaTrainerConfig):
    validation: ValidationConfig = conf_field(ValidationConfig())
    batches_per_step: int = conf_field(1, help="Batches per training step, to simulate larger effective batch sizes")


SupervisedLearningTrainerConfigT = TypeVar("SupervisedLearningTrainerConfigT", bound=SupervisedLearningTrainerConfig)
SupervisedLearningTaskT = TypeVar("SupervisedLearningTaskT", bound=SupervisedLearningTask)


@register_trainer("sl", SupervisedLearningTrainerConfig)
class SupervisedLearningTrainer(
    VanillaTrainer[SupervisedLearningTrainerConfigT, ModelT, SupervisedLearningTaskT],
    Generic[SupervisedLearningTrainerConfigT, ModelT, SupervisedLearningTaskT],
):
    def train(
        self,
        model: ModelT,
        task: SupervisedLearningTaskT,
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
            ValueError: If the task is not a supervised learning task
        """

        if not isinstance(task, SupervisedLearningTask):
            raise ValueError(f"Expected task to be a SupervisedLearningTask, got {type(task)}")

        self._init_environment()
        model = self._compile_model(model)

        with Timer("building task model", spinner=True):
            task_model = self.get_task_model(task, model)
            self.maybe_add_grad_clipping(task_model)

        optim, lr_sched = self._get_optim_and_lr_sched(model, optimizer, lr_scheduler)
        state = self._get_state(task, model, optim, lr_sched)

        def on_exit(signum: int, _: FrameType | None) -> None:
            sig = signal.Signals(signum)
            self.on_exit(sig, state, task, model, optim, lr_sched)

        def on_finish_training() -> None:
            self.save_checkpoint(state, task, model, optim, lr_sched)
            raise TrainingFinishedException

        # Handle user-defined interrupts.
        signal.signal(signal.SIGUSR1, on_exit)

        # Gets the datasets.
        with Timer("getting datasets", 0.1, spinner=True):
            train_ds = task.get_dataset("train")
            valid_ds = task.get_dataset("valid")

        # Gets the dataloaders.
        with Timer("getting dataloaders", 0.1, spinner=True):
            train_dl = task.get_dataloader(train_ds, "train")
            valid_dl = task.get_dataloader(valid_ds, "valid")

        # Gets the prefetchers.
        with Timer("getting prefetchers", 0.1, spinner=True):
            train_pf = self._device.get_prefetcher(train_dl)
            valid_pf = self._device.get_prefetcher(valid_dl)
            valid_pf_iter = iter(InfinitePrefetcher(valid_pf))

        self.on_training_start(state, task, model, optim, lr_sched)

        try:
            with contextlib.ExitStack() as ctx:
                profile = self.get_profile()
                if profile is not None:
                    ctx.enter_context(profile)

                with Timer("initial validation step(s)"):
                    if (num_init_valid_steps := self.config.validation.num_init_valid_steps) is not None:
                        for _ in range(num_init_valid_steps):
                            self.val_step(
                                task_model=task_model,
                                batch=next(valid_pf_iter),
                                state=state,
                                task=task,
                                model=model,
                            )

                while True:
                    with self.step_context("on_epoch_start"):
                        self.on_epoch_start(state, task, model, optim, lr_sched)

                    state.num_epoch_steps = 0
                    state.num_epoch_samples = 0

                    train_pf_iter = iter(train_pf)

                    while True:
                        self._log_prefetcher_stats(train_pf)

                        if task.is_training_over(state):
                            on_finish_training()

                        with self.step_context("on_step_start"):
                            self.on_step_start(state, task, model, optim, lr_sched)

                        try:

                            def batch_iterator() -> Iterator[Batch]:
                                try:
                                    yield next(train_pf_iter)
                                except StopIteration:
                                    raise EpochDoneException

                                for _ in range(self.config.batches_per_step - 1):
                                    try:
                                        yield next(train_pf_iter)
                                    except StopIteration:
                                        pass

                            loss_dict = self.train_step(
                                task_model=task_model,
                                batches=batch_iterator(),
                                state=state,
                                task=task,
                                model=model,
                                optim=optim,
                                lr_sched=lr_sched,
                            )

                        except EpochDoneException:
                            break

                        valid_every_n_steps = self.config.validation.valid_every_n_steps
                        if valid_every_n_steps is not None and state.num_steps % valid_every_n_steps == 0:
                            self._log_prefetcher_stats(valid_pf)

                            self.val_step(
                                task_model=task_model,
                                batch=next(valid_pf_iter),
                                state=state,
                                task=task,
                                model=model,
                            )

                        if self.should_checkpoint(state):
                            self.save_checkpoint(state, task, model, optim, lr_sched)

                        if profile is not None:
                            profile.step()

                        with self.step_context("on_step_end"):
                            self.on_step_end(state, loss_dict, task, model, optim, lr_sched)

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
            logger.exception("Caught exception during training loop for %s", self.config_path)

        finally:
            self.on_training_end(state, task, model, optim, lr_sched)
