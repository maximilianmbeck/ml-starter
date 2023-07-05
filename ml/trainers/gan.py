"""Defines a trainer to use for training GANs.

This trainer is similar to the supervised learning trainer, but with separate
optimizers for the generator and discriminator, and supporting round robin
training.
"""

import contextlib
import logging
import signal
from dataclasses import dataclass
from types import FrameType
from typing import Generic, Iterator, TypeVar

from torch import nn
from torch.optim.optimizer import Optimizer

from ml.core.common_types import Batch
from ml.core.config import conf_field
from ml.core.registry import register_trainer
from ml.core.state import Phase, State
from ml.loggers.multi import namespace_context
from ml.lr_schedulers.base import BaseLRScheduler, SchedulerAdapter
from ml.models.gan import GenerativeAdversarialNetworkModel
from ml.optimizers.base import BaseOptimizer
from ml.optimizers.gan import GenerativeAdversarialNetworkOptimizer
from ml.tasks.gan.base import GenerativeAdversarialNetworkTask
from ml.trainers.learning import TrainingFinishedError
from ml.trainers.sl import EpochDoneError, SupervisedLearningTrainer, SupervisedLearningTrainerConfig
from ml.utils.device.base import InfinitePrefetcher
from ml.utils.timer import Timer

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class GenerativeAdversarialNetworkTrainerConfig(SupervisedLearningTrainerConfig):
    discriminator_key: str = conf_field("dis", help="The logging key for the discriminator")
    generator_key: str = conf_field("gen", help="The logging key for the generator")


GenerativeAdversarialNetworkTrainerConfigT = TypeVar(
    "GenerativeAdversarialNetworkTrainerConfigT",
    bound=GenerativeAdversarialNetworkTrainerConfig,
)
GenerativeAdversarialNetworkModelT = TypeVar(
    "GenerativeAdversarialNetworkModelT",
    bound=GenerativeAdversarialNetworkModel,
)
GenerativeAdversarialNetworkTaskT = TypeVar(
    "GenerativeAdversarialNetworkTaskT",
    bound=GenerativeAdversarialNetworkTask,
)


@register_trainer("gan", GenerativeAdversarialNetworkTrainerConfig)
class GenerativeAdversarialNetworkTrainer(
    SupervisedLearningTrainer[
        GenerativeAdversarialNetworkTrainerConfigT,
        GenerativeAdversarialNetworkModelT,
        GenerativeAdversarialNetworkTaskT,
    ],
    Generic[
        GenerativeAdversarialNetworkTrainerConfigT,
        GenerativeAdversarialNetworkModelT,
        GenerativeAdversarialNetworkTaskT,
    ],
):
    def _logging_key(self, task: GenerativeAdversarialNetworkTaskT, state: State, phase: Phase) -> str:
        is_gen = task.is_generator_step(state, phase)
        return self.config.generator_key if is_gen else self.config.discriminator_key

    def train(
        self,
        model: GenerativeAdversarialNetworkModelT,
        task: GenerativeAdversarialNetworkTaskT,
        optimizer: BaseOptimizer,
        lr_scheduler: BaseLRScheduler,
    ) -> None:
        """Runs the training loop.

        Args:
            model: The model to train.
            task: The task to train on.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler to use.
        """
        if not isinstance(model, GenerativeAdversarialNetworkModel):
            raise ValueError(f"Expected model to be a GenerativeAdversarialNetworkModel, got {type(model)}")
        if not isinstance(task, GenerativeAdversarialNetworkTask):
            raise ValueError(f"Expected task to be a GenerativeAdversarialNetworkTask, got {type(task)}")

        self._init_environment()

        with Timer("compiling model", spinner=True):
            model = self._compile_model(model)

        with Timer("building task model", spinner=True):
            task_model = self._get_task_model(task, model)

        gen_optim, gen_lr_sched = self._get_optim_and_lr_sched(model.generator, optimizer, lr_scheduler, True)
        dis_optim, dis_lr_sched = self._get_optim_and_lr_sched(model.discriminator, optimizer, lr_scheduler, False)
        optims = {"gen": gen_optim, "dis": dis_optim}
        lr_scheds = {"gen": gen_lr_sched, "dis": dis_lr_sched}
        state = self._get_state(task, model, optims, lr_scheds)

        def on_exit(signum: int, _: FrameType | None) -> None:
            sig = signal.Signals(signum)
            self.on_exit(sig, state, task, model, optims, lr_scheds)

        def on_finish_training() -> None:
            self.save_checkpoint(state, task, model, optims, lr_scheds)
            raise TrainingFinishedError

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

        self.on_training_start(state, task, model, optims, lr_scheds)

        try:
            with contextlib.ExitStack() as ctx:
                profile = self.get_profile()
                if profile is not None:
                    ctx.enter_context(profile)

                with Timer("initial validation step(s)"):
                    if (num_init_valid_steps := self.config.validation.num_init_valid_steps) is not None:
                        for _ in range(num_init_valid_steps):
                            with namespace_context(self._logging_key(task, state, "valid")):
                                self.val_step(
                                    task_model=task_model,
                                    batch=next(valid_pf_iter),
                                    state=state,
                                    task=task,
                                    model=model,
                                )

                while True:
                    with self.step_context("on_epoch_start"):
                        self.on_epoch_start(state, task, model, optims, lr_scheds)

                    state.num_epoch_steps = 0
                    state.num_epoch_samples = 0

                    train_pf_iter = iter(train_pf)

                    def batch_iterator() -> Iterator[Batch]:
                        try:
                            yield next(train_pf_iter)
                        except StopIteration:
                            raise EpochDoneError

                        for _ in range(self.get_batches_per_step(state) - 1):
                            try:
                                yield next(train_pf_iter)
                            except StopIteration:
                                pass

                    while True:
                        self._log_prefetcher_stats(train_pf)

                        if task.is_training_over(state):
                            on_finish_training()

                        with self.step_context("on_step_start"):
                            self.on_step_start(state, task, model, optims, lr_scheds)

                        try:
                            is_gen = task.is_generator_step(state, "train")
                            optim = gen_optim if is_gen else dis_optim
                            lr_sched = gen_lr_sched if is_gen else dis_lr_sched
                            model.requires_grads_(is_gen, not is_gen)

                            with namespace_context(self._logging_key(task, state, "train")):
                                loss_dict = self.train_step(
                                    task_model=task_model,
                                    batches=batch_iterator(),
                                    state=state,
                                    task=task,
                                    model=model,
                                    optim=optim,
                                    lr_sched=lr_sched,
                                )

                        except EpochDoneError:
                            break

                        valid_every_n_steps = self.config.validation.valid_every_n_steps
                        if valid_every_n_steps is not None and state.num_steps % valid_every_n_steps == 0:
                            self._log_prefetcher_stats(valid_pf)

                            with namespace_context(self._logging_key(task, state, "valid")):
                                self.val_step(
                                    task_model=task_model,
                                    batch=next(valid_pf_iter),
                                    state=state,
                                    task=task,
                                    model=model,
                                )

                        if self.should_checkpoint(state):
                            self.save_checkpoint(state, task, model, optims, lr_scheds)

                        if profile is not None:
                            profile.step()

                        with self.step_context("on_step_end"):
                            self.on_step_end(state, loss_dict, task, model, optims, lr_scheds)

                    with self.step_context("on_epoch_end"):
                        self.on_epoch_end(state, task, model, optims, lr_scheds)

                    state.num_epochs += 1

        except TrainingFinishedError:
            logger.info(
                "Finished training after %d epochs, %d steps, %d samples",
                state.num_epochs,
                state.num_steps,
                state.num_samples,
            )

        except Exception:
            logger.exception("Caught exception during training loop for %s", self.config_path)

        finally:
            self.on_training_end(state, task, model, optims, lr_scheds)

    def _get_optim_and_lr_sched(
        self,
        task_model: nn.Module,
        optimizer: BaseOptimizer,
        lr_scheduler: BaseLRScheduler,
        is_gen: bool | None = None,
    ) -> tuple[Optimizer, SchedulerAdapter]:
        if is_gen is None or not isinstance(optimizer, GenerativeAdversarialNetworkOptimizer):
            return super()._get_optim_and_lr_sched(task_model, optimizer, lr_scheduler)
        optimizer = optimizer.generator if is_gen else optimizer.discriminator
        with Timer("building optimizer", 0.1, spinner=True):
            optim = optimizer.get(task_model)
        with Timer("building learning rate scheduler", 0.1, spinner=True):
            lr_sched = lr_scheduler.get(optim)
        return optim, lr_sched
