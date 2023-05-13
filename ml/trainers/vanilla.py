"""Defines a vanilla trainer which doesn't do any device or data manipulation.

This trainer expects the task to handle all the relevant movement of data and
models to their associated devices.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ml.core.common_types import Batch
from ml.core.config import conf_field
from ml.core.state import State, set_phase
from ml.lr_schedulers.base import BaseLRScheduler, SchedulerAdapter
from ml.optimizers.base import BaseOptimizer
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT
from ml.trainers.mixins.compile import CompileConfig, CompileMixin
from ml.trainers.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
from ml.trainers.mixins.data_parallel import ParallelMixin, TrainerParallelConfig
from ml.trainers.mixins.gpu_stats import GPUStatsConfig, GPUStatsMixin
from ml.trainers.mixins.grad_clipping import (
    GradientClippingConfig,
    GradientClippingTrainerMixin,
)
from ml.trainers.mixins.mixed_precision import (
    MixedPrecisionTrainerConfig,
    MixedPrecisionTrainerMixin,
)
from ml.trainers.mixins.profiler import ProfilerTrainerConfig, ProfilerTrainerMixin
from ml.utils.timer import Timer

logger: logging.Logger = logging.getLogger(__name__)


class TrainingFinishedError(Exception):
    pass


@dataclass
class VanillaTrainerConfig(
    ProfilerTrainerConfig,
    GradientClippingConfig,
    MixedPrecisionTrainerConfig,
    GPUStatsConfig,
    CPUStatsConfig,
    CompileConfig,
    TrainerParallelConfig,
    BaseTrainerConfig,
):
    set_to_none: bool = conf_field(True, help="Mode for clearing optimizer gradients")
    deterministic: bool = conf_field(False, help="If set, use determinstic algorithms")
    use_tf32: bool = conf_field(True, help="If set, use TensorFloat32")
    detect_anomaly: bool = conf_field(False, help="Whether to detect anomalies")
    detect_anomaly_check_nan: bool = conf_field(False, help="Whether to check for NaNs when detecting anomalies")


VanillaTrainerConfigT = TypeVar("VanillaTrainerConfigT", bound=VanillaTrainerConfig)


class VanillaTrainer(
    ProfilerTrainerMixin[VanillaTrainerConfigT, ModelT, TaskT],
    GradientClippingTrainerMixin[VanillaTrainerConfigT, ModelT, TaskT],
    MixedPrecisionTrainerMixin[VanillaTrainerConfigT, ModelT, TaskT],
    GPUStatsMixin[VanillaTrainerConfigT, ModelT, TaskT],
    CPUStatsMixin[VanillaTrainerConfigT, ModelT, TaskT],
    CompileMixin[VanillaTrainerConfigT, ModelT, TaskT],
    ParallelMixin[VanillaTrainerConfigT, ModelT, TaskT],
    BaseTrainer[VanillaTrainerConfigT, ModelT, TaskT],
    Generic[VanillaTrainerConfigT, ModelT, TaskT],
):
    def train_step(
        self,
        *,
        task_model: nn.Module,
        batches: Iterator[Batch],
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> dict[str, Tensor]:
        with self.step_context("change_mode"):
            task_model, state.phase = set_phase(task_model, "train")
        total_bsz: int | None = None
        first_batch = True
        for batch in batches:
            bsz = task.get_batch_size(batch)
            if bsz is not None:
                total_bsz = bsz if total_bsz is None else total_bsz + bsz
            with self.step_context("forward"), self.autocast_context():
                loss = task_model(batch, state)
            with self.step_context("get_single_loss"):
                single_loss, loss_names = task.get_single_loss(loss)
            with self.step_context("backward"):
                self.scale_mixed_precision(single_loss.sum()).backward()
            if first_batch:
                with self.step_context("log_losses"):
                    self.log_mp_scale()
                    single_loss_detached = single_loss.detach()
                    loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
                    task.log_loss_dict(loss_dict, state)
                first_batch = False
        with self.step_context("clip_grads"):
            self.clip_grads(model=task_model, optim=optim)
        with self.step_context("step"):
            self.step_optimizer(optim=optim)
            lr_sched.step(state)
            self.logger.log_scalar("lr_scale", lr_sched.lr_scale, namespace="optim")
        with self.step_context("zero_grads"):
            optim.zero_grad(set_to_none=self.config.set_to_none)
        with self.step_context("write_logs"), self.autocast_context():
            self.write_logs(task, model, state)
        with self.step_context("update_state"):
            state.num_steps += 1
            state.num_epoch_steps += 1
            if total_bsz is not None:
                state.num_samples += total_bsz
                state.num_epoch_samples += total_bsz
        return loss_dict

    def val_step(
        self,
        *,
        task_model: nn.Module,
        batch: Batch,
        state: State,
        task: TaskT,
        model: ModelT,
    ) -> None:
        with torch.no_grad(), self.autocast_context():
            with self.step_context("change_mode"):
                task_model, state.phase = set_phase(task_model, "valid")
            with self.step_context("forward"):
                loss = task_model(batch, state)
            with self.step_context("get_single_loss"):
                single_loss, loss_names = task.get_single_loss(loss)
            with self.step_context("log_losses"):
                single_loss_detached = single_loss.detach()
                loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
                task.log_loss_dict(loss_dict, state)
            with self.step_context("write_logs"):
                self.write_logs(task, model, state)
            with self.step_context("update_state"):
                state.num_valid_steps += 1

    def test_step(
        self,
        *,
        task_model: nn.Module,
        batch: Batch,
        state: State,
        task: TaskT,
        model: ModelT,
    ) -> None:
        with torch.no_grad(), self.autocast_context():
            with self.step_context("change_mode"):
                task_model, state.phase = set_phase(task_model, "test")
            with self.step_context("forward"):
                loss = task_model(batch, state)
            with self.step_context("get_single_loss"):
                single_loss, loss_names = task.get_single_loss(loss)
            with self.step_context("log_losses"):
                single_loss_detached = single_loss.detach()
                loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
                task.log_loss_dict(loss_dict, state)
            with self.step_context("write_logs"):
                self.write_logs(task, model, state)
            with self.step_context("update_state"):
                state.num_test_steps += 1

    def _init_environment(self) -> None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).name == "main.log":
                root_logger.removeHandler(handler)
        root_logger.addHandler(logging.FileHandler(str((self.log_dir / "main.log").resolve())))

        # Sets up environment.
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True)
        if self.config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Saves the config at the start of training.
        with Timer("saving config", spinner=True):
            self.save_config()
            self.log_run_config()

        # Enables anomaly detection.
        if self.config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True, check_nan=self.config.detect_anomaly_check_nan)

    def _get_optim_and_lr_sched(
        self,
        task_model: nn.Module,
        optimizer: BaseOptimizer,
        lr_scheduler: BaseLRScheduler,
    ) -> tuple[Optimizer, SchedulerAdapter]:
        with Timer("building optimizer", 0.1, spinner=True):
            optim = optimizer.get(task_model)
        with Timer("building learning rate scheduler", 0.1, spinner=True):
            lr_sched = lr_scheduler.get(optim)
        return optim, lr_sched

    def _get_state(
        self,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> State:
        if (ckpt_path := self.get_ckpt_path()).exists():
            return self.load_checkpoint(ckpt_path, task, model, optim, lr_sched)
        return State.init_state()
