"""Defines a vanilla trainer which doesn't do any device or data manipulation.

This trainer expects the task to handle all the relevant movement of data and
models to their associated devices.

Summary table:

|         | device 1 - N |
|---------|--------------|
| data    | data[:]      |
| step    | model(x)     |
| loss    | E(x, o)      |
"""

import logging
import signal
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Generic, TypeVar, cast

import torch
from omegaconf import II
from torch import Tensor, nn
from torch.optim import Optimizer

from ml.core.common_types import Batch, Loss
from ml.core.config import conf_field
from ml.core.state import State, set_phase
from ml.lr_schedulers.base import BaseLRScheduler, SchedulerAdapter
from ml.optimizers.base import BaseOptimizer
from ml.trainers.base import BaseTrainer, BaseTrainerConfig, ModelT, TaskT
from ml.trainers.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
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
from ml.utils.device.base import Prefetcher
from ml.utils.timer import Timer

logger: logging.Logger = logging.getLogger(__name__)


class TrainingFinishedException(Exception):
    pass


class TaskModel(nn.Module, Generic[ModelT, TaskT, Batch, Loss]):
    def __init__(self, task: TaskT, model: ModelT) -> None:
        super().__init__()

        self.task = task
        self.model = model

    def forward(self, batch: Batch, state: State) -> Loss:
        self.task.on_before_forward_step(self.model, batch, state)
        output = self.task.run_model(self.model, batch, state)
        self.task.on_after_forward_step(self.model, batch, output, state)
        loss: Loss = self.task.compute_loss(self.model, batch, state, output)
        self.task.on_after_compute_loss(self.model, batch, output, loss, state)
        return loss


@dataclass
class TorchCompileConfig:
    enabled: bool = conf_field(II("oc.env:TORCH_COMPILE,0"), help="Enable Torch compilation")
    fullgraph: bool = conf_field(False, help="Whether it is OK to break the model into subgraphs")
    dynamic: bool = conf_field(False, help="Whether to use dynamic shape tracing")
    backend: str = conf_field("auto", help="The backend to use")
    mode: str | None = conf_field("max-autotune", help="Can be either 'default', 'reduce-overhead' or 'max-autotune'")


@dataclass
class VanillaTrainerConfig(
    ProfilerTrainerConfig,
    GradientClippingConfig,
    MixedPrecisionTrainerConfig,
    GPUStatsConfig,
    CPUStatsConfig,
    BaseTrainerConfig,
):
    set_to_none: bool = conf_field(True, help="Mode for clearing optimizer gradients")
    deterministic: bool = conf_field(False, help="If set, use determinstic algorithms")
    use_tf32: bool = conf_field(True, help="If set, use TensorFloat32")
    update_interval: int = conf_field(1, help="How often to update model parameters")
    torch_compile: TorchCompileConfig = conf_field(TorchCompileConfig(), help="Torch compile config")
    detect_anomaly: bool = conf_field(False, help="Whether to detect anomalies")
    detect_anomaly_check_nan: bool = conf_field(False, help="Whether to check for NaNs when detecting anomalies")


VanillaTrainerConfigT = TypeVar("VanillaTrainerConfigT", bound=VanillaTrainerConfig)


class VanillaTrainer(
    ProfilerTrainerMixin[VanillaTrainerConfigT, ModelT, TaskT],
    GradientClippingTrainerMixin[VanillaTrainerConfigT, ModelT, TaskT],
    MixedPrecisionTrainerMixin[VanillaTrainerConfigT, ModelT, TaskT],
    GPUStatsMixin[VanillaTrainerConfigT, ModelT, TaskT],
    CPUStatsMixin[VanillaTrainerConfigT, ModelT, TaskT],
    BaseTrainer[VanillaTrainerConfigT, ModelT, TaskT],
    Generic[VanillaTrainerConfigT, ModelT, TaskT],
):
    def get_task_model_impl(self, task: TaskT, model: ModelT) -> nn.Module:
        device, dtype = self._device.get_device(), self._weight_precision
        model.init(device, dtype)
        task.to(device, dtype)
        task_model: nn.Module = TaskModel(task=task, model=model)
        return task_model

    def get_task_model(self, task: TaskT, model: ModelT) -> nn.Module:
        return self.get_task_model_impl(task, model)

    def train_step(
        self,
        *,
        task_model: nn.Module,
        batch: Batch,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> dict[str, Tensor]:
        with self.step_context("change_mode"):
            task_model, state.phase = set_phase(task_model, "train")
        with self.step_context("forward"), self.autocast_context():
            loss = task_model(batch, state)
        with self.step_context("get_single_loss"):
            single_loss, loss_names = task.get_single_loss(loss)
        with self.step_context("backward"):
            self.scale_mixed_precision(single_loss.sum()).backward()
        with self.step_context("log_losses"):
            self.log_mp_scale()
            single_loss_detached = single_loss.detach()
            loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
            task.log_loss_dict(loss_dict, state)
        if state.num_steps % self.config.update_interval == 0:
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
            bsz = task.get_batch_size(batch)
            if bsz is not None:
                state.num_samples += bsz
                state.num_epoch_samples += bsz
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

    def on_exit(
        self,
        sig: signal.Signals,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_scheduler: SchedulerAdapter,
    ) -> None:
        logger.info("Handling interrupt %s", sig.name)
        self.save_checkpoint(state, task, model, optim, lr_scheduler)

    def set_signal_handler(self, handler: Callable[[int, FrameType | None], None]) -> None:
        pass

    def _init_environment(self) -> None:
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

    def _compile_model(self, model: ModelT) -> ModelT:
        if self.config.torch_compile.enabled:
            backend: str | Callable = self.config.torch_compile.backend
            if backend == "auto":
                backend = self._device.get_torch_compile_backend()
                logger.info("Using torch-compile backend [%s]", backend)

            model = cast(
                ModelT,
                torch.compile(
                    model,
                    fullgraph=self.config.torch_compile.fullgraph,
                    dynamic=self.config.torch_compile.dynamic,
                    backend=backend,
                    mode=self.config.torch_compile.mode,
                    disable=not self.config.torch_compile.enabled,
                ),
            )

        return model

    def _get_optim_and_lr_sched(
        self,
        model: ModelT,
        optimizer: BaseOptimizer,
        lr_scheduler: BaseLRScheduler,
    ) -> tuple[Optimizer, SchedulerAdapter]:
        with Timer("building optimizer", 0.1, spinner=True):
            optim = optimizer.get(model)
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

    def _log_prefetcher_stats(self, pf: Prefetcher) -> None:
        self.logger.log_scalar("num_queued_samples", pf.num_queued_samples, namespace="trainer")
        self.logger.log_scalar("dt/get_batch", pf.get_batch_time, namespace="timers")
        self.logger.log_scalar("dt/to_device", pf.to_device_time, namespace="timers")

    def launch(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support multiprocess training")
