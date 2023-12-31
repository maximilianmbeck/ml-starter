"""Defines the base trainer class and config.

The trainer is the thing that actually runs the training loop. There are
separate trainers for supervised and reinforcement learning since the latter
requires interacting with an environment, so you use the appropriate trainer
for your task (they are defined in :mod:`ml.trainers.sl` and
:mod:`ml.trainers.rl` respectively). The base trainer handles things like
setting up the experiment directory, saving checkpoints, and logging.
"""

import enum
import functools
import logging
import os
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Callable, Generic, Literal, TypeVar, cast, get_args

import torch
from omegaconf import II, MISSING, DictConfig, ListConfig, OmegaConf
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ml.core.config import BaseConfig, BaseObject, conf_field
from ml.core.env import get_ml_config_path
from ml.core.state import State
from ml.loggers.base import BaseLogger
from ml.loggers.multi import MultiLogger
from ml.lr_schedulers.base import BaseLRScheduler, SchedulerAdapter
from ml.models.base import BaseModel
from ml.optimizers.base import BaseOptimizer
from ml.tasks.base import BaseTask
from ml.utils.colors import colorize
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice, Prefetcher
from ml.utils.distributed import is_master
from ml.utils.timer import Timer

logger: logging.Logger = logging.getLogger(__name__)


def abs_path(path: str) -> str:
    return str(Path(path).resolve())


def cpu_count(default: int) -> int:
    if (cpu_count := os.cpu_count()) is not None:
        return cpu_count
    return default


OmegaConf.register_new_resolver("ml.abs_path", abs_path, replace=True)
OmegaConf.register_new_resolver("ml.cpu_count", cpu_count, replace=True)

LockType = Literal["running", "scheduled", "ckpt"]


def add_lock_file(exp_dir: Path, lock_type: LockType, *, exists_ok: bool = False) -> None:
    if is_master():
        if (lock_file := exp_dir / f".lock_{lock_type}").exists():
            if not exists_ok:
                raise RuntimeError(f"Lock file already exists at {lock_file}")
        else:
            with open(lock_file, "w", encoding="utf-8") as f:
                f.write(f"PID: {os.getpid()}")
            logger.debug("Added %s lock file to experiment directory %s", lock_type, exp_dir)


def remove_lock_file(exp_dir: Path, lock_type: LockType, *, missing_ok: bool = False) -> None:
    if is_master():
        if (lock_file := exp_dir / f".lock_{lock_type}").exists():
            lock_file.unlink()
            logger.debug("Removed %s lock file from experiment directory %s", lock_type, exp_dir)
        elif not missing_ok:
            raise RuntimeError(f"Lock file not found at {lock_file}")


def has_lock_file(exp_dir: Path, lock_type: LockType | None = None) -> bool:
    if lock_type is not None:
        return (exp_dir / f".lock_{lock_type}").exists()
    return any((exp_dir / f".lock_{lock_type_arg}").exists() for lock_type_arg in get_args(LockType))


def get_ckpt_path(exp_dir: Path, state: State | None = None) -> Path:
    """Defines the path to the checkpoint for a given state.

    Args:
        exp_dir: The experiment directory
        state: The current trainer state

    Returns:
        The path to the PyTorch checkpoint to save or load
    """
    if state is None:
        return exp_dir / "checkpoints" / "ckpt.pt"
    return exp_dir / "checkpoints" / f"ckpt.{state.num_steps}.pt"


def get_exp_dir(run_dir: Path, run_id: int) -> Path:
    return (run_dir / f"run_{run_id}").resolve()


def get_empty_exp_dir(run_dir: Path) -> Path:
    """Returns the path to the run directory, given a run ID.

    Args:
        run_dir: The base run directory for the experiment

    Returns:
        An experiment directory without a lockfile
    """
    # If the run ID isn't specified, look at all run IDs until there is one
    # which either doesn't exist or doesn't have a checkpoint directory.
    run_id = 0
    while (exp_dir := get_exp_dir(run_dir, run_id)).is_dir() and has_lock_file(exp_dir):
        run_id += 1

    return get_exp_dir(run_dir, run_id)


def diff_configs(
    first: ListConfig | DictConfig,
    second: ListConfig | DictConfig,
    prefix: str | None = None,
) -> tuple[list[str], list[str]]:
    """Returns the difference between two configs.

    Args:
        first: The first (original) config
        second: The second (new) config
        prefix: The prefix to check (used for recursion, not main call)

    Returns:
        Two lists of lines describing the diff between the two configs
    """

    def get_diff_string(prefix: str | None, val: Any) -> str:  # noqa: ANN401
        if isinstance(val, (str, float, int)):
            return f"{prefix}={val}"
        return f"{prefix}= ... ({type(val)})"

    def cast_enums(k: Any) -> Any:  # noqa: ANN401
        return k.name if isinstance(k, enum.Enum) else k

    new_first: list[str] = []
    new_second: list[str] = []

    any_config = (ListConfig, DictConfig)

    if isinstance(first, DictConfig) and isinstance(second, DictConfig):
        first_keys, second_keys = cast(set[str], set(first.keys())), cast(set[str], set(second.keys()))

        # Gets the new keys in each config.
        new_first += [f"{prefix}.{key}" for key in first_keys.difference(second_keys)]
        new_second += [f"{prefix}.{key}" for key in second_keys.difference(first_keys)]

        # Gets the new sub-keys in each config.
        for key in first_keys.intersection(second_keys):
            sub_prefix = key if prefix is None else f"{prefix}.{key}"
            if OmegaConf.is_missing(first, key) or OmegaConf.is_missing(second, key):
                if not OmegaConf.is_missing(first, key):
                    new_first += [get_diff_string(sub_prefix, first[key])]
                if not OmegaConf.is_missing(second, key):
                    new_second += [get_diff_string(sub_prefix, second[key])]
            elif isinstance(first[key], any_config) and isinstance(second[key], any_config):
                sub_new_first, sub_new_second = diff_configs(first[key], second[key], prefix=sub_prefix)
                new_first, new_second = new_first + sub_new_first, new_second + sub_new_second
            elif cast_enums(first[key]) != cast_enums(second[key]):
                first_val, second_val = first[key], second[key]
                new_first += [get_diff_string(sub_prefix, first_val)]
                new_second += [get_diff_string(sub_prefix, second_val)]

    elif isinstance(first, ListConfig) and isinstance(second, ListConfig):
        if len(first) > len(second):
            for i in range(len(second), len(first)):
                new_first += [get_diff_string(prefix, first[i])]
        elif len(second) > len(first):
            for i in range(len(first), len(second)):
                new_second += [get_diff_string(prefix, second[i])]

        for i in range(min(len(first), len(second))):
            sub_prefix = str(i) if prefix is None else f"{prefix}.{i}"
            if isinstance(first[i], any_config) and isinstance(second[i], any_config):
                sub_new_first, sub_new_second = diff_configs(first[i], second[i], prefix=sub_prefix)
                new_first, new_second = new_first + sub_new_first, new_second + sub_new_second
    else:
        new_first += [get_diff_string(prefix, first)]
        new_second += [get_diff_string(prefix, second)]

    return new_first, new_second


def save_config(config_path: Path, raw_config: DictConfig) -> None:
    if is_master():
        if config_path.exists():
            added_keys, deleted_keys = diff_configs(raw_config, cast(DictConfig, OmegaConf.load(config_path)))
            if added_keys or deleted_keys:
                change_lines: list[str] = []
                change_lines += [f" ↪ {colorize('+', 'green')} {added_key}" for added_key in added_keys]
                change_lines += [f" ↪ {colorize('-', 'red')} {deleted_key}" for deleted_key in deleted_keys]
                change_summary = "\n".join(change_lines)
                logger.warning("Overwriting config %s:\n%s", config_path, change_summary)
                OmegaConf.save(raw_config, config_path)
        else:
            config_path.parent.mkdir(exist_ok=True, parents=True)
            OmegaConf.save(raw_config, config_path)
            logger.info("Saved config to %s", config_path)


@dataclass
class CheckpointConfig:
    save_every_n_steps: int | None = conf_field(None, help="Save a checkpoint every N steps")
    only_save_most_recent: bool = conf_field(False, help="Only keep the most recent checkpoint")
    load_from_ckpt_path: str | None = conf_field(None, help="If set, load initial model weights from this path")


@dataclass
class BaseTrainerConfig(BaseConfig):
    """Defines the base config for all trainers."""

    base_run_dir: str = conf_field(II("ml.abs_path:${oc.env:RUN_DIR}"), help="The base directory for all runs")
    exp_name: str = conf_field(II("ml.exp_name:null"), help="The name of the training job")
    exp_dir: str = conf_field(MISSING, help="The directory where the experiment is stored")
    log_dir_name: str = conf_field("logs", help="Name of the subdirectory which contains logs")
    use_double_weight_precision: bool = conf_field(False, help="If set, use doubles for weights instead of floats")
    checkpoint: CheckpointConfig = conf_field(CheckpointConfig())

    @classmethod
    def resolve(cls, config: "BaseTrainerConfig") -> None:
        if OmegaConf.is_missing(config, "exp_dir"):
            config.exp_dir = str(get_empty_exp_dir(Path(config.base_run_dir) / config.exp_name))
        elif (ml_config_path := get_ml_config_path()) is not None:
            exp_dir_path = Path(config.exp_dir).resolve()
            if exp_dir_path != ml_config_path:
                logger.warning(
                    "The `config.yaml` file is located in a different directory than the experiment directory; "
                    "updating `exp_dir` to match the new location."
                )
                config.exp_dir = str(ml_config_path)
        super().resolve(config)


TrainerConfigT = TypeVar("TrainerConfigT", bound=BaseTrainerConfig)
ModelT = TypeVar("ModelT", bound=BaseModel)
TaskT = TypeVar("TaskT", bound=BaseTask)


class BaseTrainer(BaseObject[TrainerConfigT], Generic[TrainerConfigT, ModelT, TaskT]):
    """Defines the base trainer type."""

    logger: MultiLogger
    loggers: list[BaseLogger]

    def __init__(self, config: TrainerConfigT) -> None:
        super().__init__(config)

        self.exp_name = config.exp_name
        self.exp_dir = Path(config.exp_dir)
        self.log_dir = self.exp_dir / config.log_dir_name
        self.checkpoint_config = config.checkpoint
        self.loggers = []
        self.logger = MultiLogger(default_namespace="trainer")
        self.signal_handlers: dict[signal.Signals, list[Callable[[], None]]] = {}

        logger.info("Experiment directory: %s", self.exp_dir)

    @functools.cached_property
    def _device(self) -> type[BaseDevice]:
        dev = AutoDevice.detect_device()
        device, dtype = dev.get_device(), dev.get_floating_point_type()
        self.logger.log_string("device", f"{device.type}/{device.index} - {dtype}")
        return dev

    @functools.cached_property
    def _device_type(self) -> str:
        return self._device.get_device().type

    @functools.cached_property
    def _weight_precision(self) -> torch.dtype:
        # Weights always have to be FP32 or FP64, because AMP doesn't like
        # gradients which are in FP16.
        return torch.float64 if self.config.use_double_weight_precision else torch.float32

    def add_logger(self, sublogger: BaseLogger) -> None:
        sublogger.initialize(self.log_dir)
        self.loggers += [sublogger]

    def add_loggers(self, subloggers: list[BaseLogger]) -> None:
        for sublogger in subloggers:
            self.add_logger(sublogger)

    @property
    def config_path(self) -> Path:
        return self.exp_dir / "config.yaml"

    def save_config(self) -> None:
        save_config(self.config_path, self.raw_config)

    def log_run_config(self) -> None:
        if is_master():
            for logger in self.loggers:
                logger.log_config(self.raw_config)

    def add_lock_file(self, lock_type: LockType, *, exists_ok: bool = False) -> None:
        add_lock_file(self.exp_dir, lock_type=lock_type, exists_ok=exists_ok)

    def remove_lock_file(self, lock_type: LockType, *, missing_ok: bool = False) -> None:
        remove_lock_file(self.exp_dir, lock_type=lock_type, missing_ok=missing_ok)

    def get_ckpt_path(self, state: State | None = None) -> Path:
        return get_ckpt_path(self.exp_dir, state)

    @property
    def ckpt_path(self) -> Path:
        return self.get_ckpt_path()

    def should_checkpoint(self, state: State) -> bool:
        if self.checkpoint_config.save_every_n_steps is not None:
            if state.num_steps % self.checkpoint_config.save_every_n_steps == 0:
                return True
        return False

    def load_checkpoint(
        self,
        ckpt: str | Path | dict,
        task: TaskT,
        model: ModelT,
        optims: Optimizer | dict[str, Optimizer] | None = None,
        lr_scheds: SchedulerAdapter | dict[str, SchedulerAdapter] | None = None,
        *,
        weights_only: bool = True,
    ) -> State:
        """Loads a given checkpoint, from a path or dictionary.

        Args:
            ckpt: The checkpoint to load.
            task: The task to load the checkpoint into.
            model: The model to load the checkpoint into.
            optims: The optimizer to load the checkpoint into.
            lr_scheds: The learning rate scheduler to load the checkpoint into.
            weights_only: If set, only load the model weights.

        Returns:
            The state loaded from the checkpoint.

        Raises:
            UnpicklingError: If there is some issue unpickling the checkpoint.
        """
        with Timer("loading checkpoint", spinner=True):
            if isinstance(ckpt, (str, Path)):
                try:
                    ckpt = cast(dict, torch.load(ckpt, weights_only=weights_only))
                except UnpicklingError:
                    if weights_only:
                        logger.warning("Failed to load checkpoint using `weights_only` flag, retrying without it")
                        ckpt = cast(dict, torch.load(cast(str | Path, ckpt), weights_only=False))
                    else:
                        raise

            task.on_after_load_checkpoint(ckpt)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                logger.warning("Checkpoint does not contain a model state dict")
            if "task" in ckpt:
                task.load_state_dict(ckpt["task"])
            else:
                logger.warning("Checkpoint does not contain a task state dict")
            if optims is not None:
                if "optim" in ckpt:
                    if isinstance(optims, dict):
                        for name, optim in optims.items():
                            optim.load_state_dict(ckpt["optim"][name])
                    else:
                        optims.load_state_dict(ckpt["optim"])
                else:
                    logger.warning("Checkpoint does not contain an optimizer state dict")
            if lr_scheds is not None:
                if "lr_sched" in ckpt:
                    if isinstance(lr_scheds, dict):
                        for name, lr_sched in lr_scheds.items():
                            lr_sched.load_state_dict(ckpt["lr_sched"][name])
                    else:
                        lr_scheds.load_state_dict(ckpt["lr_sched"])
                else:
                    logger.warning("Checkpoint does not contain a learning rate scheduler state dict")
            self.load_state_dict(ckpt)
            state = State(**ckpt["state"])

        return state

    def save_checkpoint(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optims: Optimizer | dict[str, Optimizer] | None = None,
        lr_scheds: SchedulerAdapter | dict[str, SchedulerAdapter] | None = None,
    ) -> Path:
        ckpt_path = self.get_ckpt_path(state)
        if is_master():
            with Timer("saving checkpoint", spinner=True):
                logger.info("Saving checkpoint to %s", ckpt_path)
                last_ckpt_path = self.get_ckpt_path()
                ckpt_path.parent.mkdir(exist_ok=True, parents=True)

                state_dict: dict[str, Any] = {
                    "model": model.state_dict(),
                    "task": task.state_dict(),
                    "state": asdict(state),
                }

                if optims is not None:
                    if isinstance(optims, dict):
                        state_dict["optim"] = {k: v.state_dict() for k, v in optims.items()}
                    else:
                        state_dict["optim"] = optims.state_dict()

                if lr_scheds is not None:
                    if isinstance(lr_scheds, dict):
                        state_dict["lr_sched"] = {k: v.state_dict() for k, v in lr_scheds.items()}
                    else:
                        state_dict["lr_sched"] = lr_scheds.state_dict()

                if self._raw_config is not None:
                    state_dict["config"] = OmegaConf.to_container(self._raw_config, enum_to_str=True)
                self.update_state_dict(state_dict)

                if last_ckpt_path.exists():
                    if self.checkpoint_config.only_save_most_recent:
                        base_ckpt = last_ckpt_path.resolve()
                        if base_ckpt.is_file():
                            base_ckpt.unlink()
                    last_ckpt_path.unlink()
                torch.save(state_dict, ckpt_path)

                try:
                    last_ckpt_path.symlink_to(ckpt_path)
                except FileExistsError:
                    logger.exception("Exception while trying to update %s", ckpt_path)
                self.add_lock_file("ckpt", exists_ok=True)
                task.on_after_save_checkpoint(ckpt_path)

        return ckpt_path

    def train(self, model: ModelT, task: TaskT, optimizer: BaseOptimizer, lr_scheduler: BaseLRScheduler) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError

    def write_logs(self, task: TaskT, model: ModelT, state: State) -> None:
        model.logger.write(self.loggers, state)
        task.logger.write(self.loggers, state)
        self.logger.write(self.loggers, state)
        for value_logger in self.loggers:
            if value_logger.should_write(state):
                value_logger.write(state)

    def load_state_dict(self, ckpt: dict[str, Any]) -> None:
        """Function for loading state dict keys for different components.

        Args:
            ckpt: The loaded state dictionary
        """

    def update_state_dict(self, ckpt: dict[str, Any]) -> None:
        """Function for getting the checkpoint to save.

        Args:
            ckpt: The checkpoint being saved (overriders should mutate inplace)
        """

    def on_exit(
        self,
        sig: signal.Signals,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_scheduler: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        logger.info("Handling interrupt %s", sig.name)
        self.save_checkpoint(state, task, model, optim, lr_scheduler)
        for signal_handler in self.signal_handlers.get(sig, []):
            signal_handler()

    def add_signal_handler(self, sig: signal.Signals, handler: Callable[[], None]) -> None:
        self.signal_handlers[sig].append(handler)

    def _log_prefetcher_stats(self, pf: Prefetcher) -> None:
        self.logger.log_scalar("dt/get_batch", pf.get_batch_time, namespace="timers")
        self.logger.log_scalar("dt/to_device", pf.to_device_time, namespace="timers")

    # -----
    # Hooks
    # -----

    def on_step_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        task.on_step_start(state, model, optim, lr_sched)

    def on_step_end(
        self,
        state: State,
        loss_dict: dict[str, Tensor],
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        task.on_step_end(state, loss_dict, model, optim, lr_sched)

    def on_epoch_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        task.on_epoch_start(state, model, optim, lr_sched)

    def on_epoch_end(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        task.on_epoch_end(state, model, optim, lr_sched)

    def on_training_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        task.on_training_start(state, model, optim, lr_sched)
        self.add_lock_file("running", exists_ok=True)

    def on_training_end(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer | dict[str, Optimizer],
        lr_sched: SchedulerAdapter | dict[str, SchedulerAdapter],
    ) -> None:
        task.on_training_end(state, model, optim, lr_sched)
        self.remove_lock_file("running", missing_ok=True)
        logger.info("Exiting training job for %s", self.exp_dir / "config.yaml")
