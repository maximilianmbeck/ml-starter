import enum
import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Generic, Literal, TypeVar, cast, get_args

import torch
from omegaconf import II, MISSING, DictConfig, ListConfig, OmegaConf
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ml.core.common_types import Batch
from ml.core.config import BaseConfig, BaseObjectWithPointers, conf_field
from ml.core.state import State
from ml.loggers.base import BaseLogger
from ml.loggers.multi import MultiLogger
from ml.lr_schedulers.base import BaseLRScheduler, SchedulerAdapter
from ml.models.base import BaseModel
from ml.optimizers.base import BaseOptimizer
from ml.tasks.base import BaseTask
from ml.utils.colors import colorize
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.distributed import is_master
from ml.utils.timer import Timer

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MultiprocessConfig:
    rank: int
    world_size: int
    devices_per_rank: int
    master_addr: str
    master_port: int


def resolve(path: str) -> str:
    return str(Path(path).resolve())


def cpu_count(default: int) -> int:
    if (cpu_count := os.cpu_count()) is not None:
        return cpu_count
    return default


OmegaConf.register_new_resolver("resolve", resolve)
OmegaConf.register_new_resolver("cpu_count", cpu_count)

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
        else:
            if not missing_ok:
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


def get_exp_dir(base_run_dir: Path, exp_name: str, run_id: int) -> Path:
    return (base_run_dir / exp_name / f"run_{run_id}").resolve()


def get_run_id(base_run_dir: Path, exp_name: str) -> int:
    """Returns the path to the run directory, given a run ID.

    Args:
        base_run_dir: The base run directory for the entire experiment
        exp_name: The name of the experiment

    Returns:
        A run ID for an experiment directory without a lockfile
    """

    # If the run ID isn't specified, look at all run IDs until there is one
    # which either doesn't exist or doesn't have a checkpoint directory.
    run_id = 0
    while (exp_dir := get_exp_dir(base_run_dir, exp_name, run_id)).is_dir() and has_lock_file(exp_dir):
        run_id += 1

    return run_id


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

    def get_diff_string(prefix: str | None, val: Any) -> str:
        if isinstance(val, (str, float, int)):
            return f"{prefix}={val}"
        return f"{prefix}= ... ({type(val)})"

    def cast_enums(k: Any) -> Any:
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


@dataclass
class BaseTrainerConfig(BaseConfig):
    """Defines the base config for all trainers."""

    exp_name: str = conf_field(II("exp_name:null"), help="The name of the training job")
    log_dir_name: str = conf_field("logs", help="Name of the subdirectory which contains logs")
    base_run_dir: str = conf_field(II("resolve:${oc.env:RUN_DIR}"), help="The base directory for all runs")
    run_id: int = conf_field(MISSING, help="The run ID to use")
    use_double_weight_precision: bool = conf_field(False, help="If set, use doubles for weights instead of floats")
    checkpoint: CheckpointConfig = conf_field(CheckpointConfig())

    @classmethod
    def resolve(cls, config: "BaseTrainerConfig") -> None:
        if OmegaConf.is_missing(config, "run_id"):
            config.run_id = get_run_id(Path(config.base_run_dir), config.exp_name)
        super().resolve(config)


TrainerConfigT = TypeVar("TrainerConfigT", bound=BaseTrainerConfig)
ModelT = TypeVar("ModelT", bound=BaseModel)
TaskT = TypeVar("TaskT", bound=BaseTask)


class BaseTrainer(BaseObjectWithPointers[TrainerConfigT], Generic[TrainerConfigT, ModelT, TaskT], ABC):
    """Defines the base trainer type."""

    logger: MultiLogger
    loggers: list[BaseLogger]

    def __init__(self, config: TrainerConfigT) -> None:
        super().__init__(config)

        self.exp_dir = get_exp_dir(Path(config.base_run_dir), config.exp_name, config.run_id)
        self.log_dir = self.exp_dir / config.log_dir_name
        self.checkpoint_config = config.checkpoint
        self.loggers = []
        self.logger = MultiLogger(default_namespace="trainer")

        logger.info("Experiment directory: %s", self.exp_dir)

    @functools.cached_property
    def _device(self) -> type[BaseDevice]:
        return AutoDevice.detect_device()

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
        optim: Optimizer | None = None,
        lr_sched: SchedulerAdapter | None = None,
        *,
        weights_only: bool = True,
    ) -> State:
        """Loads a given checkpoint, from a path or dictionary.

        Args:
            ckpt: The checkpoint to load.
            task: The task to load the checkpoint into.
            model: The model to load the checkpoint into.
            optim: The optimizer to load the checkpoint into.
            lr_sched: The learning rate scheduler to load the checkpoint into.
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
            if optim is not None:
                if "optim" in ckpt:
                    optim.load_state_dict(ckpt["optim"])
                else:
                    logger.warning("Checkpoint does not contain an optimizer state dict")
            if lr_sched is not None:
                if "lr_sched" in ckpt:
                    lr_sched.load_state_dict(ckpt["lr_sched"])
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
        optim: Optimizer | None = None,
        lr_sched: SchedulerAdapter | None = None,
    ) -> None:
        if is_master():
            with Timer("saving checkpoint", spinner=True):
                ckpt_path = self.get_ckpt_path(state)
                logger.info("Saving checkpoint to %s", ckpt_path)
                last_ckpt_path = self.get_ckpt_path()
                ckpt_path.parent.mkdir(exist_ok=True, parents=True)
                state_dict: dict[str, Any] = {
                    "model": model.state_dict(),
                    "task": task.state_dict(),
                    "optim": None if optim is None else optim.state_dict(),
                    "lr_sched": None if lr_sched is None else lr_sched.state_dict(),
                    "state": asdict(state),
                }
                if self._raw_config is not None:
                    state_dict["config"] = OmegaConf.to_container(self._raw_config)
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

    @abstractmethod
    def launch(self) -> None:
        """Launches a multiprocess command."""

    @abstractmethod
    def train(self, model: ModelT, task: TaskT, optimizer: BaseOptimizer, lr_scheduler: BaseLRScheduler) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler
        """

    def write_logs(self, task: TaskT, model: ModelT, state: State) -> None:
        model.logger.write(self.loggers, state)
        task.logger.write(self.loggers, state)
        self.logger.write(self.loggers, state)
        for value_logger in self.loggers:
            if value_logger.should_write(state):
                value_logger.write(state)
            value_logger.clear(state)

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

    # -----
    # Hooks
    # -----

    def on_step_start(
        self,
        state: State,
        train_batch: Batch,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        task.on_step_start(state, train_batch, model, optim, lr_sched)

    def on_step_end(
        self,
        state: State,
        train_batch: Batch,
        loss_dict: dict[str, Tensor],
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        task.on_step_end(state, train_batch, loss_dict, model, optim, lr_sched)

    def on_epoch_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        task.on_epoch_start(state, model, optim, lr_sched)

    def on_epoch_end(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        task.on_epoch_end(state, model, optim, lr_sched)

    def on_training_start(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        task.on_training_start(state, model, optim, lr_sched)
        self.add_lock_file("running", exists_ok=True)

    def on_training_end(
        self,
        state: State,
        task: TaskT,
        model: ModelT,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        task.on_training_end(state, model, optim, lr_sched)
        self.remove_lock_file("running", missing_ok=True)
        logger.info("Exiting training job for %s", self.exp_dir / "config.yaml")


class DummyBaseTrainer(BaseTrainer):
    """Defines a dummy trainer that does nothing.

    This trainer can be used to access the base trainer's utility functions,
    such as saving and loading checkpoints.
    """

    def launch(self) -> None:
        raise NotImplementedError

    def train(self, model: ModelT, task: TaskT, optimizer: BaseOptimizer, lr_scheduler: BaseLRScheduler) -> None:
        raise NotImplementedError
