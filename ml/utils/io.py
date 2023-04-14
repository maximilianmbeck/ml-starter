import logging
from pathlib import Path
from typing import TypeVar, cast

import torch
from omegaconf import DictConfig, OmegaConf

from ml.core.registry import Objects, register_model, register_task
from ml.models.base import BaseModel
from ml.tasks.base import BaseTask
from ml.trainers.base import DummyBaseTrainer
from ml.utils.device.auto import AutoDevice
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)

T = TypeVar("T")


def instantiate_config(config: str | Path | DictConfig | dict) -> Objects:
    """Builds the objects from the raw config.

    Args:
        config: The config to use. If a string or a Path, it is expected to
            be a path to a YAML file.

    Returns:
        The instantiated objects.
    """

    if isinstance(config, (str, Path)):
        config = cast(DictConfig, OmegaConf.load(config))
    elif isinstance(config, dict):
        config = OmegaConf.create(config)
    Objects.resolve_config(config)
    return Objects.parse_raw_config(config)


def get_checkpoint_path(
    trainer: DummyBaseTrainer,
    config_path: str | Path,
    ckpt_path: str | Path | None,
) -> Path:
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            return ckpt_path
        logger.warning("Could not find the passed checkpoint at %s", ckpt_path)

    # Tries loading the checkpoint that the trainer thinks exists.
    ckpt_path = trainer.get_ckpt_path()
    if ckpt_path.exists():
        return ckpt_path
    logger.warning("Could not find trainer checkpoint at %s", ckpt_path)

    # Tries loading other checkpoints.
    config_path = Path(config_path)
    ckpt_path = config_path.parent / "ckpt.pt"
    if ckpt_path.exists():
        return ckpt_path
    logger.warning("Could not find checkpoint at %s", ckpt_path)

    # Searches for a checkpoint in the same directory as the config.
    ckpt_paths = list(config_path.parent.rglob("ckpt*.pt"))
    if ckpt_paths:
        return max(ckpt_paths, key=lambda p: p.stat().st_mtime)
    logger.warning("Could not find checkpoints in config directory %s", config_path.parent)

    raise RuntimeError("Could not find a checkpoint to load")


def load_model_and_task(
    config_path: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    to_device: bool = True,
    missing_ckpt_okay: bool = False,
    weights_only: bool = True,
) -> tuple[BaseModel, BaseTask]:
    """Loads a trained checkpoint from a config, and optional checkpoint path.

    Args:
        config_path: The path to the config file.
        ckpt_path: The path to the checkpoint file; if None, the latest
            checkpoint will be used. This defaults to first checking in an
            adjacent `checkpoints` directory for a `ckpt.pt` file, or else
            checking for the checkpoint file in the same directory as the
            config.
        to_device: Whether to move the model to the device specified in the
            config.
        missing_ckpt_okay: Whether to return a model and task even if the
            checkpoint is missing.
        weights_only: Whether to load only the weights of the model, or the
            entire model.

    Returns:
        The model and task loaded from the checkpoint

    Raises:
        ValueError: If both `config_path` and `ckpt_path` are None.
        RuntimeError: If the checkpoint is missing and `missing_ckpt_okay` is
            False.
    """

    with Timer("loading checkpoint"):
        ckpt: str | Path | dict | None = None

        if config_path is None:
            if ckpt_path is None:
                raise ValueError("Must provide either a config path or a checkpoint path")

            ckpt = cast(dict, torch.load(ckpt_path, map_location="cpu"))
            if "config" not in ckpt:
                raise ValueError("Could not find a config in the checkpoint")
            config = OmegaConf.create(ckpt["config"])
            trainer = DummyBaseTrainer(config.trainer)

        else:
            config = cast(DictConfig, OmegaConf.load(config_path))
            trainer = DummyBaseTrainer(config.trainer)

            # Uses the dummy trainer to load the checkpoint.
            try:
                ckpt = get_checkpoint_path(trainer, config_path, ckpt_path)
            except RuntimeError:
                if missing_ckpt_okay:
                    logger.exception("Could not load checkpoint")
                else:
                    raise

        model = register_model.build_entry_non_null(config)
        task = register_task.build_entry_non_null(config)
        if ckpt is not None:
            trainer.load_checkpoint(ckpt, task, model, weights_only=weights_only)

        if to_device:
            device = AutoDevice.detect_device()
            device.module_to(model)
            device.module_to(task)

    return model, task
