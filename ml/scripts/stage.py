"""Stages the current configuration."""

import logging

from omegaconf import DictConfig, OmegaConf

from ml.core.env import get_stage_dir
from ml.core.registry import project_dirs
from ml.utils.staging import stage_environment

logger: logging.Logger = logging.getLogger(__name__)


def stage_main(config: DictConfig) -> None:
    """Stages the current configuration.

    Args:
        config: The configuration object.
    """
    # Stages the currently-imported files.
    out_dir = stage_environment(project_dirs.paths[1:], get_stage_dir())
    logger.info("Staged environment to %s", out_dir)

    # Stages the raw config.
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)
    config_id = len(list(config_dir.glob("config_*.yaml")))
    config_path = config_dir / f"config_{config_id}.yaml"
    OmegaConf.save(config, config_path)
    logger.info("Staged config to %s", config_path)
