import logging

from omegaconf import DictConfig, OmegaConf

from ml.core.env import get_project_root, get_stage_dir
from ml.utils.staging import stage_environment

logger: logging.Logger = logging.getLogger(__name__)


def stage_main(config: DictConfig) -> None:
    """Stages the current configuration.

    Args:
        config: The configuration object.

    Raises:
        RuntimeError: If the project root is not set.
    """

    # Stages the currently-imported files.
    project_root = get_project_root()
    if project_root is None:
        raise RuntimeError("Project root is not set.")
    out_dir = stage_environment(project_root, get_stage_dir())
    logger.info("Staged environment to %s", out_dir)

    # Stages the raw config.
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)
    config_id = len(list(config_dir.glob("config_*.yaml")))
    config_path = config_dir / f"config_{config_id}.yaml"
    OmegaConf.save(config, config_path)
    logger.info("Staged config to %s", config_path)
