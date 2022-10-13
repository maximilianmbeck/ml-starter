import logging

from omegaconf import OmegaConf

from ml.core.registry import Objects, stage_environment

logger = logging.getLogger(__name__)


def main(objs: Objects) -> None:
    """Stages the current configuration.

    Args:
        objs: The parsed objects
    """

    # Stages the currently-imported files.
    out_dir = stage_environment()
    logger.info("Staged environment to %s", out_dir)

    # Stages the raw config.
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)
    config_id = len(list(config_dir.glob("config_*.yaml")))
    config_path = config_dir / f"config_{config_id}.yaml"
    OmegaConf.save(objs.raw_config, config_path)
    logger.info("Staged config to %s", config_path)


if __name__ == "__main__":
    raise RuntimeError("Don't run this script directly; run `cli.py stage ...` instead")
