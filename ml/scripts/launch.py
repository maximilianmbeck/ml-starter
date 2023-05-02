import logging

from omegaconf import DictConfig

from ml.core.registry import register_launcher

logger: logging.Logger = logging.getLogger(__name__)


def launch_main(config: DictConfig) -> None:
    """Launches a distributed or multiprocessing training job.

    Args:
        config: The configuration object.
    """

    launcher = register_launcher.build_entry(config)
    assert launcher is not None, "Launcher not found in config"
    launcher.launch()
