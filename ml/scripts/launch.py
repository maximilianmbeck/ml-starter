"""Launches a distributed or multiprocessing training job."""

from omegaconf import DictConfig

from ml.core.registry import register_launcher


def launch_main(config: DictConfig) -> None:
    """Launches a distributed or multiprocessing training job.

    Args:
        config: The configuration object.
    """
    launcher = register_launcher.build_entry(config)
    assert launcher is not None, "Launcher not found in config"
    launcher.launch()
