from omegaconf import DictConfig

from ml.core.registry import register_launcher, register_trainer


def launch_main(config: DictConfig) -> None:
    """Launches a distributed or multiprocessing training job.

    Args:
        config: The configuration object.
    """

    trainer = register_trainer.build_entry(config)
    launcher = register_launcher.build_entry(config)
    assert trainer is not None, "Trainer not found in config"
    assert launcher is not None, "Launcher not found in config"
    launcher.launch(trainer)
