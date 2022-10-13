import logging

from omegaconf import DictConfig

from ml.core.registry import register_trainer

logger = logging.getLogger(__name__)


def main(config: DictConfig) -> None:
    """Runs the training loop in a subprocess.

    Args:
        config: The raw config
    """

    trainer = register_trainer.build_entry(config)
    assert trainer is not None, "Trainer is required to launch multiprocessing jobs"
    trainer.launch()


if __name__ == "__main__":
    raise RuntimeError("Don't run this script directly; run `cli.py mp_train ...` instead")
