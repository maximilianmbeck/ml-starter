from omegaconf import DictConfig

from ml.core.registry import Objects
from ml.utils.timer import Timer


def train_main(config: DictConfig) -> None:
    """Runs the training loop.

    Args:
        config: The configuration object.
    """

    with Timer("setting random seed", spinner=True):
        from ml.utils.random import set_random_seed

        set_random_seed()

    objs = Objects.parse_raw_config(config)

    train_main_with_objects(objs)


def train_main_with_objects(objs: Objects) -> None:
    """Runs the training loop.

    Args:
        objs: The objects to use for training.
    """

    # Checks that the config has the right keys for training.
    assert (model := objs.model) is not None
    assert (task := objs.task) is not None
    assert (optimizer := objs.optimizer) is not None
    assert (lr_scheduler := objs.lr_scheduler) is not None
    assert (trainer := objs.trainer) is not None

    # Runs the training loop.
    with Timer("running training loop"):
        trainer.train(model, task, optimizer, lr_scheduler)
