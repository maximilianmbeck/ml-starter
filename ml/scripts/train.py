from ml.core.registry import Objects
from ml.utils.timer import Timer


def main(objs: Objects) -> None:
    """Runs the training loop.

    Args:
        objs: The parsed objects
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


if __name__ == "__main__":
    raise RuntimeError("Don't run this script directly; run `cli.py train ...` instead")
