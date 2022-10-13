import functools
import logging

from omegaconf import DictConfig

from ml.core.env import get_distributed_backend
from ml.core.registry import Objects, register_trainer
from ml.scripts.train import main as train_main
from ml.trainers.base import MultiprocessConfig
from ml.utils.distributed import (
    init_process_group,
    set_master_addr,
    set_master_port,
    set_rank,
    set_world_size,
)
from ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def process_main(cfg: MultiprocessConfig, raw_config: DictConfig) -> None:
    set_master_addr(cfg.master_addr)
    set_master_port(cfg.master_port)
    set_rank(cfg.rank)
    set_world_size(cfg.world_size)
    init_process_group(backend=get_distributed_backend())

    configure_logging(rank=cfg.rank, world_size=cfg.world_size)

    objs = Objects.parse_raw_config(raw_config)
    train_main(objs)


def main(config: DictConfig) -> None:
    """Runs the training loop in a subprocess.

    Args:
        config: The raw config
    """

    trainer = register_trainer.build_entry(config)
    assert trainer is not None, "Trainer is required to launch multiprocessing jobs"
    trainer.launch(functools.partial(process_main, raw_config=config))


if __name__ == "__main__":
    raise RuntimeError("Don't run this script directly; run `cli.py mp_train ...` instead")
