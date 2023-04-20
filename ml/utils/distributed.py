import logging
import os
import random
import time

logger: logging.Logger = logging.getLogger(__name__)


def set_rank(rank: int) -> None:
    os.environ["RANK"] = str(rank)


def get_rank_optional() -> int | None:
    rank = os.environ.get("RANK")
    return None if rank is None else int(rank)


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def set_world_size(world_size: int) -> None:
    os.environ["WORLD_SIZE"] = str(world_size)


def get_world_size_optional() -> int | None:
    world_size = os.environ.get("WORLD_SIZE")
    return None if world_size is None else int(world_size)


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def set_master_addr(master_addr: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr


def get_master_addr() -> str:
    return os.environ["MASTER_ADDR"]


def set_master_port(port: int) -> None:
    os.environ["MASTER_PORT"] = str(port)


def get_master_port() -> int:
    return int(os.environ["MASTER_PORT"])


def is_master() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return "INIT_METHOD" in os.environ


def get_init_method() -> str:
    return os.environ["INIT_METHOD"]


def set_init_method(init_method: str) -> None:
    os.environ["INIT_METHOD"] = init_method


def get_random_port() -> int:
    return (hash(time.time()) + random.randint(0, 100000)) % (65_535 - 10_000) + 10_000
