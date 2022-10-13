import os

import torch.distributed as dist


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


def init_process_group(backend: str | dist.Backend) -> None:
    dist.init_process_group(
        backend=backend,
        world_size=get_world_size(),
        rank=get_rank(),
    )
