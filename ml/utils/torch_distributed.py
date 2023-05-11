import logging
import os

import torch
import torch.distributed as dist

from ml.utils.distributed import get_init_method, get_rank, get_world_size, set_dist
from ml.utils.logging import INFOALL

logger: logging.Logger = logging.getLogger(__name__)


def init_process_group_from_backend(backend: str | dist.Backend | None = None) -> None:
    if backend is None:
        backend = get_distributed_backend()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.log(INFOALL, "CUDA visible devices: %s", os.environ["CUDA_VISIBLE_DEVICES"])
    init_method, world_size, rank = get_init_method(), get_world_size(), get_rank()
    logger.log(INFOALL, "Initializing %d / %d using %s - %s", rank, world_size, init_method, backend)
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.log(INFOALL, "Finished initializing %d / %d with %d device(s)", rank, world_size, device_count)
        dist.all_reduce(torch.zeros(1).cuda())
    else:
        logger.log(INFOALL, "Finished initializing %d / %d", rank, world_size)
    logger.log(INFOALL, "Dummy all-reduce succeeded")


def init_dist(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    init_method: str,
    backend: str | dist.Backend | None = None,
) -> None:
    """Initializes distributed environment.

    Args:
        rank: The rank of the current process.
        world_size: The total number of processes.
        master_addr: The address of the master process.
        master_port: The port of the master process.
        init_method: The initialization method.
        backend: The distributed backend.
    """

    set_dist(rank, world_size, master_addr, master_port, init_method)
    init_process_group_from_backend(backend)


def get_distributed_backend() -> dist.Backend:
    # Used to change the distributed backend to something other than NCCL.
    # For example, if you're on a system with some strange NCCL errors, you
    # can try changing this environment variable to `gloo`.
    return dist.Backend(os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"))


def set_distributed_backend(backend: str) -> None:
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = backend
