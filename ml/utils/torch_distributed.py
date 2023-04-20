import logging
import os

import torch
import torch.distributed as dist

from ml.utils.distributed import get_init_method, get_rank, get_world_size
from ml.utils.logging import INFOALL

logger: logging.Logger = logging.getLogger(__name__)


def init_process_group(backend: str | dist.Backend) -> None:
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


def get_distributed_backend() -> dist.Backend:
    # Used to change the distributed backend to something other than NCCL.
    # For example, if you're on a system with some strange NCCL errors, you
    # can try changing this environment variable to `gloo`.
    return dist.Backend(os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"))
