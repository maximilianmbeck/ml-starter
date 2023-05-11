import logging
import sys
import traceback

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor, nn

from ml.models.parallel import ColumnParallelLinear, RowParallelLinear
from ml.utils.logging import configure_logging
from ml.utils.networking import get_unused_port
from ml.utils.parallel import initialize_parallelism, parallel_group_info
from ml.utils.torch_distributed import init_dist

logger = logging.getLogger(__name__)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Gets the model and pipeline parallel modules.
        self.layer_1 = ColumnParallelLinear(12, 16, bias=False)
        self.layer_2 = RowParallelLinear(16, 8, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y1 = self.layer_2(self.layer_1(x))
        y2 = F.linear(F.linear(x, self.layer_1.master_weight), self.layer_2.master_weight)
        return y1, y2


def func(rank: int, world_size: int, port: int, error_queue: "mp.Queue[str]") -> None:
    configure_logging(rank=rank, world_size=world_size)

    # Hides some nuisance logs.
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("ml.utils.torch_distributed").setLevel(logging.ERROR)

    try:
        # Initializes the distributed process group.
        init_dist(rank, world_size, "127.0.0.1", port, f"tcp://127.0.0.1:{port}", "gloo")

        # Initializes model parallelism.
        initialize_parallelism(model_parallelism=2)

        # Setting the seed across all processes to make sure that the weights
        # initialize to the same values (needed to make the test pass).
        torch.manual_seed(1337)

        # Builds a data-parallel model.
        model = nn.parallel.DistributedDataParallel(
            DummyModel(),
            process_group=parallel_group_info().dp.group,
        )
        model.eval()

        x = torch.randn(world_size, 12)

        # Tests that the forward passes for both models match.
        output_parallel, output_full = model(x)
        assert torch.allclose(output_parallel, output_full, atol=1e-3)

        # Tests that the gradients are the same for both models.
        output_parallel, _ = model(x)
        output_parallel.sum().backward()
        output_full.sum().backward()
        l1_grad_parallel = model.module.layer_1.weight.grad.clone()
        l2_grad_parallel = model.module.layer_2.weight.grad.clone()
        model.zero_grad()
        _, output_full = model(x)
        output_full.sum().backward()
        l1_grad_full = model.module.layer_1.weight.grad.clone()
        l2_grad_full = model.module.layer_2.weight.grad.clone()
        assert torch.allclose(l1_grad_parallel, l1_grad_full * 2, atol=1e-3)
        assert torch.allclose(l2_grad_parallel, l2_grad_full * 2, atol=1e-3)

    except KeyboardInterrupt:
        pass

    except Exception:
        logger.exception("Exception in process %d", rank)
        error_queue.put(traceback.format_exc())
        sys.exit(1)


@pytest.mark.slow
def test_parallel_model() -> None:
    """Tests model parallelism primitives.

    This function launches 8 processes, partitioned into 2 model parallel,
    2 pipeline parallel and 2 data parallel groups. We have a dummy model
    which performs each type of parallelism to ensure that the primitives
    are all working as intended.
    """

    configure_logging()
    ctx = mp.get_context("forkserver")
    error_queues = []
    procs = []
    port = get_unused_port()
    world_size = 4
    for rank in range(world_size):
        error_queue = ctx.SimpleQueue()
        proc = ctx.Process(target=func, args=(rank, world_size, port, error_queue), daemon=False)
        proc.start()
        error_queues.append(error_queue)
        procs.append(proc)
    pctx = mp.ProcessContext(procs, error_queues)
    while not pctx.join():
        pass
    for error_queue in error_queues:
        assert error_queue.empty(), error_queue.get()


if __name__ == "__main__":
    # python -m tests.utils.test_parallel
    test_parallel_model()
