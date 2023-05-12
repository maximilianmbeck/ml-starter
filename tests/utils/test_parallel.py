import logging
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ml.models.parallel import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from ml.trainers.mixins.data_parallel import ModelConfig, ddp
from ml.utils.logging import configure_logging
from ml.utils.networking import get_unused_port
from ml.utils.torch_distributed import MultiprocessConfig, launch_subprocesses

logger = logging.getLogger(__name__)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # A simple embedding layer plus two-layer MLP.
        self.emb = ParallelEmbedding(10, 12)
        self.l1 = ColumnParallelLinear(12, 16, bias=False)
        self.l2 = RowParallelLinear(16, 8, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y1 = self.l2(self.l1(self.emb(x)))
        y2 = F.linear(F.linear(F.embedding(x, self.emb.master_weight), self.l1.master_weight), self.l2.master_weight)
        return y1, y2


def setup() -> None:
    # Hides some nuisance logs.
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.ERROR)
    logging.getLogger("ml.utils.torch_distributed").setLevel(logging.ERROR)

    # Setting the seed across all processes to make sure that the weights
    # initialize to the same values (needed to make the test pass).
    torch.manual_seed(1337)


def func() -> None:
    config = ModelConfig(use_fsdp=False)
    model = ddp(DummyModel(), config)
    model.eval()
    base_model = cast(DummyModel, model.module)

    def get_grad(g: Tensor | None) -> Tensor:
        assert g is not None
        return g.clone()

    x = torch.randint(0, 10 - 1, (4, 12))

    # Tests that the forward passes for both models match.
    output_parallel, output_full = model(x)
    assert torch.allclose(output_parallel, output_full, atol=1e-3)

    # Backpropagates the parallel outputs.
    output_parallel, _ = model(x)
    output_parallel.sum().backward()
    emb_grad_parallel = get_grad(base_model.emb.weight.grad)
    l1_grad_parallel = get_grad(base_model.l1.weight.grad)
    l2_grad_parallel = get_grad(base_model.l2.weight.grad)
    model.zero_grad()

    # Backpropagates the full outputs.
    _, output_full = model(x)
    output_full.sum().backward()
    emb_grad_full = get_grad(base_model.emb.weight.grad)
    l1_grad_full = get_grad(base_model.l1.weight.grad)
    l2_grad_full = get_grad(base_model.l2.weight.grad)

    # Checks that the gradients for the parallel outputs and the full outputs
    # match - this is effectively checking that the implementations of the
    # model parallel modules are correct.
    assert torch.allclose(emb_grad_parallel, emb_grad_full, atol=1e-3)
    assert torch.allclose(l1_grad_parallel, l1_grad_full, atol=1e-3)
    assert torch.allclose(l2_grad_parallel, l2_grad_full, atol=1e-3)


@pytest.mark.slow
def test_parallel_model() -> None:
    """Tests model parallelism primitives.

    This function launches 4 processes, partitioned into 2 model parallel and
    2 data parallel groups. We check that the partitioned model outputs and
    gradients match the full model.
    """
    configure_logging()

    port = get_unused_port()

    config = MultiprocessConfig(
        world_size=4,
        master_addr="127.0.0.1",
        master_port=port,
        init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
        model_parallelism=2,
        pipeline_parallelism=1,
    )

    launch_subprocesses(func, config, setup=setup)


if __name__ == "__main__":
    # python -m tests.utils.test_parallel
    test_parallel_model()
