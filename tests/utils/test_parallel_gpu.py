import logging

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ml.models.parallel import ColumnParallelLinear, RowParallelLinear
from ml.trainers.mixins.data_parallel import ModelConfig, fsdp
from ml.utils.logging import configure_logging
from ml.utils.networking import get_unused_port
from ml.utils.torch_distributed import MultiprocessConfig, launch_subprocesses

logger = logging.getLogger(__name__)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Gets the model and pipeline parallel modules.
        self.layer_1 = ColumnParallelLinear(12, 16, bias=False)
        self.layer_2 = RowParallelLinear(16, 8, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_2(self.layer_1(x))

    def forward_non_parallel(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.layer_1.master_weight)
        x = F.linear(x, self.layer_2.master_weight)
        return x


def setup() -> None:
    # Hides some nuisance logs.
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.ERROR)
    logging.getLogger("ml.utils.torch_distributed").setLevel(logging.ERROR)

    # Setting the seed across all processes to make sure that the weights
    # initialize to the same values (needed to make the test pass).
    torch.manual_seed(1337)


def func() -> None:
    config = ModelConfig(use_fsdp=True)
    base_model = DummyModel()
    base_model.cuda()

    # Keeps a copy of the full weights on CPU for later comparison. This needs
    # to happen after moving the weights to CUDA to avoid NCCL errors, but
    # before wrapping the model in FSDP to avoid memory access errors.
    cpu_w1 = base_model.layer_1.master_weight.cpu()
    cpu_w2 = base_model.layer_2.master_weight.cpu()

    model = fsdp(base_model, config)
    model.eval()

    x = torch.randn(4, 12, device="cuda")

    # Tests that the gradients are non-null. Using gradient clipping as a
    # proxy since FSDP gradients are not stored on the tensors themselves.
    output = model.forward(x)
    output.sum().backward()
    grad_norm = model.clip_grad_norm_(1e-3)
    assert grad_norm.abs().sum() > 0.0

    # Tests that the parallel forward pass matches the reference model. This is
    # done on CPU to avoid memory access errors when overlapping with FSDP.
    cpu_y_parallel = output.detach().cpu()
    model.zero_grad()
    cpu_y_full = F.linear(F.linear(x.cpu(), cpu_w1), cpu_w2)
    assert torch.allclose(cpu_y_parallel, cpu_y_full, atol=1e-3)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_parallel_model() -> None:
    """Tests model parallelism primitives.

    This function launches 2 model parallel groups. We check that the
    partitioned model outputs and gradients match the full model, when using
    the FSDP wrapper.

    This test is also a good way to validate NCCL on your system.
    """

    configure_logging()

    port = get_unused_port()

    config = MultiprocessConfig(
        world_size=2,
        master_addr="127.0.0.1",
        master_port=port,
        init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        model_parallelism=2,
        pipeline_parallelism=1,
    )

    launch_subprocesses(func, config, setup=setup)


if __name__ == "__main__":
    # python -m tests.utils.test_parallel_gpu
    test_parallel_model()
