"""Defines primitive model parallel layers.

Before using this module, you should initialize the parallel process groups
using :func:`ml.utils.parallel.init_parallelism`. This will create
three process group for model parallelism, pipeline parallelism, and data
parallelism. The process group information can be accessed using
:func:`ml.utils.parallel.parallel_group_info`.

The following layers are defined:

- :class:`ParallelEmbedding`: A model-parallel embedding layer.
- :class:`ColumnParallelLinear`: A column model-parallel linear layer.
- :class:`RowParallelLinear`: A row model-parallel linear layer.

The :class:`RowParallelLinear` and :class:`ColumnParallelLinear` layers can
be used to create a model parallel two-layer MLP, as shown below.

.. code-block:: python

    # Create a parallel embedding layer.
    parallel_embedding = ParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=in_features,
    )

    # Create a column parallel linear layer.
    column_parallel_linear = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        gather_output=False,
    )

    # Create a row parallel linear layer.
    row_parallel_linear = RowParallelLinear(
        in_features=out_features,
        out_features=out_features,
        bias=bias,
        input_is_parallel=True,
    )

    # Applies the two linear layers together.
    x = torch.randint(0, vocab_size - 1, (bsz, tsz))
    y = row_parallel_linear(column_parallel_linear(parallel_embedding(x)))

This is equivalent to the following single-process implementation.

.. code-block:: python

    # Create a sequential model.
    model = nn.Sequential(
        nn.Embedding(vocab_size, in_features),
        nn.Linear(in_features, out_features, bias=bias),
        nn.Linear(out_features, out_features, bias=bias),
    )

    # Applies the sequential model.
    x = torch.randint(0, vocab_size - 1, (bsz, tsz))
    y = model(x)
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx
from torch.distributed.distributed_c10d import ReduceOp

from ml.models.init import InitializationType, init_
from ml.utils.parallel import parallel_group_info


class _ModelParallelCopy(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: FunctionCtx,
        x: Tensor,
        op: Any,  # noqa: ANN401
    ) -> Tensor:
        ctx.op = op
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return parallel_group_info().mp.reduce(grad, op=ctx.op), None


def mp_copy(x: Tensor, op: Any = ReduceOp.SUM) -> Tensor:  # noqa: ANN401
    """Copies the input to the model parallel region.

    Forward this is a no-op, but backward it reduces the gradient across
    model parallel replicas (i.e., it is a cross-replica sum).

    Args:
        x: Input tensor, with shape ``(*)``.
        op: Reduction operation to use when reducing the gradient.

    Returns:
        Output tensor, with shape ``(*)``.
    """
    return _ModelParallelCopy.apply(x, op)


class _ModelParallelReduce(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: FunctionCtx,
        x: Tensor,
        op: Any,  # noqa: ANN401
    ) -> Tensor:
        ctx.mark_dirty(x)
        return parallel_group_info().mp.reduce(x, op=op)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return grad, None


def mp_reduce(x: Tensor, op: Any = ReduceOp.SUM) -> Tensor:  # noqa: ANN401
    """Reduces the input from the model parallel region.

    Forward this reduces the input across model parallel replicas (i.e., it is
    a cross-replica sum), but backward it is a no-op.

    Args:
        x: Input tensor, with shape ``(*)``.
        op: Reduction operation to use when reducing the gradient.

    Returns:
        Output tensor, with shape ``(*)``.
    """
    return _ModelParallelReduce.apply(x, op)


class _ModelParallelScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:  # type: ignore[override]
        ctx.dim = dim
        return parallel_group_info().mp.split(x, dim=dim)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return parallel_group_info().mp.gather(grad, dim=ctx.dim), None


def mp_scatter(x: Tensor, dim: int = -1) -> Tensor:
    """Scatters the input across model parallel regions.

    Args:
        x: Input tensor, with shape ``(..., N, ...)``.
        dim: Dimension to scatter along.

    Returns:
        Output tensor, with shape ``(..., N // world_size, ...)``.
    """
    return _ModelParallelScatter.apply(x, dim)


class _ModelParallelGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:  # type: ignore[override]
        ctx.dim = dim
        return parallel_group_info().mp.gather(x, dim=dim)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return parallel_group_info().mp.split(grad, dim=ctx.dim), None


def mp_gather(x: Tensor, dim: int = -1) -> Tensor:
    """Gathers the input from model parallel regions.

    Args:
        x: Input tensor, with shape ``(..., N, ...)``.
        dim: Dimension to gather along.

    Returns:
        Output tensor, with shape ``(..., N * world_size, ...)``.
    """
    return _ModelParallelGather.apply(x, dim)


def initialize_model_parallel_affine_weight_(
    weight: Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_type: InitializationType = "xavier_normal",
    stride: int = 1,
) -> None:
    """Initializes an affine weight tensor for model-parallel training.

    Args:
        weight: Weight tensor to initialize.
        out_features: Number of output features.
        in_features: Number of input features.
        per_partition_size: Size of each partition.
        partition_dim: Partition dimension.
        init_type: Initialization type.
        stride: Stride for the initialization.
    """
    # Skip meta weights.
    if weight.is_meta:
        return

    mp_info = parallel_group_info().mp
    rank, world_size = mp_info.rank, mp_info.world_size

    # For single GPU cases, just initialize normally.
    if world_size == 1:
        init_(weight, None, init_type)
        return

    # Initializes the master weight.
    master_weight = weight.new_empty(out_features, in_features, requires_grad=False)
    init_(master_weight, None, init_type)

    # Splits the master weight by the world size.
    assert per_partition_size % stride == 0, f"{per_partition_size=} is not divisible by {stride=}"
    per_partition_per_stride_size = per_partition_size // stride
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)

    # Copies the rank weight to the model parallel weight.
    rank_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(rank_weight_list, dim=partition_dim, out=weight)


class ParallelEmbedding(nn.Module):
    __constants__ = ["num_embeddings", "embedding_dim", "padding_idx", "max_norm", "scale_grad_by_freq", "sparse"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_type: InitializationType = "xavier_normal",
    ) -> None:
        """Model-parallel embeddings.

        Embeddings are partitioned along the ``embedding_dim`` dimension.

        Args:
            num_embeddings: Number of embeddings (vocabulary size).
            embedding_dim: Embedding dimension; must be divisible by the
                model-parallel size.
            padding_idx: See ``nn.Embedding``.
            max_norm: See ``nn.Embedding``.
            norm_type: See ``nn.Embedding``.
            scale_grad_by_freq: See ``nn.Embedding``.
            sparse: See ``nn.Embedding``.
            init_type: Initialization type.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.init_type = init_type
        self._weight = None

        # Splits by world size.
        world_size = parallel_group_info().mp.world_size
        assert embedding_dim % world_size == 0, f"{embedding_dim=} not divisible by {world_size=}"
        self.embedding_dim_per_rank = embedding_dim // world_size

        # Allocate weights for current rank.
        self.weight = nn.Parameter(torch.empty(num_embeddings, self.embedding_dim_per_rank))

        self.reset_parameters()

    @property
    def master_weight(self) -> Tensor:
        return mp_gather(self.weight, dim=1)

    def reset_parameters(self) -> None:
        initialize_model_parallel_affine_weight_(
            weight=self.weight,
            out_features=self.num_embeddings,
            in_features=self.embedding_dim,
            per_partition_size=self.embedding_dim_per_rank,
            partition_dim=1,
            init_type=self.init_type,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = mp_copy(x)

        output_parallel = F.embedding(
            x,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return mp_gather(output_parallel)


class ColumnParallelLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "gather_output", "init_type", "stride"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_type: InitializationType = "xavier_normal",
        stride: int = 1,
    ) -> None:
        """A column parallel linear layer.

        This layer splits the weight matrix along the output feature dimension,
        and each rank is only responsible for ``out_features // world_size``
        number of output features.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.
            gather_output: Whether to gather the output from all the model
                parallel GPUs.
            init_type: Initialization type.
            stride: Stride for the initialization.
            lora_rank: The LoRA rank to use, if any.
        """
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.init_type = init_type
        self.stride = stride

        # Splits by world size.
        world_size = parallel_group_info().mp.world_size
        assert out_features % world_size == 0, f"{out_features=} not divisible by {world_size=}"
        self.output_size_per_partition = out_features // world_size

        # Initializes the per-rank weight.
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_model_parallel_affine_weight_(
            weight=self.weight,
            out_features=self.out_features,
            in_features=self.in_features,
            per_partition_size=self.output_size_per_partition,
            partition_dim=0,
            init_type=self.init_type,
            stride=self.stride,
        )

    @property
    def master_weight(self) -> Tensor:
        return mp_gather(self.weight, dim=0)

    @property
    def master_bias(self) -> Tensor | None:
        return None if self.bias is None else mp_gather(self.bias, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x: input tensor of size ``(*, in_features)``

        Returns:
            Output tensor of size ``(*, out_features // world_size)``, or
            ``(*, out_features)`` if ``gather_output`` is set to ``True``.
        """
        input_parallel = mp_copy(x)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return mp_gather(output_parallel) if self.gather_output else output_parallel


class RowParallelLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "input_is_parallel", "init_type", "stride"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_type: InitializationType = "xavier_normal",
        stride: int = 1,
    ) -> None:
        """A row parallel linear layer.

        This layer splits the weight matrix along the input feature dimension,
        and each rank is only responsible for ``in_features // world_size``
        number of input features.

        This can be paired with a column parallel layer to create a model
        parallel two-stage linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.
            input_is_parallel: Whether the input tensor is already split
                along the feature dimension.
            init_type: Initialization type.
            stride: Stride for the initialization.
        """
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.init_type = init_type
        self.stride = stride

        # Splits by world size.
        world_size = parallel_group_info().mp.world_size
        assert in_features % world_size == 0, f"{in_features=} not divisible by {world_size=}"
        self.input_size_per_partition = in_features // world_size

        # Initializes the per-rank weight.
        self.weight = nn.Parameter(Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_model_parallel_affine_weight_(
            weight=self.weight,
            out_features=self.out_features,
            in_features=self.in_features,
            per_partition_size=self.input_size_per_partition,
            partition_dim=-1,
            init_type=self.init_type,
            stride=self.stride,
        )

    @property
    def master_weight(self) -> Tensor:
        return mp_gather(self.weight, dim=-1)

    @property
    def master_bias(self) -> Tensor | None:
        return None if self.bias is None else mp_gather(self.bias, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x: input tensor of size ``(*, in_features)``, or
                ``(*, in_features // world_size)`` if ``input_is_parallel``
                is set to ``True``.

        Returns:
            Output tensor of size ``(*, out_features)``.
        """
        input_parallel = x if self.input_is_parallel else mp_scatter(x)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output = mp_reduce(output_parallel)
        return output if self.bias is None else output + self.bias
