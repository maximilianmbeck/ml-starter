"""Utility functions for configuring distributed parallel training.

Distributed training is broken up into three types of parallelism:

Model Parallelism
-----------------

Model parallelism partitions a single layer across multiple GPUs. During the
forward pass, within a layer, different GPUs perform different parts of the
computation, then communicate the results to each other.

Data Parallelism
----------------

Data parallelism splits the data across multiple GPUs. During the forward pass,
each GPU performs the same computation on different data, then communicates the
results to each other.

Pipeline Parallelism
--------------------

Pipeline parallelism splits the model across multiple GPUs. During the forward
pass, the output of one layer is computed on one GPU, then passed to the next
layer on another GPU.

Parallelism Example
-------------------

Consider doing distributed training on a model with 8 total GPUs. The model is
split length-wise into two parts, and each part is split width-wise into two
more parts. This gives a model parallelism of 4 and a data parallelism of 2.

The model parallel groups are then ``[[0, 1], [2, 3], [4, 5], [6, 7]]``. This
means that when GPUs 0 and 1 are finished computing their part of some layer,
they will communicate the results to each other. The same is true for the other
pairs of GPUs.

The pipeline parallel groups are ``[[0, 2], [1, 3], [4, 6], [5, 7]]``. This
means that when GPU 0 is finished computing its part of some layer and
syncing with GPU 1, it will communicate the output to GPU 2.

The data parallel groups are ``[[0, 4], [1, 5], [2, 6], [3, 7]]``. This means
that each minibatch will be split in half, with one half being sent to GPUS
``[0, 1, 2, 3]`` and the other half being sent to GPUs ``[4, 5, 6, 7]``.

So in summary, the resulting groups are:

- ``Model parallel groups``: ``[[0, 1], [2, 3], [4, 5], [6, 7]]``
- ``Data parallel groups``: ``[[0, 4], [1, 5], [2, 6], [3, 7]]``
- ``Pipeline parallel groups``: ``[[0, 2], [1, 3], [4, 6], [5, 7]]``
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal, overload

import torch
import torch.distributed
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import Backend, ReduceOp, Work, _get_default_group, is_initialized

logger = logging.getLogger(__name__)


@dataclass
class _GroupInfo:
    """Information and helper functions for a process group.

    This is a singleton which can be accessed via ``group_info()``. For example,
    to do a model parallel reduction, you can do:

    .. code-block:: python

        group_info().mp.reduce(tensor)

    Attributes:
        group: The process group.
        global_ranks: The global ranks of all processes in the group.
        rank: The rank of the current process in the group.
        world_size: The number of processes in the group.
    """

    group: ProcessGroup
    global_ranks: list[int]
    rank: int
    world_size: int

    @overload
    def reduce(
        self,
        tensor: Tensor,
        op: Any = ReduceOp.SUM,  # noqa: ANN401
        *,
        async_op: Literal[False] = False,
    ) -> Tensor:
        ...

    @overload
    def reduce(
        self,
        tensor: Tensor,
        op: Any = ReduceOp.SUM,  # noqa: ANN401
        *,
        async_op: Literal[True],
    ) -> Work:
        ...

    def reduce(
        self,
        tensor: Tensor,
        op: Any = ReduceOp.SUM,
        *,
        async_op: bool = False,
    ) -> Tensor | Work:  # noqa: ANN401
        """Reduces the tensor across all processes in the group.

        Consider two tensors in the same process group on different processes,
        with values ``[1, 2, 3]`` and ``[4, 5, 6]``. After calling this
        function, both tensors will have the value ``[5, 7, 9]``.

        Args:
            tensor: The tensor to reduce.
            op: The reduction operation to perform.
            async_op: Whether to perform the operation asynchronously.

        Returns:
            The reduced tensor.
        """
        if self.world_size == 1:
            return tensor
        work = torch.distributed.all_reduce(tensor, op=op, group=self.group, async_op=async_op)
        return work if async_op else tensor

    def split(self, tensor: Tensor, dim: int = 0) -> Tensor:
        """Splits the tensor across all processes in the group.

        Consider a tensor with shape ``[8, 4]`` split across 4 processes. After
        calling this function, each process will have a tensor with shape
        ``[2, 4]``.

        Args:
            tensor: The tensor to split.
            dim: The dimension to split along.

        Returns:
            The split tensor.
        """
        if self.world_size == 1:
            return tensor
        slice_len = tensor.shape[dim] // self.world_size
        return tensor.narrow(dim, self.rank * slice_len, slice_len)

    @overload
    def gather(self, tensor: Tensor, dim: int = -1, *, async_op: Literal[False] = False) -> Tensor:
        ...

    @overload
    def gather(self, tensor: Tensor, dim: int = -1, *, async_op: Literal[True]) -> Work:
        ...

    def gather(self, tensor: Tensor, dim: int = -1, *, async_op: bool = False) -> Tensor | Work:
        """Gathers the tensor across all processes in the group.

        Consider a tensor with shape ``[2, 4]`` split across 4 processes. After
        calling this function, the process with rank 0 will have a tensor with
        shape ``[8, 4]``.

        Args:
            tensor: The tensor to gather.
            dim: The dimension to gather along.
            async_op: Whether to perform the operation asynchronously.

        Returns:
            The gathered tensor, or a work pointer if async.
        """
        if self.world_size == 1:
            return tensor
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        work = torch.distributed.all_gather(output, tensor, group=self.group, async_op=async_op)
        return work if async_op else torch.cat(output, dim=dim)


@dataclass
class _GroupsInfos:
    mp: _GroupInfo
    pp: _GroupInfo
    dp: _GroupInfo


_parallel_group_info: _GroupsInfos | None = None
_default_group_info: _GroupInfo | None = None


def parallel_group_info() -> _GroupsInfos:
    assert _parallel_group_info is not None
    return _parallel_group_info


def default_group_info() -> _GroupInfo | None:
    global _default_group_info
    if _default_group_info is None and is_initialized():
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        _default_group_info = _GroupInfo(_get_default_group(), list(range(world_size)), rank, world_size)
    return _default_group_info


class ParallismError(Exception):
    pass


def init_parallelism(
    model_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    *,
    mp_backend: str | Backend | None = None,
    pp_backend: str | Backend | None = None,
    dp_backend: str | Backend | None = None,
) -> None:
    """Initializes parallelism groups and parameters.

    Args:
        model_parallelism: Number of model parallel GPUs. Each layer of
            computation will simultaneously run on this many GPUs.
        pipeline_parallelism: Number of pipeline parallel layers. The total
            number of GPUs processing a single input will be the product
            of ``model_parallelism`` and ``pipeline_parallelism``.
        mp_backend: Backend to use for model parallelism.
        pp_backend: Backend to use for pipeline parallelism.
        dp_backend: Backend to use for data parallelism.

    Raises:
        ParallismError: If some settings are invalid.
    """
    global _parallel_group_info

    if _parallel_group_info is not None:
        raise ParallismError("Parallelism is already initialized; call `reset_parallelism` first.")

    if not torch.distributed.is_initialized():
        raise ParallismError("Distributed training is not initialized.")

    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()

    # This is specific behavior - if model parallelism is too large for the
    # current machine, we just clamp it to whatever the world size is. We
    # don't do this for pipeline parallelism because there are fewer use cases
    # where it is necessary.
    if model_parallelism > world_size:
        logger.warning(
            "Model parallelism %d is greater than world size %d, setting to %d",
            model_parallelism,
            world_size,
            world_size,
        )
        model_parallelism = world_size

    # Validates parallelism for current world size.
    if world_size % model_parallelism != 0:
        raise ParallismError(f"{world_size=} is not divisible by {model_parallelism=}")
    if world_size % (model_parallelism * pipeline_parallelism) != 0:
        pipeline_size = model_parallelism * pipeline_parallelism
        raise ParallismError(f"{world_size=} is not divisible by {pipeline_size=}")

    data_parallelism = world_size // (model_parallelism * pipeline_parallelism)

    logger.info(
        "Initializing model parallelism %d, pipeline parallelism %d, and data parallelism %d",
        model_parallelism,
        pipeline_parallelism,
        data_parallelism,
    )

    # [[[0, 1],
    #   [2, 3]],
    #  [[4, 5],
    #   [6, 7]]]
    groups = torch.arange(world_size).view(data_parallelism, pipeline_parallelism, model_parallelism)

    # We split this way so that two near-by GPUs are more likely to be in the
    # same model parallel group than data parallel group. This is because for
    # typical environments we have data parallel groups that are on separate
    # devices.
    dp_rank = rank % (model_parallelism * pipeline_parallelism)
    pp_rank = (rank // pipeline_parallelism) % model_parallelism
    mp_rank = rank // (model_parallelism * pipeline_parallelism)

    def get_groups(groups: list[Tensor], backend: str | Backend | None) -> list[tuple[ProcessGroup, list[int]]]:
        return [(torch.distributed.new_group(group.tolist(), backend=backend), group.tolist()) for group in groups]

    # [[0, 4], [1, 5], [2, 6], [3, 7]].
    dp_groups = get_groups(groups.flatten(1).unbind(1), dp_backend)
    # [[0, 2], [1, 3], [4, 6], [5, 7]
    pp_groups = get_groups(groups.transpose(0, 1).flatten(1).unbind(1), pp_backend)
    # [[0, 1], [2, 3], [4, 5], [6, 7]]
    mp_groups = get_groups(groups.flatten(0, 1).unbind(0), mp_backend)

    # We need to initialize all groups across all devices, but then we choose
    # the specific group for this device.
    dp_group, dp_ids = dp_groups[dp_rank]
    pp_group, pp_ids = pp_groups[pp_rank]
    mp_group, mp_ids = mp_groups[mp_rank]

    # Sets the group info now that it is initialized.
    _parallel_group_info = _GroupsInfos(
        mp=_GroupInfo(mp_group, mp_ids, mp_rank, model_parallelism),
        pp=_GroupInfo(pp_group, pp_ids, pp_rank, pipeline_parallelism),
        dp=_GroupInfo(dp_group, dp_ids, dp_rank, data_parallelism),
    )


def parallelism_is_initialized() -> bool:
    return _parallel_group_info is not None


def reset_parallelism() -> None:
    global _parallel_group_info
    _parallel_group_info = None
