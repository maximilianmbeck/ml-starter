"""Wrapper around the Shampoo optimizer.

This optimizer was proposed in `Shampoo: Preconditioned Stochastic Tensor
Optimization <https://arxiv.org/abs/1802.09568>`_.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
from ml.optimizers.common import separate_decayable_params
from ml.optimizers.types import OptFloat, OptLossClosure, Params


def _matrix_power(matrix: Tensor, power: float) -> Tensor:
    # Use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)


class Shampoo(Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.

    This is taken from the ``pytorch-optimizer`` package.

    .. highlight:: python
    .. code-block:: python

        import torch_optimizer as optim
        optimizer = optim.Shampoo(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss_fn(model(input), target).backward()
        optimizer.step()

    It has been proposed in ``Shampoo: Preconditioned Stochastic Tensor
    Optimization``.

    .. note::
        This is *not* an implementation of the later paper, ``Scalable Second
        Order Optimization for Deep Learning``, which is becoming more popular.

    Parameters:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)

    .. note::
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if update_freq < 1:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "epsilon": epsilon,
            "update_freq": update_freq,
        }

        super().__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The total loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss


@dataclass
class ShampooOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    momentum: float = conf_field(0.0, help="Momentum")
    weight_decay: float = conf_field(0.0, help="Weight decay")
    epsilon: float = conf_field(1e-4, help="Epsilon")
    update_freq: int = conf_field(1, help="Update frequency")
    default_decay: bool = conf_field(True, help="Whether to decay module params which aren't explicitly specified")


@register_optimizer("shampoo", ShampooOptimizerConfig)
class ShampooOptimizer(BaseOptimizer[ShampooOptimizerConfig, Shampoo]):
    def get(self, model: nn.Module) -> Shampoo:
        return Shampoo(
            separate_decayable_params(model, self.config.default_decay, self.config.weight_decay),
            lr=self.config.lr,
            momentum=self.config.momentum,
            epsilon=self.config.epsilon,
            update_freq=self.config.update_freq,
        )
