"""Wrapper around the Lion optimizer.

This optimizer was proposed in `Symbolic Discovery of Optimization Algorithms
<https://arxiv.org/abs/2302.06675>`_.

Lion stands for "Evolved Sign Momentum". It is more memory-efficient than
Adam since it only keeps track of momentum.

In the original paper, the authors suggest using a larger batch size and a
smaller learning rate compared to Adam.

This optimizer shines for tasks like contrasitve learning and diffusion which
optimize proxy objectives rather than doing something like cross-entropy
classification, although in the paper the authors show that it performs
comparably to Adam on language modeling.

This implementation is based on the ``lucidrain's`` implementation
`here <https://github.com/lucidrains/lion-pytorch/>`_ and on the pseudo-code
from the paper, which is reproduced below:

.. highlight:: python
.. code-block:: python

    def train(weight, gradient, momentum, lr):
        update = interp(gradient, momentum, beta1)
        update = sign(update)
        momentum = interp(gradient, momentum, beta2)
        update = update + weight * weight_deacy
        update = update * lr
        return update, momentum
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
from ml.optimizers.types import OptFloat, OptLossClosure, Params

logger = logging.getLogger(__name__)


def _update_fn_vanilla(
    p: nn.Parameter,
    grad: Tensor,
    exp_avg: Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
) -> None:
    """Runs the update function for a given parameter.

    This can be made slightly faster using Triton, if GPU acceleration is
    available. Make sure Triton is installed and set ``use_triton=True`` in
    the optimizer configuration.

    Args:
        p: Parameter to update.
        grad: Gradient for the parameter.
        exp_avg: Exponential average of the gradient.
        lr: Learning rate.
        wd: Weight decay.
        beta1: First momentum coefficient.
        beta2: Second momentum coefficient.
    """
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.data.mul_(1 - lr * wd).add_(update, alpha=-lr)
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


def get_update_fn(use_triton: bool) -> Callable[[nn.Parameter, Tensor, Tensor, float, float, float, float], None]:
    if torch.cuda.is_available() and use_triton:
        try:
            from ml.utils.triton.lion import update_fn as triton_update_fn

            return triton_update_fn

        except ImportError:
            logger.warning("Triton is not installed. Using vanilla Lion update function.")

    return _update_fn_vanilla


class Lion(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not all([0.0 <= beta <= 1.0 for beta in betas]):
            raise ValueError(f"Invalid beta: {betas}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

        self.update_fn = get_update_fn(use_triton)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                lr: float = group["lr"]
                wd: float = group["weight_decay"]
                beta1, beta2 = group["betas"]
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                self.update_fn(p, grad, state["exp_avg"], lr, wd, beta1, beta2)

        return loss


@dataclass
class LionOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-4, help="Learning rate.")
    betas: tuple[float, float] = conf_field((0.9, 0.99), help="Beta coefficients.")
    weight_decay: float = conf_field(1e-2, help="Weight decay.")
    use_triton: bool = conf_field(True, help="Whether to use Triton for faster updates.")

    @classmethod
    def get_defaults(cls) -> dict[str, "LionOptimizerConfig"]:
        return {
            "lion-stable": LionOptimizerConfig(
                betas=(0.95, 0.98),
            ),
        }


@register_optimizer("lion", LionOptimizerConfig)
class LionOptimizer(BaseOptimizer[LionOptimizerConfig, Lion]):
    def get(self, model: nn.Module) -> Lion:
        return Lion(
            model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            use_triton=self.config.use_triton,
        )
