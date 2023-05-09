"""Defines common types used in the ML package.

This package makes heavy use of static typing to help with code readability and
maintainability. For example, when implementing a new task, you should override
the five generic types in the base task, as in the example below:

.. highlight:: python
.. code-block:: python

    Batch = Tensor
    Output = Tensor
    Loss = Tensor

    class SomeTask(BaseTask[SomeTaskConfig, SomeModel, Batch, Output, Loss]):
        def run_model(self, model: SomeModel, batch: Batch, state: State) -> Output:
            return model(batch)

        def compute_loss(self, model: SomeModel, batch: Batch, state: State, output: Output) -> Loss:
            return F.mse_loss(output, batch)

This will provide type hints for the task's methods, so that Mypy or whatever
other static analysis tool you use can verify that the types are correct.
"""

from typing import TypeVar

from torch import Tensor

Batch = TypeVar("Batch")
Output = TypeVar("Output")
Loss = TypeVar("Loss", bound=Tensor | dict[str, Tensor])

RLAction = TypeVar("RLAction")
RLState = TypeVar("RLState")
