"""Function(s) for dealing with Numpy arrays."""

import numpy as np
import torch
from torch import Tensor


def as_cpu_tensor(value: np.ndarray | Tensor) -> Tensor:
    # numpy_to_tensor - This is here because I grep for this term sometimes.
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value.detach().cpu()


def as_numpy_array(value: np.ndarray | Tensor) -> np.ndarray:
    # to_numpy - This is here because I grep for this term sometimes.
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()
