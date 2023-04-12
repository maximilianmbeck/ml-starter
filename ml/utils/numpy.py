import numpy as np
import torch
from torch import Tensor


def as_cpu_tensor(value: np.ndarray | Tensor) -> Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value.detach().cpu()


def as_numpy_array(value: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()
