"""Defines a general-purpose API for losses.

.. highlight:: python
.. code-block:: python

    from ml.tasks.losses.loss import loss_fn

    mse_loss = loss_fn("mse")
    loss = mse_loss(pred, target)
    assert loss.shape == pred.shape

The loss functions are defined as follows:

- ``"mse"``: Mean squared error loss
- ``"l1"``: L1 loss
- ``"huber"``: Huber loss, which is a smoothed version of the L1 loss
- ``"log_cosh"``: Log cosh loss, which is a smoothed version of the L1 loss
- ``"xent"``: Cross-entropy loss
- ``"bce"``: Binary cross-entropy loss
- ``"bce-logits"``: Binary cross-entropy loss with logits
- ``"stft"``: Short-time Fourier transform loss
- ``"multi-stft"``: Multi-resolution short-time Fourier transform loss
"""

from typing import Callable, Literal, overload

import torch
from torch import Tensor, nn

from ml.tasks.losses.audio import (
    MultiResolutionSTFTLoss,
    STFTLoss,
    WindowFn,
    log_stft_magnitude_loss,
    spectral_convergence_loss,
)
from ml.tasks.losses.image import ImageGradLoss, SsimFn, SSIMLoss

Loss = Literal[
    "mse",
    "l1",
    "huber",
    "log_cosh",
    "xent",
    "bce",
    "bce-logits",
    "spectral-convergence",
    "log-stft-magnitude",
    "stft",
    "multi-stft",
    "ssim",
    "image-grad",
]

# Defines type aliases for the different loss functions.
OneInOneOutLoss = Callable[[Tensor], Tensor]
TwoInOneOutLoss = Callable[[Tensor, Tensor], Tensor]
TwoInTwoOutLoss = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
LossFn = OneInOneOutLoss | TwoInOneOutLoss | TwoInTwoOutLoss


def log_cosh_loss(pred: Tensor, target: Tensor) -> Tensor:
    loss = pred - target
    return torch.log(torch.cosh(loss))


@overload
def loss_fn(loss: Literal["mse"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["l1"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["huber"], *, huber_beta: float = 1.0) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["log_cosh"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["xent"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["bce"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["bce-logits"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["spectral-convergence"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(loss: Literal["log-stft-magnitude"]) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(
    loss: Literal["stft"],
    *,
    fft_size: int = 1024,
    shift_size: int = 120,
    win_length: int = 600,
    window_fn: WindowFn = "hann",
) -> TwoInTwoOutLoss:
    ...


@overload
def loss_fn(
    loss: Literal["multi-stft"],
    *,
    fft_size: int = 1024,
    shift_size: int = 120,
    win_length: int = 600,
    window_fn: WindowFn = "hann",
    fft_size_multiples: list[float] = [0.5, 1.0, 2.0],
) -> TwoInTwoOutLoss:
    ...


@overload
def loss_fn(
    loss: Literal["ssim"],
    *,
    image_kernel_size: int = 3,
    ssim_stride: int = 1,
    ssim_channels: int = 3,
    ssim_mode: SsimFn = "avg",
    image_sigma: float = 1.0,
    ssim_dynamic_range: float = 1.0,
) -> TwoInOneOutLoss:
    ...


@overload
def loss_fn(
    loss: Literal["image-grad"],
    *,
    image_kernel_size: int = 3,
    image_sigma: float = 1.0,
) -> OneInOneOutLoss:
    ...


@overload
def loss_fn(
    loss: Loss,
    *,
    huber_beta: float = 1.0,
    fft_size: int = 1024,
    shift_size: int = 120,
    win_length: int = 600,
    window_fn: WindowFn = "hann",
    fft_size_multiples: list[float] = [0.5, 1.0, 2.0],
    image_kernel_size: int = 3,
    ssim_stride: int = 1,
    ssim_channels: int = 3,
    ssim_mode: SsimFn = "avg",
    image_sigma: float = 1.0,
    ssim_dynamic_range: float = 1.0,
) -> LossFn:
    ...


def loss_fn(
    loss: Loss,
    *,
    huber_beta: float = 1.0,
    fft_size: int = 1024,
    shift_size: int = 120,
    win_length: int = 600,
    window_fn: WindowFn = "hann",
    fft_size_multiples: list[float] = [0.5, 1.0, 2.0],
    image_kernel_size: int = 3,
    ssim_stride: int = 1,
    ssim_channels: int = 3,
    ssim_mode: SsimFn = "avg",
    image_sigma: float = 1.0,
    ssim_dynamic_range: float = 1.0,
) -> LossFn:
    """Returns a loss function.

    Args:
        loss: The loss function to use.
        huber_beta: The beta parameter for the Huber loss.
        fft_size: The size of the FFT.
        shift_size: The size of the shift.
        win_length: The size of the window.
        window_fn: The window function to use.
        fft_size_multiples: The multiples of the FFT size to use for the
            multi-resolution STFT loss.
        image_kernel_size: The size of the kernel for the SSIM loss.
        ssim_stride: The stride of the kernel for the SSIM loss.
        ssim_channels: The number of channels for the SSIM loss.
        ssim_mode: The mode for the SSIM loss, either ``"avg"`` or ``"sum"``.
        image_sigma: The sigma parameter for the SSIM loss.
        ssim_dynamic_range: The dynamic range parameter for the SSIM loss.

    Returns:
        The loss function, as a callable that takes in the input tensor or
        tensors and returns the loss tensor or tensors.
    """
    match loss:
        case "mse":
            return nn.MSELoss(reduction="none")
        case "l1":
            return nn.L1Loss(reduction="none")
        case "huber":
            return nn.SmoothL1Loss(reduction="none", beta=huber_beta)
        case "log_cosh":
            return log_cosh_loss
        case "xent":
            return nn.CrossEntropyLoss(reduction="none")
        case "bce":
            return nn.BCELoss(reduction="none")
        case "bce-logits":
            return nn.BCEWithLogitsLoss(reduction="none")
        case "spectral-convergence":
            return spectral_convergence_loss
        case "log-stft-magnitude":
            return log_stft_magnitude_loss
        case "stft":
            return STFTLoss(
                fft_size=fft_size,
                shift_size=shift_size,
                win_length=win_length,
                window=window_fn,
            )
        case "multi-stft":
            return MultiResolutionSTFTLoss(
                fft_sizes=[int(fft_size * multiple) for multiple in fft_size_multiples],
                hop_sizes=[int(shift_size * multiple) for multiple in fft_size_multiples],
                win_lengths=[int(win_length * multiple) for multiple in fft_size_multiples],
                window=window_fn,
            )
        case "ssim":
            return SSIMLoss(
                kernel_size=image_kernel_size,
                stride=ssim_stride,
                channels=ssim_channels,
                mode=ssim_mode,
                sigma=image_sigma,
                dynamic_range=ssim_dynamic_range,
            )
        case "image-grad":
            return ImageGradLoss(
                kernel_size=image_kernel_size,
                sigma=image_sigma,
            )
        case _:
            raise NotImplementedError(f"Unexpected loss type: {loss}")
