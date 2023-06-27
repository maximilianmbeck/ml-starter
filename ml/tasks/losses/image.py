"""Defines some loss functions which are suitable for images."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SsimFn = Literal["avg", "std"]


class SSIMLoss(nn.Module):
    """Computes structural similarity loss (SSIM).

    The `dynamic_range` is the difference between the maximum and minimum
    possible values for the image. This value is the actually the negative
    SSIM, so that minimizing it maximizes the SSIM score.

    Parameters:
        kernel_size: Size of the Gaussian kernel.
        stride: Stride of the Gaussian kernel.
        channels: Number of channels in the image.
        mode: Mode of the SSIM function, either ``avg`` or ``std``. The
            ``avg`` mode uses unweighted ``(K, K)`` regions, while the ``std``
            mode uses Gaussian weighted ``(K, K)`` regions, which allows for
            larger regions without worrying about blurring.
        sigma: Standard deviation of the Gaussian kernel.
        dynamic_range: Difference between the maximum and minimum possible
            values for the image.

    Inputs:
        x: float tensor with shape ``(B, C, H, W)``
        y: float tensor with shape ``(B, C, H, W)``

    Outputs:
        float tensor with shape ``(B, C, H - K + 1, W - K + 1)``
    """

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        channels: int = 3,
        mode: SsimFn = "avg",
        sigma: float = 1.0,
        dynamic_range: float = 1.0,
    ) -> None:
        super().__init__()

        self.c1 = (0.01 * dynamic_range) ** 2
        self.c2 = (0.03 * dynamic_range) ** 2

        match mode:
            case "avg":
                window = self.get_avg_window(kernel_size)
            case "std":
                window = self.get_gaussian_window(kernel_size, sigma)
            case _:
                raise NotImplementedError(f"Unexpected mode: {mode}")

        window = window.expand(channels, 1, kernel_size, kernel_size)
        self.window = nn.Parameter(window.clone(), requires_grad=False)
        self.stride = stride

    def get_gaussian_window(self, ksz: int, sigma: float) -> Tensor:
        x = torch.linspace(-(ksz // 2), ksz // 2, ksz)
        num = (-0.5 * (x / float(sigma)) ** 2).exp()
        denom = sigma * math.sqrt(2 * math.pi)
        window_1d = num / denom
        return window_1d[:, None] * window_1d[None, :]

    def get_avg_window(self, ksz: int) -> Tensor:
        return torch.full((ksz, ksz), 1 / (ksz**2))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.flatten(0, -4)
        y = y.flatten(0, -4)

        channels = x.size(1)
        mu_x = F.conv2d(x, self.window, groups=channels, stride=self.stride)
        mu_y = F.conv2d(y, self.window, groups=channels, stride=self.stride)
        mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

        sigma_x = F.conv2d(x**2, self.window, groups=channels, stride=self.stride) - mu_x_sq
        sigma_y = F.conv2d(y**2, self.window, groups=channels, stride=self.stride) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.window, groups=channels, stride=self.stride) - mu_xy

        num_a = 2 * mu_x * mu_y + self.c1
        num_b = 2 * sigma_xy + self.c2
        denom_a = mu_x_sq + mu_y_sq + self.c1
        denom_b = sigma_x**2 + sigma_y**2 + self.c2

        score = (num_a * num_b) / (denom_a * denom_b)
        return -score


class ImageGradLoss(nn.Module):
    """Computes image gradients, for smoothing.

    This function convolves the image with a special Gaussian kernel that
    contrasts the current pixel with the surrounding pixels, such that the
    output is zero if the current pixel is the same as the surrounding pixels,
    and is larger if the current pixel is different from the surrounding pixels.

    Parameters:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.

    Inputs:
        x: float tensor with shape ``(B, C, H, W)``

    Outputs:
        float tensor with shape ``(B, C, H - ksz + 1, W - ksz + 1)``
    """

    kernel: Tensor

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0) -> None:
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert kernel_size > 1, "Kernel size must be greater than 1"

        self.kernel_size = kernel_size
        self.register_buffer("kernel", self.get_kernel(kernel_size, sigma), persistent=False)

    def get_kernel(self, ksz: int, sigma: float) -> Tensor:
        x = torch.linspace(-(ksz // 2), ksz // 2, ksz)
        num = (-0.5 * (x / float(sigma)) ** 2).exp()
        denom = sigma * math.sqrt(2 * math.pi)
        window_1d = num / denom
        window = window_1d[:, None] * window_1d[None, :]
        window[ksz // 2, ksz // 2] = 0
        window = window / window.sum()
        window[ksz // 2, ksz // 2] = -1.0
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.size(1)
        return F.conv2d(x, self.kernel.repeat_interleave(channels, 0), stride=1, padding=0, groups=channels)
