"""Augmentation utilities for image data."""

import torch
from torch import Tensor


def get_image_mask(image: Tensor, *, min_prct: float = 0.2, max_prct: float = 0.5) -> Tensor:
    """Gets a random mask for the image.

    Args:
        image: The image to mask, with shape (..., H, W)
        min_prct: Minimum percent of image to mask in either dimension
        max_prct: Maximum percent of image to mask in either dimension

    Returns:
        An image mask, with shape (..., H, W) matching the original image shape
    """
    device, bsz, (height, width) = image.device, image.shape[:-2], image.shape[-2:]
    mask_dims = bsz + (1, 1)

    min_height, max_height = int(height * min_prct), int(height * max_prct)
    min_width, max_width = int(width * min_prct), int(width * max_prct)

    mask_height = torch.floor(torch.rand(*mask_dims, device=device) * (max_height - min_height) + min_height).long()
    mask_bottom = torch.floor(torch.rand(*mask_dims, device=device) * (height - mask_height)).long()
    mask_top = mask_bottom + mask_height

    mask_width = torch.floor(torch.rand(*mask_dims, device=device) * (max_width - min_width) + min_width)
    mask_left = torch.floor(torch.rand(*mask_dims, device=device) * (width - mask_height)).long()
    mask_right = mask_left + mask_width

    ys, xs = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    ys, xs = ys.view((1,) * len(bsz) + (height, width)), xs.view([1] * len(bsz) + [height, width])

    mask = (xs >= mask_left) & (xs <= mask_right) & (ys >= mask_bottom) & (ys <= mask_top)

    return mask
