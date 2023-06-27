"""Tests the loss functions API."""

import torch

from ml.tasks.losses.loss import loss_fn


def test_loss_fns() -> None:
    a, b = torch.rand(2, 3, 4), torch.rand(2, 3, 4)
    c = torch.randint(0, 2, (2, 4), dtype=torch.int64)
    d = torch.randn(2, 3, 4)

    mse_fn = loss_fn("mse")
    assert mse_fn(a, b).shape == (2, 3, 4)

    l1_fn = loss_fn("l1")
    assert l1_fn(a, b).shape == (2, 3, 4)

    huber_fn = loss_fn("huber", huber_beta=0.5)
    assert huber_fn(a, b).shape == (2, 3, 4)

    log_cosh_fn = loss_fn("log_cosh")
    assert log_cosh_fn(a, b).shape == (2, 3, 4)

    xent_fn = loss_fn("xent")
    assert xent_fn(a, c).shape == (2, 4)

    bce_fn = loss_fn("bce")
    assert bce_fn(a, b).shape == (2, 3, 4)

    bce_logits_fn = loss_fn("bce-logits")
    assert bce_logits_fn(a, d).shape == (2, 3, 4)

    a_audio, b_audio = torch.randn(2, 2048), torch.randn(2, 2048)
    stft_fn = loss_fn("stft")
    e_out, f_out = stft_fn(a_audio, b_audio)
    assert e_out.shape == f_out.shape == (2, 18)

    multi_stft_fn = loss_fn("multi-stft")
    e_out, f_out = multi_stft_fn(a_audio, b_audio)
    assert e_out.shape == f_out.shape == (2,)

    a_image, b_image = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
    ssim_fn = loss_fn("ssim")
    assert ssim_fn(a_image, b_image).shape == (2, 3, 2, 3)

    image_grad_fn = loss_fn("image-grad")
    assert image_grad_fn(a_image).shape == (2, 3, 2, 3)
