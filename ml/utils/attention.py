"""Defines some augmentations for attention functions."""

import torch
from torch import Tensor, nn


class NextTokenDiscriminator(nn.Module):
    """Defines a module for training a discriminator on next token predictions.

    Consider doing GAN-style training on an autoregressive transformer model.
    For an input sequence with shape ``(T, C)``, the generator model outputs
    a next token prediction for each timestep, giving a tensor with shape
    ``(T, C)``. The discriminator model then conditions on the ground truth
    tensors and the predicted next tokens to give a discriminator score. The
    trick is that for each of the predicted token distributions, the
    discriminator should only be able to see the previous tokens and not the
    input token.

    This module takes the input tensors described above, applies an "initial
    token" to the first one, and concatenates the two tensors to get the input
    tensor to use when training the generator model. It also returns the
    attention mask to use for training the model.

    This module can be used for other applications which define a conditional
    distribution over next token predictions, such as reinforcement learning.

    Parameters:
        emb_dim: The attention embedding dimension.
        max_tsz: The maximum number of input tokens.
    """

    def __init__(self, emb_dim: int, max_tsz: int) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.empty(1, 1, emb_dim))

        # Causal mask.
        causal_mask = ~torch.ones(max_tsz, max_tsz, dtype=torch.bool).tril(diagonal=0)
        self.register_buffer("causal_mask", causal_mask)

        # Current token mask.
        curr_mask = ~torch.eye(max_tsz, max_tsz, dtype=torch.bool)
        self.register_buffer("curr_mask", curr_mask)

        self.reset_parameters()

    causal_mask: Tensor
    curr_mask: Tensor

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.init_emb)

    def forward(self, prev_embs: Tensor, curr_embs: Tensor) -> tuple[Tensor, Tensor]:
        """Combines the embeddings to get the transformer inputs.

        Note that the tokens for the `prev_embs`` and ``curr_embs`` should be
        the same for each timestep; the only difference is that ``prev_embs``
        should be the ground truth tokens while ``curr_embs`` are the outputs
        of some upstream model, conditioned on ``prev_embs`` padded by one
        timestep.

        Args:
            prev_embs: The embeddings for the ``T - 1`` tokens, with shape
                ``(B, T, C)``
            curr_embs: The embeddings for the ``T`` tokens, with shape
                ``(B, T, C)``, which can only attend to the previous embeddings

        Returns:
            The inputs to the discriminator transformer, with shape
            ``(B, 2T, C``), and the attention mask, with shape ``(2T, 2T)``
        """
        bsz, tsz, _ = prev_embs.shape
        init_embs = self.init_emb.repeat(bsz, 1, 1)
        embs = torch.cat((init_embs, prev_embs[:, :-1], curr_embs), dim=1)
        causal_mask, curr_mask = self.causal_mask[:tsz, :tsz], self.curr_mask[:tsz, :tsz]
        mask = torch.cat(
            (
                torch.cat((causal_mask, torch.ones_like(curr_mask)), dim=1),
                torch.cat((causal_mask, curr_mask), dim=1),
            ),
            dim=0,
        )
        return embs, mask
