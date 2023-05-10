"""Defines a simpl API for using Meta's pretrained Hubert model.

.. highlight:: python
.. code-block:: python

    from ml.models.pretrained.hubert import pretrained_hubert

    model = pretrained_hubert("base")
    predictor = model.predictor()

    # Gets HuBERT embeddings for a waveform.
    predictor.predict(torch.randn(1, 16_000), output_layer=None)

    # Gets HuBERT embeddings for a long waveform, in batches.
    predictor.predict_in_chunks(torch.randn(1, 160_000), 16_000, output_layer=None)

The choices for the model key are:

- ``"base"`` - 12 layers, 768 hidden size, 12 attention heads.
- ``"large"`` - 24 layers, 1024 hidden size, 16 attention heads.
- ``"extra_large"`` - 48 layers, 1280 hidden size, 16 attention heads.
"""

import argparse
from dataclasses import dataclass
from typing import Literal, cast, get_args

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.datasets.utils import download_url

from ml.core.env import get_model_dir
from ml.models.activations import ActivationType, get_activation
from ml.utils.data import check_sha256
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.logging import configure_logging

HubertSize = Literal["base", "large", "extra_large"]


@dataclass
class HubertConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: ActivationType
    hidden_dropout: float
    activation_dropout: float
    attention_dropout: float
    feat_proj_layer_norm: bool
    feat_proj_dropout: float
    layer_norm_eps: float
    feat_extract_norm: str
    feat_extract_dropout: float
    feat_extract_activation: ActivationType
    conv_dim: tuple[int, ...]
    conv_stride: tuple[int, ...]
    conv_kernel: tuple[int, ...]
    conv_bias: bool
    num_conv_pos_embeddings: int
    num_conv_pos_embedding_groups: int
    do_stable_layer_norm: bool
    pre_normalize: bool

    @property
    def num_feat_extract_layers(self) -> int:
        return len(self.conv_dim)


def apply_mask(hidden_states: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
    # Make sure the padded tokens output 0.
    expand_attn_mask = attn_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
    hidden_states[~expand_attn_mask] = 0

    # Extends the attention mask.
    neg_inf = torch.finfo(hidden_states.dtype).min
    output_shape = attn_mask.shape[0], 1, attn_mask.shape[-1], attn_mask.shape[-1]
    attn_mask = ((1.0 - attn_mask[:, None, None, :].to(hidden_states)) * neg_inf).expand(output_shape)

    return hidden_states, attn_mask


class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings: int) -> None:
        super().__init__()

        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class HubertAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"`embed_dim` must be divisible by num_heads (got {self.embed_dim=} and {num_heads=}).")

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int) -> Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Runs the HuBERT attention layer.

        Args:
            hidden_states: Input states for the attention layer.
            key_value_states: If provided, will use this as the key and value
                states instead of deriving them from `hidden_states`.
            past_key_value: If provided, will use this as the past key and
                value states instead of deriving them from `hidden_states`.
            attn_mask: If provided, will mask the attention probabilities.
            layer_head_mask: If provided, will mask individual heads of the
                attention layer.

        Returns:
            A tuple consisting of the attention output and the
            updated past key and value states (if `past_key_value` is provided).

        Raises:
            ValueError: If `past_key_value` is not `None` and `key_value_states`
                is either `None` or has a different sequence length than the
                `past_key_value` tuple.
        """

        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if (
            key_value_states is not None
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]

        elif key_value_states is not None:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if (aw_size := attn_weights.size()) != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Weights should have size {(bsz * self.num_heads, tgt_len, src_len)}, but are {aw_size}")

        if attn_mask is not None:
            if (am_size := attn_mask.size()) != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Mask should have size {(bsz, 1, tgt_len, src_len)}, but is {am_size}")

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if (lhm_size := layer_head_mask.size()) != (self.num_heads,):
                raise ValueError(f"Head mask should be of size {(self.num_heads,)}, but is {lhm_size}")

            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if (ao_size := attn_output.size()) != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"Expected size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {ao_size}")

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value


class HubertFeedForward(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class HubertEncoderLayer(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        attn_residual = hidden_states
        hidden_states, _ = self.attention(hidden_states, attn_mask=attn_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class HubertEncoder(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
        output_layer: int | None = None,
    ) -> Tensor:
        if attn_mask is not None:
            # Make sure the padded tokens output 0.
            expand_attn_mask = attn_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attn_mask] = 0

            # Extends the attention mask.
            neg_inf = torch.finfo(hidden_states.dtype).min
            output_shape = attn_mask.shape[0], 1, attn_mask.shape[-1], attn_mask.shape[-1]
            attn_mask = ((1.0 - attn_mask[:, None, None, :].to(hidden_states)) * neg_inf).expand(output_shape)

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attn_mask=attn_mask)
            if output_layer is not None and i == output_layer:
                break

        return hidden_states


class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, layer_id: int = 0) -> None:
        super().__init__()

        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = get_activation(config.feat_extract_activation)

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, layer_id: int = 0) -> None:
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertLayerNormConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, layer_id: int = 0) -> None:
        super().__init__()

        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertFeatureEncoder(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        conv_layers: list[nn.Module] = []
        if config.feat_extract_norm == "group":
            conv_layers += [HubertGroupNormConvLayer(config, layer_id=0)]
            for i in range(config.num_feat_extract_layers - 1):
                conv_layers += [HubertNoLayerNormConvLayer(config, layer_id=i + 1)]
        elif config.feat_extract_norm == "layer":
            for i in range(config.num_feat_extract_layers):
                conv_layers += [HubertLayerNormConvLayer(config, layer_id=i)]
        else:
            raise ValueError(f"{config.feat_extract_norm=}, but has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)

    def _freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_values: Tensor) -> Tensor:
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class HubertFeatureProjection(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class HubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, _, _ = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        return hidden_states


class HubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        layers = [HubertEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
        output_layer: int | None = None,
    ) -> Tensor:
        if attn_mask is not None:
            hidden_states, attn_mask = apply_mask(hidden_states, attn_mask)

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attn_mask=attn_mask)
            if output_layer is not None and i == output_layer:
                break
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class Hubert(nn.Module):
    __constants__ = ["conv_kernel", "conv_stride"]

    def __init__(self, config: HubertConfig):
        super().__init__()

        self.conv_kernel = config.conv_kernel
        self.conv_stride = config.conv_stride
        self.pre_normalize = config.pre_normalize

        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)
        self.encoder = HubertEncoderStableLayerNorm(config) if config.do_stable_layer_norm else HubertEncoder(config)

    def _get_feat_extract_output_lengths(self, input_lengths: Tensor | int) -> Tensor:
        def _conv_out_length(input_length: int | Tensor, kernel_size: int, stride: int) -> Tensor:
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        lengths = input_lengths if isinstance(input_lengths, Tensor) else torch.tensor(input_lengths)
        for kernel_size, stride in zip(self.conv_kernel, self.conv_stride):
            lengths = _conv_out_length(lengths, kernel_size, stride)

        return lengths

    def _get_feature_vector_attn_mask(self, feature_vector_length: int, attn_mask: Tensor) -> Tensor:
        output_lengths = self._get_feat_extract_output_lengths(attn_mask.sum(-1)).to(torch.long)
        batch_size = attn_mask.shape[0]
        attn_mask = torch.zeros((batch_size, feature_vector_length), dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask[(torch.arange(attn_mask.shape[0], device=attn_mask.device), output_lengths - 1)] = 1
        attn_mask = attn_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attn_mask

    def forward(
        self,
        input_values: Tensor | None,
        attn_mask: Tensor | None = None,
        output_layer: int | None = None,
    ) -> Tensor:
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = self._get_feature_vector_attn_mask(extract_features.shape[1], attn_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self.encoder(hidden_states, attn_mask=attn_mask, output_layer=output_layer)
        return hidden_states

    def predictor(self, *, device: BaseDevice | None = None) -> "HubertPredictor":
        return HubertPredictor(self, device=device)


class HubertPredictor:
    def __init__(self, hubert_model: Hubert, *, device: BaseDevice | None = None) -> None:
        """Provides an API for doing predictoins with a HuBERT model.

        Note that this module is not an `nn.Module`, so you can use it in your
        module without worrying about storing all the weights on accident.

        Args:
            hubert_model: The HuBERT model to use for predictions.
            device: The device to use for predictions. If `None`, will use the
                device returned by AutoDevice.detect_device().
        """

        super().__init__()

        self.device = AutoDevice.detect_device() if device is None else device
        self.model = hubert_model.eval()
        self.device.module_to(self.model)

    def predict(self, waveform: np.ndarray | Tensor, output_layer: int | None = None) -> Tensor:
        """Gets the hidden states for the given waveform.

        Args:
            waveform: The waveform to get hidden states for, with shape (B, T)
            output_layer: The layer to get hidden states from. If `None`, will
                return the hidden states from the last layer.

        Returns:
            The hidden states for the given waveform, with shape (B, T, D)
        """

        waveform = self.device.tensor_to(waveform)
        return self.model.forward(waveform, attn_mask=None, output_layer=output_layer)

    def predict_in_chunks(
        self,
        waveform: Tensor | np.ndarray,
        chunk_size: int,
        output_layer: int | None = None,
    ) -> Tensor:
        """Gets the hidden states for the given waveform, in chunks.

        This is useful for processing very long waveforms, as it allows you to
        process the waveform in chunks, rather than loading the entire waveform
        into memory at once.

        Args:
            waveform: The waveform to get hidden states for, with shape (B, T)
            chunk_size: The size of each chunk to process.
            output_layer: The layer to get hidden states from. If `None`, will
                return the hidden states from the last layer.

        Returns:
            The hidden states for the given waveform, with shape (B, T, D)
        """

        with torch.inference_mode():
            x = self.device.tensor_to(waveform)  # Loads entire waveform into device memory.

            if self.model.pre_normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), chunk_size):
                x_chunk = x[:, start : start + chunk_size]
                feat_chunk, _ = self.model(x_chunk, output_layer=output_layer)
                feat.append(feat_chunk)

        return torch.cat(feat, 1).squeeze(0)


EXCLUDE_KEYS = {"masked_spec_embed", ".weight", ".bias"}


def _load_pretrained_hubert(
    size: HubertSize,
    ckpt_url: str,
    sha256: str,
    config: HubertConfig,
    remove_prefix: str | None = None,
) -> Hubert:
    model = Hubert(config)

    model_fname = f"{size}.bin"
    model_path = get_model_dir() / "hubert" / model_fname

    # Downloads the model if it doesn't exist
    if not model_path.is_file() or not check_sha256(model_path, sha256):
        model_path.parent.mkdir(exist_ok=True)
        download_url(ckpt_url, str(model_path.parent), model_fname)
        assert model_path.is_file(), f"Failed to download {model_path}"

    # Loads the model weights.
    ckpt = torch.load(model_path, map_location="cpu")
    if remove_prefix:
        ckpt = {k[len(remove_prefix) :]: v for k, v in ckpt.items()}
    ckpt = {k: v for k, v in ckpt.items() if k not in EXCLUDE_KEYS}
    model.load_state_dict(ckpt)

    return model


def hubert(size: HubertSize) -> Hubert:
    match size:
        case "base":
            return _load_pretrained_hubert(
                size,
                ckpt_url="https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin",
                sha256="062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad",
                config=HubertConfig(
                    vocab_size=32,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_act="gelu",
                    hidden_dropout=0.1,
                    activation_dropout=0.1,
                    attention_dropout=0.1,
                    feat_proj_layer_norm=True,
                    feat_proj_dropout=0.0,
                    layer_norm_eps=1e-5,
                    feat_extract_norm="group",
                    feat_extract_dropout=0.0,
                    feat_extract_activation="gelu",
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                    conv_stride=(5, 2, 2, 2, 2, 2, 2),
                    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                    num_conv_pos_embeddings=128,
                    num_conv_pos_embedding_groups=16,
                    conv_bias=False,
                    do_stable_layer_norm=False,
                    pre_normalize=False,
                ),
            )

        case "large":
            return _load_pretrained_hubert(
                size,
                ckpt_url="https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/main/pytorch_model.bin",
                sha256="9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0",
                remove_prefix="hubert.",
                config=HubertConfig(
                    vocab_size=32,
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    hidden_act="gelu",
                    hidden_dropout=0.1,
                    activation_dropout=0.1,
                    attention_dropout=0.1,
                    feat_proj_layer_norm=True,
                    feat_proj_dropout=0.0,
                    layer_norm_eps=1e-5,
                    feat_extract_norm="group",
                    feat_extract_dropout=0.0,
                    feat_extract_activation="gelu",
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                    conv_stride=(5, 2, 2, 2, 2, 2, 2),
                    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                    num_conv_pos_embeddings=128,
                    num_conv_pos_embedding_groups=16,
                    conv_bias=True,
                    do_stable_layer_norm=True,
                    pre_normalize=True,
                ),
            )

        case "extra_large":
            return _load_pretrained_hubert(
                size,
                ckpt_url="https://huggingface.co/facebook/hubert-xlarge-ll60k/resolve/main/pytorch_model.bin",
                sha256="6131dc27f4508595daa1a13fec4aa1f6b4a579b5d93550bae26c13a83221f8a7",
                config=HubertConfig(
                    vocab_size=32,
                    hidden_size=1280,
                    num_hidden_layers=48,
                    num_attention_heads=16,
                    intermediate_size=5120,
                    hidden_act="gelu",
                    hidden_dropout=0.1,
                    activation_dropout=0.1,
                    attention_dropout=0.1,
                    feat_proj_layer_norm=True,
                    feat_proj_dropout=0.0,
                    layer_norm_eps=1e-5,
                    feat_extract_norm="layer",
                    feat_extract_dropout=0.0,
                    feat_extract_activation="gelu",
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                    conv_stride=(5, 2, 2, 2, 2, 2, 2),
                    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                    num_conv_pos_embeddings=128,
                    num_conv_pos_embedding_groups=16,
                    conv_bias=True,
                    do_stable_layer_norm=True,
                    pre_normalize=True,
                ),
            )


def test_hubert_adhoc() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, choices=get_args(HubertSize))
    parser.add_argument("-t", "--tsz", type=int, default=22400)
    args = parser.parse_args()

    configure_logging()

    # Loads the model and moves to the right device.
    model = hubert(size=cast(HubertSize, args.size))
    predictor = model.predictor()

    # Test the model on a random waveform.
    y = predictor.predict(torch.randn(1, args.tsz))
    assert (args.tsz // 320) == y.shape[1] + 1


if __name__ == "__main__":
    # python -m ml.models.pretrained.hubert
    test_hubert_adhoc()
