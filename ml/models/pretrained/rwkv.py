# mypy: disable-error-code="import"
"""Defines a simple API for using the RWKV model.

This code is adapted from the minimimal implementation
`here <https://johanwind.github.io/2023/03/23/rwkv_details.html>`_, adapted
to be fine-tunable.

.. highlight:: python
.. code-block:: python

    from ml.models.pretrained.rwkv import pretrained_rwkv

    model = pretrained_rwkv("7B")
    predictor = model.predictor()

    predictor.predict("The quick brown fox jumps over the lazy dog.")

Using the tokenizer requires installing the ``tokenizers`` library:

.. code-block:: bash

    pip install tokenizers

The choices for the model key are:

- ``"169m"``
- ``"430m"``
- ``"1.5b"``
- ``"3b"``
- ``"7b"``
- ``"14b"``
- ``raven``
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.datasets.utils import download_url

from ml.core.env import get_model_dir
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)

PretrainedRwkvKey = Literal["169m", "430m", "1.5b", "3b", "7b", "14b", "raven"]

AttentionState = tuple[Tensor, Tensor, Tensor]
FeedForwardState = Tensor
State = tuple[AttentionState, FeedForwardState]


@dataclass
class ModelArgs:
    url: str
    emb_dim: int
    num_layers: int


PRETRAINED_MODEL_SIZES: dict[PretrainedRwkvKey, ModelArgs] = {
    "430m": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth",
        emb_dim=1024,
        num_layers=24,
    ),
}

TOKENIZER_URL = "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json"


class Attention(nn.Module):
    init_x: Tensor
    init_num: Tensor
    init_den: Tensor

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.time_decay = nn.Parameter(torch.empty(emb_dim))
        self.time_first = nn.Parameter(torch.empty(emb_dim))

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, emb_dim))

        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.receptance = nn.Linear(emb_dim, emb_dim, bias=False)
        self.output = nn.Linear(emb_dim, emb_dim, bias=False)

        self.register_buffer("init_x", torch.zeros(emb_dim), persistent=False)
        self.register_buffer("init_num", torch.zeros(emb_dim), persistent=False)
        self.register_buffer("init_den", torch.zeros(emb_dim), persistent=False)

    def forward(self, x: Tensor, state: AttentionState) -> tuple[Tensor, AttentionState]:
        last_x, last_num, last_den = (self.init_x, self.init_num, self.init_den) if state is None else state

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))

        wkv = (last_num + torch.exp(self.time_first + k) * v) / (last_den + torch.exp(self.time_first + k))
        rwkv = torch.sigmoid(r) * wkv

        num = torch.exp(-torch.exp(self.time_decay)) * last_num + torch.exp(k) * v
        den = torch.exp(-torch.exp(self.time_decay)) * last_den + torch.exp(k)

        return self.output(rwkv), (x[..., -1, :], num[..., -1, :], den[..., -1, :])


class FeedForward(nn.Module):
    init_state: Tensor

    def __init__(self, emb_dim: int, ffn_dim: int) -> None:
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, emb_dim))

        self.key = nn.Linear(emb_dim, ffn_dim, bias=False)
        self.receptance = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(ffn_dim, emb_dim, bias=False)

        self.register_buffer("init_state", torch.zeros(emb_dim), persistent=False)

    def forward(self, x: Tensor, state: FeedForwardState | None = None) -> tuple[Tensor, FeedForwardState]:
        last_x = self.init_state if state is None else state
        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        vk = self.value(F.relu(k) ** 2)

        return torch.sigmoid(r) * vk, x[..., -1, :]


class Block(nn.Module):
    def __init__(self, emb_dim: int, pre_norm: bool) -> None:
        super().__init__()

        self.ln0 = nn.LayerNorm(emb_dim) if pre_norm else None
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

        self.att = Attention(emb_dim)
        self.ffn = FeedForward(emb_dim, emb_dim * 4)

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        if self.ln0 is not None:
            x = self.ln0(x)
        dx, att_state_out = self.att(self.ln1(x), None if state is None else state[0])
        x = x + dx
        dx, ffn_state_out = self.ffn(self.ln2(x), None if state is None else state[1])
        x = x + dx
        return x, (att_state_out, ffn_state_out)


class Rwkv(nn.Module):
    def __init__(self, emb_dim: int, num_tokens: int, num_layers: int) -> None:
        super().__init__()

        self.emb = nn.Embedding(num_tokens, emb_dim)
        self.blocks = nn.ModuleList([Block(emb_dim, i == 0) for i in range(num_layers)])
        self.ln_out = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_tokens, bias=False)

    def forward(self, tokens: Tensor, states_in: list[State] | None = None) -> tuple[Tensor, list[State]]:
        x = self.emb(tokens)
        states_out: list[State] = []
        for i, block in enumerate(self.blocks):
            x, state_out = block(x, None if states_in is None else states_in[i])
            states_out.append(state_out)
        x = self.head(self.ln_out(x))
        e_x = torch.exp(x - torch.max(x))
        probs = e_x / e_x.sum()
        return probs, states_out

    def predictor(self) -> "RwkvPredictor":
        return RwkvPredictor(self)


def get_tokenizer() -> Any:
    tokenizer_path = get_model_dir() / "RWKV" / "tokenizer.json"

    try:
        from tokenizers import Tokenizer

    except ImportError:
        raise ImportError("Please install tokenizers with: `pip install tokenizers`")

    # Downloads the model if it doesn't exist
    if not tokenizer_path.is_file():
        tokenizer_path.parent.mkdir(exist_ok=True)
        download_url(TOKENIZER_URL, str(tokenizer_path.parent), tokenizer_path.name)
        assert tokenizer_path.is_file(), f"Failed to download {tokenizer_path}"

    return Tokenizer.from_file(str(tokenizer_path.resolve()))


class RwkvPredictor:
    def __init__(self, rwkv_model: Rwkv, *, device: BaseDevice | None = None) -> None:
        """Provides an API for sampling from the RWKV model.

        Args:
            rwkv_model: The RWKV model to use for sampling.
            device: The device to use for sampling. If None, the device will be
                automatically detected.
        """
        super().__init__()

        self.device = AutoDevice.detect_device() if device is None else device
        self.device.module_to(rwkv_model)
        self.tokenizer = get_tokenizer()
        self.model = rwkv_model

    def sample_probs(self, probs: Tensor, temperature: float = 1.0, top_p: float = 0.85) -> Tensor:
        probs = probs ** (1 / temperature)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort.squeeze(-3), num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token[..., None, :, :]).squeeze(-1)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_len: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.85,
        end_tok: int | None = None,
    ) -> Iterator[str]:
        tokens = self.tokenizer.encode(prompt).ids
        state: list[State] | None = None
        for token in tokens:
            probs, state = self.model(self.device.tensor_to(torch.tensor([token])), state)

        for i in range(max_len):
            token = self.sample_probs(probs, temperature=temperature, top_p=top_p)
            if token == end_tok:
                break
            yield self.tokenizer.decode([token.item()])
            if i < max_len - 1:
                probs, state = self.model(self.device.tensor_to(torch.tensor([token])), state)


def pretrained_rwkv(key: PretrainedRwkvKey, *, device: BaseDevice | None = None) -> Rwkv:
    device = AutoDevice.detect_device() if device is None else device
    ckpt_path = get_model_dir() / "RWKV" / f"{key}.pth"
    model_args = PRETRAINED_MODEL_SIZES[key]

    # Downloads the model if it doesn't exist
    if not ckpt_path.is_file():
        ckpt_path.parent.mkdir(exist_ok=True)
        download_url(model_args.url, str(ckpt_path.parent), ckpt_path.name)
        assert ckpt_path.is_file(), f"Failed to download {ckpt_path}"

    with Timer("loading model checkpoint", spinner=True):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    with Timer("building model skeleton", spinner=True), init_empty_weights():
        model = Rwkv(model_args.emb_dim, 50277, model_args.num_layers)

    # Logs model summary.
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s has %s parameters", key, f"{total_params:,}")

    # Build the transformer and loads the checkpoint.
    with Timer("loading state dict", spinner=True):
        model._apply(meta_to_empty_func(device.get_device(), torch.half))
        model.load_state_dict(ckpt, strict=False)

    return model


def test_rwkv_adhoc() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, choices=get_args(PretrainedRwkvKey))
    parser.add_argument("prompt", type=str)
    parser.add_argument("-t", "--tsz", type=int, default=256)
    parser.add_argument("-m", "--temperature", type=float, default=1.0)
    parser.add_argument("-p", "--top-p", type=float, default=0.85)
    args = parser.parse_args()

    configure_logging()

    model = pretrained_rwkv(args.size)
    predictor = model.predictor()

    print(args.prompt, end="")
    for token in predictor.generate(
        args.prompt,
        max_len=args.tsz,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    # python -m ml.models.pretrained.rwkv
    test_rwkv_adhoc()
