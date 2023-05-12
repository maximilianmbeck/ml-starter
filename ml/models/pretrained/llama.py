# mypy: disable-error-code="import"
"""Defines a simple API for using Meta's pretrained LLaMa model.

This code is adapted from the original implementation
`here <https://github.com/facebookresearch/llama>`_, adapted to use
the parallelism primitives in this codebase.

.. highlight:: python
.. code-block:: python

    from ml.model.pretrained.llama import pretrained_llama

    model = pretrained_llama("7B")
    predictor = model.predictor()

    predictor.predict("The quick brown fox jumps over the lazy dog.")

Using the tokenizer requires installing the ``sentencepiece`` library:

.. code-block:: bash

    pip install sentencepiece

The choices for the model key are:

- ``"7B"``
- ``"13B"``
- ``"30B"``
- ``"65B"``
"""

import argparse
import functools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ml.core.env import get_model_dir
from ml.models.parallel import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.logging import configure_logging
from ml.utils.parallel import parallel_group_info
from ml.utils.timer import Timer
from ml.utils.torch_distributed import MultiprocessConfig, launch_subprocesses

logger = logging.getLogger(__name__)

PretrainedLlamaKey = Literal["7B", "13B", "30B", "65B"]


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    __constants__ = ["eps"]

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    cache_k: Tensor
    cache_v: Tensor

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // parallel_group_info().mp.world_size
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False)
        self.wk = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False)
        self.wv = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True)

        cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim))
        cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim))
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        freqs_cis: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ) -> None:
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        freqs_cis: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Tokenizer:
    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path).resolve()
        assert model_path.is_file(), f"Tokenizer model file not found at {model_path}"

        try:
            from sentencepiece import SentencePieceProcessor

        except ImportError:
            raise ImportError("Please install sentencepiece with: `pip install sentencepiece`")

        self.sp_model = SentencePieceProcessor(model_file=str(model_path))
        logger.info("Loaded sentence piece model from %s", model_path)

        # Gets the sentence statistics.
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        logger.info("Number of words: %d, BOS ID: %d, EOS ID: %d", self.n_words, self.bos_id, self.eos_id)

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        return self.sp_model.decode(t)


class Llama(nn.Module):
    def __init__(self, params: ModelArgs, tokenizer: Tokenizer | None = None) -> None:
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tokenizer = tokenizer

        self.tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    @torch.inference_mode()
    def forward(self, tokens: Tensor, start_pos: int) -> Tensor:
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])
        return output.float()

    def predictor(self) -> "LlamaPredictor":
        return LlamaPredictor(self)


class LlamaPredictor:
    def __init__(self, llama_model: Llama, *, device: BaseDevice | None = None) -> None:
        """Provides an API for sampling from the LLaMa model.

        Args:
            llama_model: The LLaMa model.
            device: The device to use for sampling. If None, the device will be
                automatically detected.

        Raises:
            ValueError: If the tokenizer is not set.
        """

        super().__init__()

        self.device = AutoDevice.detect_device() if device is None else device
        self.model = llama_model
        tokenizer = llama_model.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer must be set to use predictor")
        self.tokenizer = tokenizer
        self.device.module_to(self.model)

    def generate(
        self,
        prompts: list[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = self.device.tensor_to(torch.full((bsz, total_len), self.tokenizer.pad_id)).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs: Tensor, p: float) -> Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def pretrained_llama(key: PretrainedLlamaKey) -> Llama:
    rank, world_size = parallel_group_info().mp.rank, parallel_group_info().mp.world_size

    root_dir = get_model_dir() / "LLaMa"

    ckpt_dir = root_dir / key
    if not ckpt_dir.exists():
        raise ValueError(f"LLaMa model {key} not found at {ckpt_dir}; download it first")

    tokenizer_path = root_dir / "tokenizer.model"
    if not tokenizer_path.exists():
        raise ValueError(f"LLaMa tokenizer not found at {tokenizer_path}; download it first")

    # Loads the checkpoint for the current rank.
    with Timer("loading checkpoint", spinner=True):
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if world_size != (num_ckpts := len(checkpoints)):
            raise ValueError(f"Loading a checkpoint for {num_ckpts=} but {world_size=}")
        ckpt_path = checkpoints[rank]
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Loads the checkpoint parameters from the JSON file.
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args = ModelArgs(**params)

    # Builds the tokenizer and updates the vocab size.
    with Timer("loading tokenizer", spinner=True):
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

    # Builds the transformer.
    with Timer("building model", spinner=True):
        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.half)
        model = Llama(model_args, tokenizer)
        torch.set_default_dtype(prev_default_dtype)

    # Loads the checkpoint.
    with Timer("loading state dict", spinner=True):
        model.load_state_dict(checkpoint, strict=False)

    return model


def worker(
    key: PretrainedLlamaKey,
    prompts: list[str],
    max_gen_len: int,
    temperature: float,
    top_p: float,
) -> None:
    # Setting the seed across all processes to make sure that the weights
    # initialize to the same values (needed to make the test pass).
    torch.manual_seed(1337)

    model = pretrained_llama(key)
    predictor = model.predictor()

    generated = predictor.generate(prompts, max_gen_len, temperature=temperature, top_p=top_p)

    logger.info("Generated:\n%s", "\n ↪ ".join(generated))


def test_pretrained_model() -> None:
    parser = argparse.ArgumentParser(description="Tests a pretrained SAM model")
    parser.add_argument("key", type=str, choices=get_args(PretrainedLlamaKey))
    parser.add_argument("prompts", type=str, nargs="+")
    parser.add_argument("-m", "--max-gen-len", type=int, default=256)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("-p", "--top-p", type=float, default=0.95)
    args = parser.parse_args()

    configure_logging()

    # Gets the world size by counting the number of checkpoints.
    ckpt_dir = get_model_dir() / "LLaMa" / args.key
    if not ckpt_dir.exists():
        raise ValueError(f"LLaMa model {args.key} not found at {ckpt_dir}; download it first")
    world_size = len(list(Path(ckpt_dir).glob("*.pth")))

    launch_subprocesses(
        functools.partial(worker, args.key, args.prompts, args.max_gen_len, args.temperature, args.top_p),
        MultiprocessConfig(world_size=world_size),
    )


if __name__ == "__main__":
    # python -m ml.models.pretrained.llama
    test_pretrained_model()
