import argparse
import contextlib
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cuda
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def naive_self_attention(query: Tensor, key: Tensor, value: Tensor, is_causal: bool) -> Tensor:
    qlen, klen = query.shape[-2], key.shape[-2]
    attn_mask = query.new_ones(qlen, klen, dtype=torch.bool)
    attn_mask = attn_mask.tril(diagonal=0) if is_causal else attn_mask
    attn_mask_fl = query.new_zeros(qlen, klen).masked_fill(~attn_mask, -float("inf"))
    attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask_fl, dim=-1)
    return attn_weight @ value


def fast_self_attention(query: Tensor, key: Tensor, value: Tensor, is_causal: bool) -> Tensor:
    return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


@dataclass
class BenchmarkInput:
    bsz: int
    heads: int
    tsz: int
    emb: int
    is_causal: bool
    device: torch.device
    dtype: torch.dtype


@dataclass
class BenchmarkResult:
    forward_time: float
    backward_time: float
    memory: float


def run_benchmark(
    benchmark: BenchmarkInput,
    func: Callable[[Tensor, Tensor, Tensor, bool], Tensor],
    num_iters: int,
    run_backward: bool = True,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for _ in range(num_iters):
        qkv = torch.randn(
            benchmark.bsz,
            benchmark.heads,
            benchmark.tsz,
            benchmark.emb * 3,
            device=benchmark.device,
            dtype=benchmark.dtype,
        )
        if run_backward:
            qkv = qkv.requires_grad_()
        query, key, value = qkv.tensor_split(3, dim=-1)

        # Forward pass.
        torch.cuda.synchronize()
        start = time.time()
        output = func(query, key, value, benchmark.is_causal)
        torch.cuda.synchronize()
        forward_time = time.time() - start

        # Memory usage during forward pass.
        memory = torch.cuda.max_memory_allocated() / 1024**3

        # Backward pass.
        if run_backward:
            torch.cuda.synchronize()
            start = time.time()
            output.sum().backward()
            torch.cuda.synchronize()
            backward_time = time.time() - start
        else:
            backward_time = 0.0

        # Clean up.
        del qkv, query, key, value, output
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        results += [BenchmarkResult(forward_time, backward_time, memory)]

    return results


def run_all_benchmark(
    bsz: int,
    heads: int,
    tszs: list[int],
    embs: list[int],
    num_iters: int,
    run_backward: bool,
    no_cpu: bool,
    compile: bool,
    use_float32: bool,
    use_float16: bool,
    use_bfloat16: bool,
) -> pd.DataFrame:
    """Runs the benchmarking script for the two self-attention implementations.

    Args:
        bsz: Batch size to benchmark.
        heads: Number of heads to benchmark.
        tszs: List of sequence lengths to benchmark.
        embs: List of embedding sizes to benchmark.
        num_iters: Number of iterations to run for each benchmark.
        run_backward: Whether to run the backward pass.
        no_cpu: If True, does not run the CPU benchmark.
        compile: If True, uses `torch.compile`.
        use_float32: If True, use float32.
        use_float16: If True, use float16.
        use_bfloat16: If True, use bfloat16.

    Returns:
        A pandas DataFrame containing the benchmark results.
    """

    df = pd.DataFrame(
        columns=[
            "name",
            "bsz",
            "heads",
            "tsz",
            "emb",
            "is_causal",
            "device",
            "dtype",
            "forward_time",
            "backward_time",
            "memory",
        ]
    )

    def add_to_df(benchmark_input: BenchmarkInput, benchmark_results: list[BenchmarkResult], name: str) -> None:
        for benchmark_result in benchmark_results:
            df.loc[len(df)] = {
                "name": name,
                "bsz": benchmark_input.bsz,
                "heads": benchmark_input.heads,
                "tsz": benchmark_input.tsz,
                "emb": benchmark_input.emb,
                "is_causal": benchmark_input.is_causal,
                "device": benchmark_input.device.type,
                "dtype": benchmark_input.dtype,
                "forward_time": benchmark_result.forward_time,
                "backward_time": benchmark_result.backward_time,
                "memory": benchmark_result.memory,
            }

    devices = [] if no_cpu else [torch.device("cpu")]
    if torch.cuda.is_available():
        devices += [torch.device("cuda")]
    assert devices, "No devices available."

    # Gets all the benchmark inputs to evaluate.
    benchmark_inputs: list[BenchmarkInput] = []
    for tsz in tszs:
        for emb in embs:
            for is_causal in [True, False]:
                for device in devices:
                    dtypes: list[torch.dtype] = []
                    if use_float32:
                        dtypes += [torch.float32]
                    if use_float16:
                        dtypes += [torch.float16]
                    if use_bfloat16:
                        dtypes += [torch.bfloat16]
                    assert dtypes, "No dtypes available."
                    for dtype in dtypes:
                        benchmark_inputs.append(BenchmarkInput(bsz, heads, tsz, emb, is_causal, device, dtype))

    random.shuffle(benchmark_inputs)

    if compile:
        naive_self_attention_fn = torch.compile(naive_self_attention)
        fast_self_attention_fn = torch.compile(fast_self_attention)
    else:
        naive_self_attention_fn = naive_self_attention
        fast_self_attention_fn = fast_self_attention

    # Runs on each benchmark.
    for benchmark_input in tqdm(benchmark_inputs):
        naive_results = run_benchmark(benchmark_input, naive_self_attention_fn, num_iters, run_backward)
        fast_results = run_benchmark(benchmark_input, fast_self_attention_fn, num_iters, run_backward)
        add_to_df(benchmark_input, naive_results, "naive")
        add_to_df(benchmark_input, fast_results, "fast")

    if not run_backward:
        df = df.drop(columns=["backward_time"])

    return df


def make_graph(save_loc: Path, df: pd.DataFrame, name: str, run_backward: bool) -> None:
    if df.empty:
        return

    # Turns `forward_time`, `batchward_time` and `memory` colums into ratios
    # between `name` as "naive" verses "fast", then gets only the "fast" rows.
    df_fast = df[df["name"] == "fast"].copy().reset_index(drop=True)
    df_naive = df[df["name"] == "naive"].copy().reset_index(drop=True)
    for col in ["forward_time", "backward_time", "memory"] if run_backward else ["forward_time", "memory"]:
        df_fast[col] = df_fast[col] / df_naive[col]

    for is_causal in [True, False]:
        df = df_fast[df_fast["is_causal"] == is_causal].copy().reset_index(drop=True)

        # Plots `tsz` vs `forward_time`, striated by `emb`.
        sns.lineplot(
            data=df,
            x="tsz",
            y="forward_time",
            hue="emb",
            style="emb",
            markers=True,
            dashes=False,
            # errorbar=None,
        )
        plt.axhline(1.0, color="black", linestyle="--")
        plt.xlabel("Sequence Length")
        plt.ylabel("Fast / Naive, Forward Pass Time")
        plt.title(f"{name} forward{' (causal)' if is_causal else ''}")
        plt.savefig(save_loc / f"{name}_forward_time{'_causal' if is_causal else ''}.png")
        plt.clf()

        # Plots `tsz` vs `backward_time`, striated by `emb`.
        if run_backward:
            sns.lineplot(
                data=df,
                x="tsz",
                y="backward_time",
                hue="emb",
                style="emb",
                markers=True,
                dashes=False,
                # errorbar=None,
            )
            plt.axhline(1.0, color="black", linestyle="--")
            plt.xlabel("Sequence Length")
            plt.ylabel("Fast / Naive, Backward Pass Time")
            plt.title(f"{name} backward{' (causal)' if is_causal else ''}")
            plt.savefig(save_loc / f"{name}_backward_time{'_causal' if is_causal else ''}.png")
            plt.clf()

        # Plots `tsz` vs `memory`, striated by `emb`.
        sns.lineplot(
            data=df,
            x="tsz",
            y="memory",
            hue="emb",
            style="emb",
            markers=True,
            dashes=False,
            # errorbar=None,
        )
        plt.axhline(1.0, color="black", linestyle="--")
        plt.xlabel("Sequence Length")
        plt.ylabel("Fast / Naive, Memory Usage")
        plt.title(f"{name} memory{' (causal)' if is_causal else ''}")
        plt.savefig(save_loc / f"{name}_memory{'_causal' if is_causal else ''}.png")
        plt.clf()


def make_graphs(save_loc: Path, run_backward: bool) -> None:
    """Makes the graphs for the benchmark results.

    Args:
        save_loc: Path to the save directory
        run_backward: Whether to run the backward pass.
    """

    df = pd.read_csv(save_loc / "benchmark_self_attention.csv")
    make_graph(save_loc, df[(df["device"] == "cpu") & (df["dtype"] == "torch.float32")], "cpu_fp32", run_backward)
    make_graph(save_loc, df[(df["device"] == "cuda") & (df["dtype"] == "torch.float32")], "gpu_fp32", run_backward)
    make_graph(save_loc, df[(df["device"] == "cuda") & (df["dtype"] == "torch.float16")], "gpu_fp16", run_backward)
    make_graph(save_loc, df[(df["device"] == "cuda") & (df["dtype"] == "torch.bfloat16")], "gpu_bf16", run_backward)

    # Concatenates all PNGs to one PNG.
    image_paths = list(sorted(save_loc.glob("*.png")))
    n = 6 if run_backward else 4
    image_paths_chunked = [image_paths[i : i + n] for i in range(0, len(image_paths), n)]
    images = [[Image.open(p) for p in image_path_chunk] for image_path_chunk in image_paths_chunked]
    width_arr = np.array([[i.size[0] for i in ii] for ii in images])
    height_arr = np.array([[i.size[1] for i in ii] for ii in images])
    widths, heights = width_arr.max(1), height_arr.sum(1)
    total_width = widths.sum()
    max_height = heights.max()
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for image_chunk in images:
        y_offset = 0
        for im in image_chunk:
            new_im.paste(im, (x_offset, y_offset))
            y_offset += im.size[1]
        x_offset += im.size[0]
    new_im.save(save_loc / "benchmark_self_attention.png")


def main() -> None:
    """Script to run self-attention benchmarks.

    Usage:
        python -m scripts.benchmark_self_attention
    """

    class TqdmHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])
    handler = TqdmHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Surpress logs from torch._inductor and torch._dynamo
    logging.getLogger("torch._inductor").setLevel(logging.WARNING)
    logging.getLogger("torch._dynamo").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=32, help="Batch size to benchmark.")
    parser.add_argument("--heads", type=int, default=16, help="Number of heads to benchmark.")
    parser.add_argument("--tszs", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--embs", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--save-loc", type=str, default="out")
    parser.add_argument("--num-iters", type=int, default=25, help="Number of iterations to run for each benchmark.")
    parser.add_argument("--no-backward", action="store_true", help="Don't run backward pass.")
    parser.add_argument("--no-cpu", default=False, action="store_true", help="Don't run CPU benchmarks.")
    parser.add_argument("--compile", default=False, action="store_true", help="Compile the model before running.")
    parser.add_argument("--kernel", choices=["flash", "math", "mem_efficient"], default="flash", help="Kernel to use.")
    parser.add_argument("--matmul-precision", choices=["high", "medium"], default="high", help="Matmul precision.")
    parser.add_argument("--no-float32", default=False, action="store_true", help="Don't run float32 benchmarks.")
    parser.add_argument("--no-bfloat16", default=False, action="store_true", help="Don't run bfloat16 benchmarks.")
    parser.add_argument("--no-float16", default=False, action="store_true", help="Don't run float16 benchmarks.")
    args = parser.parse_args()

    save_loc = Path(args.save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)
    run_backward = not args.no_backward

    with contextlib.ExitStack() as cm:
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision(args.matmul_precision)

            kernel_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=args.kernel == "flash",
                enable_math=args.kernel == "math",
                enable_mem_efficient=args.kernel == "mem_efficient",
            )
            cm.enter_context(kernel_ctx)

        results_df = run_all_benchmark(
            args.bsz,
            args.heads,
            args.tszs,
            args.embs,
            args.num_iters,
            run_backward,
            args.no_cpu,
            args.compile,
            not args.no_float32,
            not args.no_bfloat16,
            not args.no_float16,
        )

    results_df.to_csv(save_loc / "benchmark_self_attention.csv", index=False)
    make_graphs(save_loc, run_backward)


if __name__ == "__main__":
    main()
