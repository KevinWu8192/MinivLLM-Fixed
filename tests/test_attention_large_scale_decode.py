"""Correctness, timing, and profiler driver for paged decode attention.

Examples:
    uv run python tests/test_attention_large_scale_decode.py --mode correctness
    uv run python tests/test_attention_large_scale_decode.py --mode benchmark
    uv run python tests/test_attention_large_scale_decode.py --mode profile --iterations 1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch

from myvllm.layers.attention import paged_attention_decode as old_decode
from myvllm.layers.attention_large_scale import (
    _choose_decode_num_splits,
    paged_attention_decode as new_decode,
)


HEAD_DIM = 128
BLOCK_SIZE = 16
BLOCK_N = 32
SCALE = HEAD_DIM**-0.5


@dataclass
class Inputs:
    query: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    num_heads: int
    num_kv_heads: int

    def args(self) -> tuple:
        return (
            self.query,
            self.k_cache,
            self.v_cache,
            self.block_tables,
            self.context_lens,
            SCALE,
            self.num_heads,
            self.num_kv_heads,
            HEAD_DIM,
            BLOCK_SIZE,
        )


def make_inputs(
    context_lens_cpu: list[int],
    max_num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
) -> Inputs:
    batch_size = len(context_lens_cpu)
    required_blocks = sum(math.ceil(length / BLOCK_SIZE) for length in context_lens_cpu)
    num_physical_blocks = required_blocks + 11
    generator = torch.Generator(device="cuda").manual_seed(
        20260720 + max_num_blocks + num_heads
    )
    shape = (num_physical_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    query = torch.randn(
        batch_size,
        num_heads,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    k_cache = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    v_cache = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    block_tables = torch.zeros(
        batch_size, max_num_blocks, device="cuda", dtype=torch.int32
    )
    permutation = torch.randperm(
        num_physical_blocks, device="cuda", generator=generator
    ).to(torch.int32)
    cursor = 0
    for batch_idx, context_len in enumerate(context_lens_cpu):
        count = math.ceil(context_len / BLOCK_SIZE)
        block_tables[batch_idx, :count] = permutation[cursor : cursor + count]
        cursor += count
    return Inputs(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=torch.tensor(context_lens_cpu, device="cuda", dtype=torch.int32),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


def reference_decode(inputs: Inputs) -> torch.Tensor:
    outputs = []
    q_per_kv = inputs.num_heads // inputs.num_kv_heads
    for batch_idx, context_len_tensor in enumerate(inputs.context_lens.cpu()):
        context_len = int(context_len_tensor)
        positions = torch.arange(context_len, device="cuda")
        logical_blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        physical_blocks = inputs.block_tables[batch_idx, logical_blocks]
        block_offsets = positions % BLOCK_SIZE
        keys = inputs.k_cache[physical_blocks, block_offsets].float()
        values = inputs.v_cache[physical_blocks, block_offsets].float()
        keys = keys.repeat_interleave(q_per_kv, dim=1)
        values = values.repeat_interleave(q_per_kv, dim=1)
        scores = torch.einsum("hd,thd->ht", inputs.query[batch_idx].float(), keys)
        probabilities = torch.softmax(scores * SCALE, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, values))
    return torch.stack(outputs).to(inputs.query.dtype)


def error_text(actual: torch.Tensor, expected: torch.Tensor) -> str:
    error = (actual.float() - expected.float()).abs()
    return f"max_abs={error.max().item():.6g}, mean_abs={error.mean().item():.6g}"


def run_correctness() -> None:
    # Qwen3-0.6B: 2 query heads share one KV head.
    for name, contexts, max_blocks in (
        ("short", [1, 17, 31], 2),
        ("long", [2049, 4093], 256),
    ):
        inputs = make_inputs(contexts, max_blocks, num_heads=16, num_kv_heads=8)
        actual = new_decode(*inputs.args())
        expected = reference_decode(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        splits = _choose_decode_num_splits(
            len(contexts), inputs.num_kv_heads, max_blocks * BLOCK_SIZE, BLOCK_N
        )
        print(f"PASS {name}: contexts={contexts}, splits={splits}, {error_text(actual, expected)}")


def timed_ms(fn: Callable[..., torch.Tensor], inputs: Inputs, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn(*inputs.args())
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def warmup(inputs: Inputs, iterations: int) -> None:
    # Compile and warm both kernels outside profiler capture and timing regions.
    for _ in range(iterations):
        old_decode(*inputs.args())
        new_decode(*inputs.args())
    torch.cuda.synchronize()


def run_benchmark(warmup_iterations: int, iterations: int) -> None:
    # Qwen3-32B: 8 query heads share one KV head.
    inputs = make_inputs([4096], 256, num_heads=64, num_kv_heads=8)
    warmup(inputs, warmup_iterations)
    old_output = old_decode(*inputs.args())
    new_output = new_decode(*inputs.args())
    torch.cuda.synchronize()
    torch.testing.assert_close(new_output, old_output, atol=2e-2, rtol=2e-2)

    with torch.cuda.nvtx.range("old_kernel_benchmark"):
        old_ms = timed_ms(old_decode, inputs, iterations)
    with torch.cuda.nvtx.range("new_kernel_benchmark"):
        new_ms = timed_ms(new_decode, inputs, iterations)
    print(
        f"Qwen3-32B shape: batch=1, context=4096, Q/KV={inputs.num_heads}/"
        f"{inputs.num_kv_heads}, head_dim={HEAD_DIM}, fp16"
    )
    print(f"correctness old-vs-new: {error_text(new_output, old_output)}")
    print(f"old: {old_ms * 1000:.3f} us | new: {new_ms * 1000:.3f} us")
    print(f"speedup: {old_ms / new_ms:.3f}x | latency reduction: {(1-new_ms/old_ms)*100:.2f}%")


def run_profile(warmup_iterations: int, iterations: int) -> None:
    inputs = make_inputs([4096], 256, num_heads=64, num_kv_heads=8)
    warmup(inputs, warmup_iterations)
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range("old_kernel"):
        for _ in range(iterations):
            old_decode(*inputs.args())
        torch.cuda.synchronize()
    with torch.cuda.nvtx.range("new_kernel"):
        for _ in range(iterations):
            new_decode(*inputs.args())
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("correctness", "benchmark", "profile"), default="benchmark"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    print(
        f"GPU: {torch.cuda.get_device_name()} | torch={torch.__version__} | "
        f"CUDA={torch.version.cuda}"
    )
    if args.mode == "correctness":
        run_correctness()
    elif args.mode == "benchmark":
        run_benchmark(args.warmup, args.iterations)
    else:
        run_profile(args.warmup, args.iterations)


if __name__ == "__main__":
    main()
