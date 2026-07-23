<p align="center">
  <img src="./assets/minivllm.png" alt="图片描述" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# MinivLLM-Fixed

First, many thanks to the original author for providing a minimal vLLM implementation that makes vLLM much easier to learn. However, the original MiniVLLM project is not yet complete: it contains a number of bugs and cannot reliably run an end-to-end tensor-parallel (TP) inference pipeline with correct scheduling and KV cache management.

This repository provides a fully runnable MiniVLLM inference framework. It fixes the bugs in the original project and adds new features on top of it. Bug fixes and new features are published as releases, each with a corresponding commit ID range, so learners can follow the releases step by step to fix the bugs, understand each new feature, or build their own features.

> **Inference starting point:** [Release 8](https://github.com/KevinWu8192/MinivLLM/tree/release-8) is the first fully end-to-end runnable release. Start with Release 8 or a newer version when running inference or benchmarks; earlier releases are incremental learning checkpoints and may not run the complete inference pipeline.

## Recent Releases

| Release | Commit range | Tag | Highlights | Documentation |
|---|---|---|---|---|
| Release 1 | `23d95ae–acdac94` | [`release-1`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-1) | Cross-request Prefix Cache block reuse and prefix-aware KV Cache allocation | [English](releases/release-1.md) · [简体中文](releases/release-1_zh.md) |
| Release 2 | `798b455` | [`release-2`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-2) | Context limits, KV Cache capacity, and Decode block-boundary allocation | [English](releases/release-2.md) · [简体中文](releases/release-2_zh.md) |
| Release 3 | `686b547` | [`release-3`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-3) | Prefix Cache-aware Prefill Attention over the complete KV context | [English](releases/release-3.md) · [简体中文](releases/release-3_zh.md) |
| Release 4 | `4d1760d` | [`release-4`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-4) | Qwen3 tensor-parallel checkpoint loading and strict validation | [English](releases/release-4.md) · [简体中文](releases/release-4_zh.md) |
| Release 5 | `ee950af` | [`release-5`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-5) | Multi-GPU coordination, scheduling fairness, numerical stability, and CUDA Graph correctness | [English](releases/release-5.md) · [简体中文](releases/release-5_zh.md) |
| Release 6 | `dcff99f–f71bd44` | [`release-6`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-6) | Split-KV GQA Decode Kernel with tiled MMA, stable reduction, and ~75× kernel-level speedup | [English](releases/release-6.md) · [简体中文](releases/release-6_zh.md) |
| Release 7 | `7d1b0f1–5608dc3` | [`release-7`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-7) | Qwen3-32B integration, large-scale Attention, and single-download TP startup | [English](releases/release-7.md) · [简体中文](releases/release-7_zh.md) |
| **Release 8** | `1e73f71–10f61d4` | **[`release-8`](https://github.com/KevinWu8192/MinivLLM/tree/release-8)** | **First fully end-to-end runnable release; recommended starting point for inference tests.** Includes Qwen3-32B stress benchmarks plus BF16, KV Cache, TP cleanup, and CUDA Graph fixes. | [English](releases/release-8.md) · [简体中文](releases/release-8_zh.md) |

A custom implementation of vLLM inference engine with attention mechanism benchmarks, based on Nano-vLLM but with self-contained paged attention and flash attention implementation. 

Benchmarking on flash attention in prefilling time and paged attention in decoding time are provided.

**New to vLLM?** Check out [HowToApproachvLLM.md](HowToApproachvLLM.md) for a step-by-step implementation guide covering layers, models, paged attention, CUDA graphs, and scheduling.

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run the main inference engine
uv run python main.py

# Run prefilling benchmark
uv run python tests/benchmarks/benchmark_attention_prefill.py

# Run decoding benchmark
uv run python tests/benchmarks/benchmark_attention_decode.py --mode benchmark
```

To run multi-GPU setting, simply change world_size to n > 1 in config in main.py

## What Each Script Does

```bash
uv run python main.py
```

This is the main inference engine demo

Demonstrates the complete LLM inference pipeline using a custom engine implementation:
- Create a small version of Qwen3 with random initialization
- Creates 60 chat prompts (2 base prompts repeated 30 times each)
- Processes them through the custom LLM engine with batch processing
- Uses paged attention and KV cache management for efficient inference
- Generates up to 256 tokens per prompt with temperature sampling

This showcases how the custom vLLM implementation handles batched text generation with memory-efficient attention.

```bash
uv run python tests/benchmarks/benchmark_attention_prefill.py
```

This is the pefilling phase comparison

Compares three attention implementations during the **prefilling phase** (processing input prompts):

1. **PyTorch Standard (O(N²) memory)**: Traditional attention that materializes full attention matrix
2. **Naive Triton (O(N²) memory)**: GPU kernel that also uses O(N²) memory, limited by shared memory constraints (≤128 tokens)
3. **Flash Attention (O(N) memory)**: Memory-efficient online softmax algorithm that processes attention in blocks

```bash
uv run python tests/benchmarks/benchmark_attention_decode.py --mode benchmark
```

This is the decoding phase comparison

Compares the production implementations during the **decoding phase** (generating output tokens one at a time):

1. **Original Paged Attention Decode**: The original single-stage Triton implementation
2. **Large-scale Split-KV Decode**: The optimized two-stage GQA implementation used by Qwen3-32B

The driver also provides correctness and CUDA profiling modes. See the complete [test and benchmark catalog](tests/README.md).


## Project Structure

```
myvllm/
├── src/
│   └── myvllm/           # Core vLLM implementation
│       ├── models/       # Model implementations
│       ├── engine/       # LLM engine logic, including sequence definition for input prompts, block management for KV cache management for GPU, scheduler for iteration-based scheduling of sequences, runner for actual implementation of running prefilling and decoding, and engine for generation API interface
│       ├── layers/       # Components for model/
│       ├── utils/        # context
│       └── sampling_parameters.py
├── main.py              # Full inference demo
└── tests/
    ├── test_*.py             # Unit and regression tests
    ├── benchmarks/           # CUDA and end-to-end benchmarks
    └── results/              # Committed benchmark results
```

## Requirements

- Python ≥3.11, < 3.12
- CUDA-capable GPU
- Dependencies: `transformers`, `torch`, `xxhash` (managed by uv)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://www.star-history.com/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)
