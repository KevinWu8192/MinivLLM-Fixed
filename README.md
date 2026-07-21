<p align="center">
  <img src="./assets/minivllm.png" alt="图片描述" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# miniVLLM

First, many thanks to the original author for providing a minimal vLLM implementation that makes vLLM much easier to learn. However, the original MiniVLLM project is not yet complete: it contains a number of bugs and cannot reliably run an end-to-end tensor-parallel (TP) inference pipeline with correct scheduling and KV cache management.

This repository provides a fully runnable MiniVLLM inference framework. It fixes the bugs in the original project and adds new features on top of it. Bug fixes and new features are published as releases, each with a corresponding commit ID range, so learners can follow the releases step by step to fix the bugs, understand each new feature, or build their own features.

## Recent Releases

| Release | Commit range | Tag | Highlights | Documentation |
|---|---|---|---|---|
| Release 1 | `23d95ae–acdac94` | [`release-1`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-1) | Cross-request Prefix Cache block reuse and prefix-aware KV Cache allocation | [English](releases/release-1.md) · [简体中文](releases/release-1_zh.md) |
| Release 2 | `798b455` | [`release-2`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-2) | Context limits, KV Cache capacity, and Decode block-boundary allocation | [English](releases/release-2.md) · [简体中文](releases/release-2_zh.md) |
| Release 3 | `686b547` | [`release-3`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-3) | Prefix Cache-aware Prefill Attention over the complete KV context | [English](releases/release-3.md) · [简体中文](releases/release-3_zh.md) |
| Release 4 | `4d1760d` | [`release-4`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-4) | Qwen3 tensor-parallel checkpoint loading and strict validation | [English](releases/release-4.md) · [简体中文](releases/release-4_zh.md) |
| Release 5 | `ee950af` | [`release-5`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-5) | Multi-GPU coordination, scheduling fairness, numerical stability, and CUDA Graph correctness | [English](releases/release-5.md) · [简体中文](releases/release-5_zh.md) |
| Release 6 | `dcff99f–f71bd44` | [`release-6`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-6) | Split-KV GQA Decode Kernel with tiled MMA, stable reduction, and ~75× kernel-level speedup | [English](releases/release-6.md) · [简体中文](releases/release-6_zh.md) |
| Release 7 | `7d1b0f1` | [`release-7`](https://github.com/KevinWu8192/MinivLLM/tree/release-7) | Qwen3-32B configuration, model registration, and large-scale Attention integration | [English](releases/release-7.md) · [简体中文](releases/release-7_zh.md) |

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
uv run python benchmark_prefilling.py

# Run decoding benchmark
uv run python benchmark_decoding.py
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
uv run python benchmark_prefilling.py
```

This is the pefilling phase comparison

Compares three attention implementations during the **prefilling phase** (processing input prompts):

1. **PyTorch Standard (O(N²) memory)**: Traditional attention that materializes full attention matrix
2. **Naive Triton (O(N²) memory)**: GPU kernel that also uses O(N²) memory, limited by shared memory constraints (≤128 tokens)
3. **Flash Attention (O(N) memory)**: Memory-efficient online softmax algorithm that processes attention in blocks

```bash
uv run python benchmark_decoding.py
```

This is the decoding phase comparison

Compares three implementations during the **decoding phase** (generating output tokens one at a time):

1. **Naive PyTorch**: Simple loop-based implementation using paged KV cache
2. **Optimized PyTorch**: Vectorized implementation with batch gathering and masking
3. **Triton Kernel**: Custom GPU kernel optimized for paged attention decode


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
├── benchmark_prefilling.py   # Prefilling attention comparison
└── benchmark_decoding.py     # Decoding attention comparison
```

## Requirements

- Python ≥3.11, < 3.12
- CUDA-capable GPU
- Dependencies: `transformers`, `torch`, `xxhash` (managed by uv)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://www.star-history.com/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)
