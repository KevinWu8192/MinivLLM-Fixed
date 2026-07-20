# Summary — Commit ID Range: [`dcff99f`](https://github.com/KevinWu8192/MinivLLM/commit/dcff99f3b791a230eddc447b0519583bf6c209e5) → [`f71bd44`](https://github.com/KevinWu8192/MinivLLM/commit/f71bd443275e6f0b1ddfa3e482fa34e160ef92a9)

**Tag:** [`release-6`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-6)

This release prepares the decode-attention foundation for the next release. The next release will add single-node, multi-GPU deployment for Qwen3-32B and use it to validate Tensor Parallelism (TP) together with the complete inference framework's Prefill/Decode scheduling, KV Cache, CUDA Graph, and multi-rank execution paths. Release 6 first removes the large-model Decode Kernel bottleneck: in the isolated Qwen3-32B decode-attention benchmark, the new kernel delivers approximately **75× kernel-level speedup** over the original implementation.

The optimized implementation is intentionally introduced as `attention_large_scale.py` instead of silently replacing the default `attention.py`. This keeps the comparison reproducible and provides a controlled kernel for the upcoming end-to-end Qwen3-32B integration.

## Major Fixes

### 1. Remove the Serial Per-Query-Head Decode Bottleneck

**Files**

* `src/myvllm/layers/attention.py`
* `src/myvllm/layers/attention_large_scale.py`

**Bugs**

* The original launch grid was `(batch_size, num_heads)`. With Qwen3-32B at Batch 1, it launched only 64 Triton programs, and every program serially traversed the complete KV context. Long-context Decode therefore exposed too little independent work to occupy the GPU. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/25ce870d2319b387133fd5680fa0fc95cd10b2a8/src/myvllm/layers/attention.py#L475-L528))
* Although Qwen3-32B uses GQA and eight Query Heads share one KV Head, the old program was assigned to one Query Head. The same K/V data was consequently loaded and processed again by eight separate programs instead of being shared by the Query group. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/25ce870d2319b387133fd5680fa0fc95cd10b2a8/src/myvllm/layers/attention.py#L363-L424))
* QK and PV were expressed as scalar loops over `BLOCK_N`: the kernel loaded one K vector, reduced `q * k`, inserted one score into `qk`, and then repeated a second scalar loop to extract each probability and accumulate one V vector. This prevented the two dominant matrix products from using tiled `tl.dot` operations and Tensor Core-friendly MMA shapes. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/25ce870d2319b387133fd5680fa0fc95cd10b2a8/src/myvllm/layers/attention.py#L397-L465))

**Example**

The benchmark models Qwen3-32B Decode with `batch_size = 1`, `context_len = 4096`, `num_heads = 64`, `num_kv_heads = 8`, `head_dim = 128`, `BLOCK_N = 32`, and FP16 inputs. The old grid contains 64 programs. Each program owns one Query Head and walks all 128 N tiles; programs belonging to the same KV Head independently reload the same K/V tiles eight times.

**Fixes**

* The stage-1 grid is changed to `(batch_size, num_kv_heads, num_splits)`. Each program now owns one `(batch, KV Head, context split)` tuple, so long contexts expose N-direction parallelism instead of remaining serial inside one program. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L344-L387))
* All Query Heads sharing a KV Head are loaded as one Q group. For Qwen3-32B the logical group is `[8, 128]`; it is padded to an MMA-compatible `[16, 128]` tile, with padded rows masked from loads and stores. One K tile is then reused by all eight real Query Heads. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L388-L408))
* Paged K/V addresses for all 32 Tokens in an N tile are translated together. QK becomes `tl.dot(q, k)` over `[16, 128] @ [128, 32]`, and PV becomes `tl.dot(p, v)` over `[16, 32] @ [32, 128]`. The existing online-softmax state `m_i`, `l_i`, and `acc` is preserved across tiles. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L410-L455))

### 2. Split-KV Parallelism and Numerically Stable Reduction

**Files**

* `src/myvllm/layers/attention_large_scale.py`

**Bugs**

* The original kernel had no context-split dimension, so increasing `context_len` increased the serial work inside each Query-Head program without increasing the launch grid. This is particularly inefficient for Batch-1 autoregressive Decode, where Batch parallelism is unavailable. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/25ce870d2319b387133fd5680fa0fc95cd10b2a8/src/myvllm/layers/attention.py#L382-L465))
* A split context cannot be combined by averaging independently normalized Attention outputs. Each split has a different maximum logit and exponential denominator; discarding these statistics would change the global Softmax result. The original single-program implementation had no partial-state representation or reduction path. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/25ce870d2319b387133fd5680fa0fc95cd10b2a8/src/myvllm/layers/attention.py#L429-L472))

**Example**

For the Qwen3-32B benchmark shape, `4096 / BLOCK_N = 128` N tiles and `_choose_decode_num_splits()` selects 16 splits. Stage 1 therefore launches `1 × 8 × 16 = 128` programs. Each program processes a contiguous range of complete 32-Token tiles and emits one `(m_s, l_s, acc_s)` state for each of its eight real Query Heads.

**Fixes**

* The context is divided by N tiles rather than arbitrary Token boundaries. `num_n_tiles`, `tiles_per_split`, `split_tile_start`, and `split_tile_end` assign every complete `BLOCK_N = 32` tile to exactly one split while the final tile masks Tokens beyond `context_len`. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L381-L425))
* Stage 1 stores unnormalized online-softmax states in workspaces shaped `[batch, query_head, split]` for `m_i/l_i` and `[batch, query_head, split, head_dim]` for `acc`. Keeping split adjacent allows each reduction program to read all states for one Query Head. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L457-L473))
* The reduction reconstructs the exact global online Softmax. It computes `m = max_s(m_s)`, rescales every split by `exp(m_s - m)`, obtains `l = Σ_s exp(m_s - m) l_s` and `acc = Σ_s exp(m_s - m) acc_s`, then writes `acc / l`. Empty splits have `l_s = 0` and contribute nothing. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L490-L552))
* `_choose_decode_num_splits()` caps N parallelism by the available tile count, 32 splits, and a target of 128 stage-1 programs. The decision uses launch-time Shapes rather than runtime `context_lens`, keeping the kernel grid stable for CUDA Graph capture. When `num_splits == 1`, stage 1 normalizes directly into `output`, avoiding partial workspaces and the reduction launch. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/dcff99f3b791a230eddc447b0519583bf6c209e5/src/myvllm/layers/attention_large_scale.py#L555-L695))

### 3. New Decode Kernel Execution Flow

```text
                         Decode Q [B, 64, 128]
                                  │
                    group Q Heads by shared KV Head
                                  │
                                  ▼
                logical Q_group [8, 128] per KV Head
                 padded Q tile [16, 128] for MMA
                                  │
                       split context into aligned
                           BLOCK_N = 32 tiles
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
           N split 0           N split 1          N split S-1
              │                   │                   │
       paged K/V lookup     paged K/V lookup     paged K/V lookup
              │                   │                   │
      tiled QKᵀ / Softmax  tiled QKᵀ / Softmax  tiled QKᵀ / Softmax
          / PV with tl.dot     / PV with tl.dot     / PV with tl.dot
              │                   │                   │
        m₀, l₀, acc₀        m₁, l₁, acc₁        mₛ, lₛ, accₛ
              └───────────────────┼───────────────────┘
                                  ▼
                    log-sum-exp state reduction
               m = max(mₛ), rescale lₛ and accₛ
                                  │
                                  ▼
                   final Attention output [B, 64, 128]
```

When `S = 1`, the reduction stage is skipped and stage 1 writes the normalized output directly.

### 4. Reproducible Correctness, Benchmark, NCU, and NSYS Analysis

**Files**

* `tests/test_attention_large_scale_decode.py`
* `releases/release-6-ncu.png`
* `releases/release-6-nsys.png`

**Fixes**

* A deterministic CUDA driver now creates non-contiguous physical Block Tables and compares the optimized kernel with a PyTorch reference for short and long contexts. It separately compares old and new kernels for the Qwen3-32B benchmark shape, performs CUDA Event timing after warmup, and provides NVTX ranges plus CUDA profiler control for NCU/NSYS capture. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/f71bd443275e6f0b1ddfa3e482fa34e160ef92a9/tests/test_attention_large_scale_decode.py#L1-L221))

Run the committed driver with:

```bash
uv run python tests/test_attention_large_scale_decode.py --mode correctness
uv run python tests/test_attention_large_scale_decode.py --mode benchmark
uv run python tests/test_attention_large_scale_decode.py --mode profile --iterations 1
```

The NCU comparison on an NVIDIA GeForce RTX 4070 Ti SUPER reports approximately 3.11 ms for the original kernel and approximately 0.04 ms for the new stage-1 kernel, with the reduction shown separately at a much smaller duration. This corresponds to approximately **75× kernel-level Decode speedup** for the profiled Qwen3-32B shape.

![NCU comparison of the original and split-KV Decode Kernels](release-6-ncu.png)

The NSYS timeline independently shows the old serial kernel occupying a long continuous range while the stage-1 and reduction launches of the new path complete in a much shorter range. NSYS range duration includes launch, synchronization, and profiler overhead, so it is supporting timeline evidence rather than the source of the 75× kernel-only figure.

![NSYS timeline of the original and split-KV Decode Kernels](release-6-nsys.png)

## Validation

The committed test driver contains PyTorch-reference checks for context lengths `[1, 17, 31]` and `[2049, 4093]` with tolerance `atol=2e-2, rtol=2e-2`, plus an old-versus-new comparison for the Qwen3-32B shape `batch=1, context=4096, Q/KV=64/8, head_dim=128, fp16`. The supplied NCU and NSYS captures document the approximately 75× isolated Decode Kernel improvement. This documentation update did not rerun the CUDA tests locally because the current environment has no CUDA-capable PyTorch/Triton runtime. Full Qwen3-32B single-node multi-GPU TP execution and end-to-end scheduler validation are intentionally reserved for the next release.
