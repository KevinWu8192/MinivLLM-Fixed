# Summary — Commit ID Range: [`1e73f71`](https://github.com/KevinWu8192/MinivLLM/commit/1e73f71de402c9a59ec0fd17dc7895b16070dfd9) → [`10f61d4`](https://github.com/KevinWu8192/MinivLLM/commit/10f61d40a8705bae90123016d2c81ad70e44d0f8)

**Tag:** [`release-8`](https://github.com/KevinWu8192/MinivLLM/tree/release-8)

This release validates Qwen3-32B end to end on two RTX PRO 6000 Blackwell GPUs. It adds four reproducible stress workloads and committed raw results, then fixes the BF16 Tensor Parallel output gather, steady-state KV Cache sizing, failed-startup worker cleanup, and CUDA Graph replay bugs exposed by those workloads.

## Major Features

### 1. Qwen3-32B Stress Benchmark

**Files**

* `tests/benchmarks/benchmark_qwen3_32b_stress.py`
* `tests/benchmarks/high-concurrency.json`
* `tests/benchmarks/long-decode.json`
* `tests/benchmarks/prefix-cache.json`

**Features**

* Four independent workload presets cover Prefix Cache reuse, high concurrency, sustained Decode, and the maximum-output boundary. Their defaults are respectively `8 × 16K + 256`, `16 × 4K + 256`, `2 × 8K + 4K`, and `1 × 8K + 32K`, where the values denote concurrent requests, Prompt length, and generated Tokens. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L64-L149))
* Deterministic long system instructions and two previous user/assistant turns create a realistic shared prefix. Cold requests place a unique marker in Block 0; the seed request materializes the common prefix; hot requests reuse it while retaining request-specific final turns. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L154-L248))
* The runner measures computed and cached Prompt Tokens, cache-hit rate, Prefill and Decode time, aggregate Decode Tokens per second, end-to-end output Tokens per second, and requests per second. `ignore_eos=True` makes every workload generate the exact requested Token count. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L260-L338))
* Each suite creates an engine shaped to its actual maximum Prompt, output length, concurrency, and batched-token budget, so a 32K maximum-output graph does not inflate the high-concurrency workload. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L400-L530))

#### Benchmark Results

The test system used two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 95.0 GiB each, TP=2, 50 Intel Xeon Platinum 8470Q vCPUs, 240 GiB host memory, BF16 weights, and CUDA Graphs enabled.

| Suite | Shape | Key result |
|---|---|---|
| High concurrency | 16 requests × 4,075 Prompt + 256 output | 5,954.27 Prefill tok/s, 435.13 aggregate Decode tok/s, 201.48 end-to-end output tok/s, 20.33 s total |
| Long decode | 2 requests × 8,169 Prompt + 4,096 output | 68.64 aggregate Decode tok/s, 66.57 end-to-end output tok/s, 123.05 s total |
| Maximum output | 1 request × 8,169 Prompt + 32,768 output | All 32,768 Tokens completed; 35.60 Decode tok/s; 920.45 s Decode and 923.03 s total |
| Prefix Cache | 8 requests × approximately 16K Prompt + 256 output | 98.69% hot-cache hit rate; Prefill fell from 31.902 s to 1.087 s; total time improved 3.91× |

The Prefix Cache reused 129,024 Tokens, or 16,128 Tokens per request: exactly 63 complete 256-token Blocks. The maximum-output run reached 40,937 total Tokens, only 23 below the configured 40,960 position capacity. Updated raw evidence is committed for [high concurrency](https://github.com/KevinWu8192/MinivLLM/blob/10f61d40a8705bae90123016d2c81ad70e44d0f8/tests/benchmarks/high-concurrency.json), [long Decode](https://github.com/KevinWu8192/MinivLLM/blob/10f61d40a8705bae90123016d2c81ad70e44d0f8/tests/benchmarks/long-decode.json), and [Prefix Cache](https://github.com/KevinWu8192/MinivLLM/blob/10f61d40a8705bae90123016d2c81ad70e44d0f8/tests/benchmarks/prefix-cache.json).

## Major Fixes

### 1. Preserve BF16 in Tensor-Parallel Logit Gather

**Files**

* `src/myvllm/layers/embedding_head.py`

**Bugs**

* Rank 0 allocated `gather_list` with `torch.empty(..., device=...)`, which used the current default FP32 Dtype instead of the BF16 Dtype produced by Qwen3-32B's local LM Head. `dist.gather()` rejects mixed Dtypes before concatenation. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/1e73f71de402c9a59ec0fd17dc7895b16070dfd9/src/myvllm/layers/embedding_head.py#L78-L90))

**Example**

With TP=2, both Ranks produce BF16 local Logits. Rank 0 creates two FP32 receive tensors, so PyTorch raises `ValueError: Invalid usage of tensors with different dtypes Found torch.bfloat16 and torch.float32` during the warmup gather.

**Fixes**

* Rank 0 now creates every receive tensor with `torch.empty_like(logits)`, inheriting Shape, Device, layout, and BF16 Dtype from the local output before gathering and concatenating the vocabulary shards. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/layers/embedding_head.py#L78-L93))

### 2. Measure Steady-State Memory Before Allocating KV Cache

**Files**

* `src/myvllm/engine/model_runner.py`

**Bugs**

* KV Cache capacity used the peak from the first and only warmup pass. That pass includes one-time Triton and `torch.compile` allocations; because the Sampler runs only on rank 0, rank 0 could record a much larger peak than rank 1 and calculate fewer than one available KV Block despite having enough steady-state memory. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/engine/model_runner.py#L215-L249))
* Warmup derived `batch_size` as `max_num_batched_tokens // max_model_length`, so the `65,200 / 4,331` high-concurrency shape warmed only 15 requests even though runtime admitted 16. Peak-memory sizing therefore did not cover the configured concurrency boundary. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/engine/model_runner.py#L215-L249))

**Example**

On the 16-request Qwen3-32B workload, rank 1 reported 1,581 local KV Blocks while rank 0 failed `num_available_kv_blocks >= 1`. The model weights fit on both 95 GiB GPUs; the asymmetry came from rank 0's compilation peak rather than persistent inference memory.

**Fixes**

* Warmup distributes the batched-token budget across at most `max_num_sequences`, including any remainder, so the exact maximum request count and Token budget are represented. The first synchronized pass compiles kernels; after clearing cached allocations and resetting peak statistics, a second synchronized pass measures steady-state execution. KV capacity is based only on that second peak. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/ee91fb66a2a065003c5bb8986f623f1c1da4fff6/src/myvllm/engine/model_runner.py#L219-L281))

### 3. Reap Tensor-Parallel Workers After Failed Initialization

**Files**

* `src/myvllm/engine/llm_engine.py`

**Bugs**

* Engine initialization was a straight-line sequence with no failure cleanup. If rank 0 raised after rank 1 entered NCCL, the parent traceback returned while rank 1 kept polling the dead TCPStore, repeatedly printing `Broken pipe`; `Ctrl+C` could stop the parent without reaping the spawned worker. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/engine/llm_engine.py#L17-L105))

**Example**

The BF16 gather or KV sizing exception occurs on rank 0 after both processes initialize NCCL. Rank 1 remains inside the ProcessGroup heartbeat because no `exit` message arrives and its process is never joined, leaving GPU memory allocated and the terminal flooded with TCPStore errors.

**Fixes**

* Worker entry now destroys its process group in `finally`, covering constructor errors and parent loss. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/ee91fb66a2a065003c5bb8986f623f1c1da4fff6/src/myvllm/engine/llm_engine.py#L17-L32))
* The parent initializes cleanup state before spawning, catches `BaseException`, requests graceful exit when possible, sends SIGTERM, joins with a five-second bound, escalates stuck workers to `kill()`, and destroys the local process group. Normal `exit()` uses the same idempotent bounded cleanup. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/ee91fb66a2a065003c5bb8986f623f1c1da4fff6/src/myvllm/engine/llm_engine.py#L45-L180))

### 4. Select the Smallest CUDA Graph and Neutralize Padding Rows

**Files**

* `src/myvllm/engine/model_runner.py`
* `tests/test_engine_boundaries.py`

**Bugs**

* CUDA Graphs are inserted in descending capture order, but Decode selected the first captured size greater than or equal to the live Batch size. A three-request Decode therefore selected the 16-row graph instead of the four-row graph, wasting work and exposing more padding state. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/9a43595c7fc3fba1e234bb7fe1160c2e56e808fe/src/myvllm/engine/model_runner.py#L435-L451))
* Replay refreshed live rows but did not clear all rows in the selected captured Batch. Stale `slot_mapping` and Block Table entries in padded rows could make inactive tokens write into real KV Cache slots. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/9a43595c7fc3fba1e234bb7fe1160c2e56e808fe/src/myvllm/engine/model_runner.py#L435-L451))

**Example**

For captured sizes `{16, 8, 4, 2, 1}` and runtime `bs = 3`, the old dictionary iteration chooses 16. Only rows 0–2 receive new request data; rows 3–15 can retain slot IDs and physical Block mappings from an earlier replay, so the graph executes 13 false requests against stale cache locations.

**Fixes**

* Decode explicitly takes `min(graph_bs for graph_bs in self.graphs if graph_bs >= bs)`. Before replay it zeroes padded input and context lengths, fills padded slot mappings and Block Tables with `-1`, and then copies only live request rows. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/6972cb5c3853e5e0099ad7947688c704391fb3af/src/myvllm/engine/model_runner.py#L435-L465))
* A boundary test uses `bs = 3`, verifies selection of the four-row graph, checks every inert sentinel value, and confirms that only three output rows are returned. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/6972cb5c3853e5e0099ad7947688c704391fb3af/tests/test_engine_boundaries.py#L100-L147))
