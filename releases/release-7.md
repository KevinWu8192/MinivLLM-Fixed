# Summary — Commit ID Range: [`7d1b0f1`](https://github.com/KevinWu8192/MinivLLM/commit/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf)

**Tag:** [`release-7`](https://github.com/KevinWu8192/MinivLLM/tree/release-7)

This release adds end-to-end model assembly support for Qwen3-32B. It provides a dedicated Hugging Face-compatible runtime configuration, registers the model with the existing tensor-parallel runner, and routes its Attention layers to the large-scale Decode Kernel introduced in Release 6 while preserving the default Qwen3-0.6B path.

## Major Features

### 1. Dedicated Qwen3-32B Runtime Configuration

**Files**

* `main_qwen32.py`

**Features**

* A standalone Qwen3-32B entry point defines the model architecture with `hidden_size = 5120`, `intermediate_size = 25600`, 64 layers, 64 Query Heads, 8 KV Heads, `head_dim = 128`, BF16 weights, a 40,960-token position capacity, and untied input/output embeddings. The runtime defaults to eight-way Tensor Parallelism and explicitly enables large-scale Attention. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L13-L41))
* The entry point loads the official `Qwen/Qwen3-32B` tokenizer, applies its chat template, initializes the existing `LLMEngine`, and runs generation through the same scheduling, KV Cache, CUDA Graph, sampling, and checkpoint-loading pipeline used by other supported models. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L44-L64))

### 2. Qwen3-32B Model Registration and Large-Scale Attention Routing

**Files**

* `src/myvllm/engine/model_runner.py`
* `src/myvllm/models/qwen3.py`

**Features**

* `ModelRunner` now recognizes both `Qwen3-0.6B` and `Qwen3-32B`. It builds either checkpoint with the shared `Qwen3ForCausalLM` implementation and defaults `use_large_scale_attention` to true for the 32B model. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/engine/model_runner.py#L48-L72))
* The Attention-backend choice is propagated from `Qwen3ForCausalLM` through `Qwen3Model` and every `Qwen3DecoderLayer`, so all 64 transformer layers receive the same backend selection without changing the existing layer implementations. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L190-L292))
* `Qwen3Attention` selects `attention_large_scale.Attention` when requested. The option remains false by default at the model API boundary, preserving the original Attention path for Qwen3-0.6B and existing callers. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L30-L93))

The resulting assembly path is:

```text
main_qwen32.py
    │ Qwen/Qwen3-32B configuration
    ▼
ModelRunner
    │ Qwen3-32B registration + TP-local model construction
    ▼
Qwen3ForCausalLM → Qwen3Model → 64 × Qwen3DecoderLayer
    │ use_large_scale_attention = True
    ▼
attention_large_scale.Attention
    ├── Flash Attention Prefill
    └── Split-KV GQA Decode
```

### 3. Attention Backend Assembly Coverage

**Files**

* `tests/test_loader.py`

**Features**

* A focused model-assembly test constructs a small Qwen3 model with `use_large_scale_attention=True` and verifies that every decoder layer contains the large-scale Attention implementation. This checks the complete configuration-propagation chain without allocating the 32B checkpoint. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/tests/test_loader.py#L122-L140))
