# 摘要 — Commit ID 区间：[`7d1b0f1`](https://github.com/KevinWu8192/MinivLLM/commit/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf)

**Tag：** [`release-7`](https://github.com/KevinWu8192/MinivLLM/tree/release-7)

本次 Release 新增 Qwen3-32B 的端到端模型组装支持：提供独立且兼容 Hugging Face 的运行配置，将模型注册到现有张量并行 Runner，并把其 Attention 层接入 Release 6 引入的 Large-scale Decode Kernel，同时保留 Qwen3-0.6B 的默认执行路径。

## 主要功能

### 1. 独立的 Qwen3-32B 运行配置

**涉及文件**

* `main_qwen32.py`

**功能**

* 新增独立的 Qwen3-32B 入口，模型架构配置为 `hidden_size = 5120`、`intermediate_size = 25600`、64 层、64 个 Query Head、8 个 KV Head、`head_dim = 128`、BF16 权重、40,960 Token 的位置容量，以及不共享的输入/输出 Embedding。运行配置默认使用八路张量并行，并显式启用 Large-scale Attention。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L13-L41))
* 该入口加载官方 `Qwen/Qwen3-32B` Tokenizer、应用其 Chat Template、初始化现有 `LLMEngine`，并通过其他已支持模型共用的调度、KV Cache、CUDA Graph、采样和 Checkpoint 加载链路执行生成。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L44-L64))

### 2. Qwen3-32B 模型注册与 Large-scale Attention 路由

**涉及文件**

* `src/myvllm/engine/model_runner.py`
* `src/myvllm/models/qwen3.py`

**功能**

* `ModelRunner` 现在同时识别 `Qwen3-0.6B` 和 `Qwen3-32B`，使用共享的 `Qwen3ForCausalLM` 实现构造两种 Checkpoint，并默认对 32B 模型启用 `use_large_scale_attention`。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/engine/model_runner.py#L48-L72))
* Attention Backend 选择从 `Qwen3ForCausalLM` 经过 `Qwen3Model` 传递到每一个 `Qwen3DecoderLayer`，因此全部 64 个 Transformer Layer 都能获得一致的 Backend 配置，同时无需修改现有 Layer 实现。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L190-L292))
* `Qwen3Attention` 在收到启用配置时选择 `attention_large_scale.Attention`。模型 API 边界上的该选项仍默认为 false，因此 Qwen3-0.6B 与现有调用方继续使用原 Attention 路径。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L30-L93))

最终模型组装路径如下：

```text
main_qwen32.py
    │ Qwen/Qwen3-32B 配置
    ▼
ModelRunner
    │ Qwen3-32B 注册 + TP Rank 本地模型构造
    ▼
Qwen3ForCausalLM → Qwen3Model → 64 × Qwen3DecoderLayer
    │ use_large_scale_attention = True
    ▼
attention_large_scale.Attention
    ├── Flash Attention Prefill
    └── Split-KV GQA Decode
```

### 3. Attention Backend 组装测试

**涉及文件**

* `tests/test_loader.py`

**功能**

* 新增聚焦模型组装的测试：使用 `use_large_scale_attention=True` 构造一个小型 Qwen3 模型，并验证每个 Decoder Layer 都包含 Large-scale Attention 实现。该测试无需分配 32B Checkpoint，即可检查完整的配置传递链路。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/tests/test_loader.py#L122-L140))
