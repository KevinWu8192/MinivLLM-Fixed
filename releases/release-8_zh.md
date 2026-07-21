# 摘要 — Commit ID 区间：[`1e73f71`](https://github.com/KevinWu8192/MinivLLM/commit/1e73f71de402c9a59ec0fd17dc7895b16070dfd9) → [`bfefde5`](https://github.com/KevinWu8192/MinivLLM/commit/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6)

**Tag：** [`release-8`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-8)

本次 Release 在两张 RTX PRO 6000 Blackwell GPU 上完成了 Qwen3-32B 端到端验证，新增四种可复现压力测试与原始结果，并修复测试暴露出的 BF16 张量并行输出 Gather、稳态 KV Cache 容量计算、启动失败 Worker 清理及 CUDA Graph Replay 问题。

## 主要功能

### 1. Qwen3-32B 压力测试

**涉及文件**

* `tests/benchmarks/benchmark_qwen3_32b_stress.py`
* `tests/results/qwen3_32b/README.md`
* `tests/results/qwen3_32b/stress_high_concurrency.json`
* `tests/results/qwen3_32b/stress_long_decode.json`
* `tests/results/qwen3_32b/stress_max_output.json`
* `tests/results/qwen3_32b/stress_prefix_cache.json`

**功能**

* 四个独立预设分别覆盖 Prefix Cache 复用、高并发、持续 Decode 与最大输出边界。默认 Shape 依次为 `8 × 16K + 256`、`16 × 4K + 256`、`2 × 8K + 4K` 与 `1 × 8K + 32K`，分别表示并发请求数、Prompt 长度和生成 Token 数。([实现](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L64-L149))
* 确定性的长 System 指令与两轮历史 user/assistant Turn 构成真实共享前缀。冷请求在 Block 0 写入唯一标记；Seed 请求物化公共前缀；热请求复用该前缀并保留请求特有的最后一轮。([实现](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L154-L248))
* Runner 统计实际计算与缓存的 Prompt Token、缓存命中率、Prefill 与 Decode 时间、聚合 Decode Tokens/s、端到端输出 Tokens/s 和 requests/s。`ignore_eos=True` 保证每个负载精确生成请求的 Token 数。([实现](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L260-L338))
* 每个 Suite 都按照实际最大 Prompt、输出长度、并发数和批 Token 预算创建独立 Engine，因此 32K 最大输出的 Graph 不会放大高并发负载。([实现](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/benchmark_qwen3_32b_stress.py#L400-L530))

### 2. 统一 Test、Benchmark 与结果目录

**涉及文件**

* `tests/README.md`
* `tests/benchmarks/README.md`
* `tests/benchmarks/benchmark_attention_decode.py`
* `tests/benchmarks/benchmark_attention_prefill.py`
* `tests/benchmarks/benchmark_engine_tps.py`
* `tests/benchmarks/benchmark_qwen3_32b_stress.py`
* `tests/results/qwen3_32b/`

**功能**

* 单元测试继续以 `tests/test_*.py` 供 Pytest 发现；性能 Driver 统一使用 `tests/benchmarks/benchmark_<scope>.py`；提交的测量结果统一使用 `tests/results/<model>/stress_<suite>.json`。目录说明记录了所有保留 Benchmark 的依赖与标准命令。([整理](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/benchmarks/README.md#L1-L56))
* 删除根目录中的旧 `benchmark_decoding.py`，因为它在 Driver 内复制了旧 Decode Kernel。`benchmark_attention_decode.py` 改为直接导入两个生产实现，避免 Benchmark 代码与 Engine 漂移。
* 硬件、负载 Shape、原始指标和派生对比与不可变 JSON 证据放在同一目录。([结果](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/results/qwen3_32b/README.md#L1-L51))

## 主要修复

### 1. Tensor Parallel Logit Gather 保持 BF16

**涉及文件**

* `src/myvllm/layers/embedding_head.py`

**问题**

* rank 0 使用 `torch.empty(..., device=...)` 分配 `gather_list`，采用当前默认 FP32 Dtype，而不是 Qwen3-32B 本地 LM Head 输出的 BF16 Dtype。`dist.gather()` 会在拼接前拒绝混合 Dtype。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/1e73f71de402c9a59ec0fd17dc7895b16070dfd9/src/myvllm/layers/embedding_head.py#L78-L90))

**示例**

TP=2 时两个 Rank 都生成 BF16 本地 Logit，但 rank 0 创建两个 FP32 接收 Tensor，因此 Warmup Gather 抛出 `ValueError: Invalid usage of tensors with different dtypes Found torch.bfloat16 and torch.float32`。

**修复**

* rank 0 现在通过 `torch.empty_like(logits)` 创建每个接收 Tensor，在 Gather 与词表分片拼接前继承本地输出的 Shape、Device、Layout 与 BF16 Dtype。([修复](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/layers/embedding_head.py#L78-L93))

### 2. 分配 KV Cache 前测量稳态显存

**涉及文件**

* `src/myvllm/engine/model_runner.py`

**问题**

* KV Cache 容量使用第一次也是唯一一次 Warmup 的峰值。该过程包含一次性 Triton 与 `torch.compile` 分配；Sampler 又只在 rank 0 执行，因此 rank 0 可能记录远高于 rank 1 的峰值，并在稳态显存足够时仍算出不足一个 KV Block。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/engine/model_runner.py#L215-L249))
* Warmup 通过 `max_num_batched_tokens // max_model_length` 计算 `batch_size`，因此高并发 Shape `65,200 / 4,331` 只预热 15 个请求，而运行时允许 16 个请求。峰值显存计算没有覆盖配置的并发边界。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/engine/model_runner.py#L215-L249))

**示例**

在 16 请求 Qwen3-32B 负载中，rank 1 报告 1,581 个本地 KV Block，rank 0 却在 `num_available_kv_blocks >= 1` 失败。模型权重能够装入两张 95 GiB GPU；不对称来自 rank 0 的编译峰值，而不是持久推理显存。

**修复**

* Warmup 把批 Token 预算分配给最多 `max_num_sequences` 个请求，并分配余数，从而覆盖精确最大请求数和 Token 预算。第一次同步执行用于编译 Kernel；清理缓存分配并重置峰值后，第二次同步执行测量稳态显存。KV 容量只使用第二次峰值。([修复](https://github.com/KevinWu8192/MinivLLM/blob/ee91fb66a2a065003c5bb8986f623f1c1da4fff6/src/myvllm/engine/model_runner.py#L219-L281))

### 3. 初始化失败后回收 Tensor Parallel Worker

**涉及文件**

* `src/myvllm/engine/llm_engine.py`

**问题**

* Engine 初始化是没有失败清理的直线流程。rank 1 进入 NCCL 后，如果 rank 0 抛出异常，父进程返回 Traceback，但 rank 1 会继续轮询已经死亡的 TCPStore 并重复打印 `Broken pipe`；`Ctrl+C` 可能只停止父进程而不回收已创建 Worker。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/44a74daf253fb9b4f98ee2d27aab4a2b77d96b0c/src/myvllm/engine/llm_engine.py#L17-L105))

**示例**

BF16 Gather 或 KV 容量异常发生在 rank 0，而两个进程已经初始化 NCCL。rank 1 因为收不到 `exit` 消息而停留在 ProcessGroup Heartbeat 内，既不释放 GPU 显存，也持续向终端输出 TCPStore 错误。

**修复**

* Worker 入口现在在 `finally` 中销毁 Process Group，覆盖构造失败与父进程丢失。([修复](https://github.com/KevinWu8192/MinivLLM/blob/ee91fb66a2a065003c5bb8986f623f1c1da4fff6/src/myvllm/engine/llm_engine.py#L17-L32))
* 父进程在 Spawn 前初始化清理状态，捕获 `BaseException`，条件允许时请求正常退出，发送 SIGTERM，以五秒上限 Join，并对仍卡住的 Worker 升级到 `kill()`，最后销毁本地 Process Group。正常 `exit()` 也复用相同的幂等有界清理。([修复](https://github.com/KevinWu8192/MinivLLM/blob/ee91fb66a2a065003c5bb8986f623f1c1da4fff6/src/myvllm/engine/llm_engine.py#L45-L180))

### 4. 选择最小 CUDA Graph 并中和 Padding Row

**涉及文件**

* `src/myvllm/engine/model_runner.py`
* `tests/test_engine_boundaries.py`

**问题**

* CUDA Graph 按 Capture Size 降序插入，但 Decode 会选择第一个大于等于实时 Batch 的 Capture Size。三个请求因此选择 16 Row Graph 而不是四 Row Graph，浪费计算并暴露更多 Padding 状态。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/9a43595c7fc3fba1e234bb7fe1160c2e56e808fe/src/myvllm/engine/model_runner.py#L435-L451))
* Replay 会刷新实时 Row，却没有清理所选 Capture Batch 的全部 Row。Padding Row 中残留的 `slot_mapping` 与 Block Table 条目可能让非活跃 Token 写入真实 KV Cache Slot。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/9a43595c7fc3fba1e234bb7fe1160c2e56e808fe/src/myvllm/engine/model_runner.py#L435-L451))

**示例**

Capture Size 为 `{16, 8, 4, 2, 1}`、运行时 `bs = 3` 时，旧字典迭代会选择 16。只有 Row 0–2 获得新请求数据；Row 3–15 可能保留上一次 Replay 的 Slot ID 与物理 Block 映射，导致 Graph 对旧 Cache 位置执行 13 个虚假请求。

**修复**

* Decode 显式执行 `min(graph_bs for graph_bs in self.graphs if graph_bs >= bs)`。Replay 前将 Padding Input 与 Context Length 清零，把 Padding Slot Mapping 与 Block Table 填为 `-1`，再仅复制实时请求 Row。([修复](https://github.com/KevinWu8192/MinivLLM/blob/6972cb5c3853e5e0099ad7947688c704391fb3af/src/myvllm/engine/model_runner.py#L435-L465))
* 边界测试使用 `bs = 3`，验证选择四 Row Graph、检查每个非活跃 Sentinel，并确认只返回三个输出 Row。([修复](https://github.com/KevinWu8192/MinivLLM/blob/6972cb5c3853e5e0099ad7947688c704391fb3af/tests/test_engine_boundaries.py#L100-L147))

## Qwen3-32B Benchmark 结果

测试系统使用两张 NVIDIA RTX PRO 6000 Blackwell Server Edition GPU（每张 95.0 GiB）、TP=2、50 个 Intel Xeon Platinum 8470Q vCPU、240 GiB 主机内存、BF16 权重，并启用 CUDA Graph。

| Suite | Shape | 关键结果 |
|---|---|---|
| 高并发 | 16 请求 × 4,075 Prompt + 256 输出 | 5,954.27 Prefill tok/s、435.13 聚合 Decode tok/s、201.48 端到端输出 tok/s、总计 20.33 s |
| 长 Decode | 2 请求 × 8,169 Prompt + 4,096 输出 | 68.64 聚合 Decode tok/s、66.57 端到端输出 tok/s、总计 123.05 s |
| 最大输出 | 1 请求 × 8,169 Prompt + 32,768 输出 | 完成全部 32,768 Token；35.60 Decode tok/s；Decode 920.45 s、总计 923.03 s |
| Prefix Cache | 8 请求 × 约 16K Prompt + 256 输出 | 热缓存命中率 98.69%；Prefill 从 31.902 s 降到 1.087 s；总时间提升 3.91× |

Prefix Cache 共复用 129,024 Token，即每请求 16,128 Token，正好是 63 个完整的 256-Token Block。最大输出测试达到 40,937 个总 Token，距离配置的 40,960 位置容量仅 23。原始证据分别提交为[高并发](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/results/qwen3_32b/stress_high_concurrency.json)、[长 Decode](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/results/qwen3_32b/stress_long_decode.json)、[最大输出](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/results/qwen3_32b/stress_max_output.json)和 [Prefix Cache](https://github.com/KevinWu8192/MinivLLM/blob/bfefde51df92d1bbc1b3291c10c0b5e88077f3a6/tests/results/qwen3_32b/stress_prefix_cache.json)。
