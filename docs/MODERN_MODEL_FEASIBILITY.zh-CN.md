# GLM-5.3 / Qwen3.8 / DeepSeek V4.1 Flash 适配评估

核实日期：2026-10-16。这里的“原生适配”指模型在本仓库的 ModelRunner、调度与状态管理中执行，
而不是把请求转发给 vLLM/SGLang。下列模型均有公开模型仓库；本次只读取配置和权重索引，没有下载权重或运行 GPU。

## 结论与优先级

**优先只选 Qwen3.8-27B。三个都不是当前 dense Qwen3 路径的简单模型别名。**
保留三天期限时，先完成已有 nano-vLLM 改造和真实 GPU 验证；如要使用新模型，可另做成熟引擎上的 Qwen3.8 部署实验。
若目标必须是原生 Qwen3.8，需要额外排期，并减少第一版功能组合。

| 型号 | 已确认的结构 | 当前缺口 | 判断 |
|---|---|---|---|
| Qwen3.8-27B | `qwen3_5`；dense MLP；48 层 Gated DeltaNet + 16 层 gated full attention | recurrent/conv state、partial RoPE、attention gate、多模态 checkpoint 命名与文本分支加载 | 三个中最合适；不能在原三天范围内承诺完成原生集成 |
| GLM-5.3 | `glm_moe_dsa`；78 层；256 routed experts，top-8；MLA/DSA；FP8 checkpoint | 专用 attention/KV 格式、indexer、MoE routing/experts、FP8 权重处理、显存与多卡验证 | 多周级工作；不作为三天项目新增模型 |
| DeepSeek V4.1 Flash | `deepseek_v41`；MoE；CED、压缩/稀疏 attention、Engram；FP8/FP4 checkpoint | 专用执行与缓存布局、encoder/decoder 阶段语义、路由和量化、多卡容量 | 复杂度与硬件需求最高，不作为本次原生适配候选 |

以上为基于结构差异的工程判断，工时不是已经完成的实测。

## 为什么 Qwen3.8-27B 可以优先

它没有 MoE expert routing，能先把问题集中在混合注意力与状态管理。官方配置仍使用
`Qwen3_5ForConditionalGeneration`，不能将它误判成现有的 `Qwen3ForCausalLM`。

官方配置：

- 总层数 64，`full_attention_interval=4`，即 48 个 linear-attention 层和 16 个 full-attention 层。
- full attention 使用 output gate，head_dim=256，但只对 0.25 的维度做 RoPE。
- linear attention 需要长度为 4 的卷积状态，以及 Gated DeltaNet recurrent state。
- checkpoint 有 `model.language_model.*` 和视觉权重；文本实现要显式映射和验证语言权重，不能任意忽略未知权重。
- 当前仓库限制 Transformers 4.x，该模型配置记录 5.x；参考实现和依赖迁移需要单独验证。

本次读取的权重索引 `metadata.total_size` 为 55,562,855,904 bytes（约 55.6 GB，51.7 GiB），
包括整个发布 checkpoint。该值不是运行显存：还需要 recurrent state、KV、工作区和加载余量。
原生非量化短上下文验证可优先考虑单卡 80GB；这是容量规划建议，不是已确认的可运行配置。
24GB A10/L4 无法全量驻留该 BF16 checkpoint，另做量化/offload 会增加范围。

### 真正影响 Chunked Prefill 的部分

普通 attention 每个 token 对应 KV，可以用当前 block table 定位与共享。
Gated DeltaNet 则维护对历史的 recurrent 摘要及卷积状态；继续下一个 chunk 时，这些状态必须跟随请求。

因此需要分别定义：

1. **分块续算**：同一请求的后续 chunk 从正确 recurrent/conv state 继续。
2. **prefix reuse**：命中一个历史前缀时，不只拿到 attention KV，还要拿到该边界对应的 recurrent/conv state。
3. **抢占恢复**：不能只释放 KV 却留下旧 recurrent state；重算或恢复必须有明确策略。
4. **混合批次**：每行状态归属于正确的请求，结束和重新使用 slot 时必须清零。

第一版可以关闭 prefix reuse 和 CUDA graphs，并限制单请求、greedy、文本输入。
但必须明确标为最小模型执行路径；不能说已有功能已全部兼容。

## 原生适配的阶段与验收

以下是包含 AI 辅助、GPU 可用、能复用正确参考算子的粗估：

| 阶段 | 范围 | 额外投入估计 |
|---|---|---:|
| 最小模型路径 | Qwen3.8-27B，单请求、文本、eager、greedy；关闭 prefix cache 与 TP | 30–50 小时 |
| 与引擎集成 | 多请求 chunked prefill、状态生命周期、抢占重算、正确性与真实模型实验 | 累计 60–100 小时以上 |
| 完整功能组合 | recurrent prefix snapshots、CUDA graphs、TP、多模态、优化 kernels | 另行估算，不计入前两档 |

时间不包括长时间 GPU 配额等待；算子/依赖兼容问题可能显著增加投入。
最低验收包括：相同权重/输入对照 Transformers；整段与分块 logits/greedy 对照；连续 decode；
请求状态隔离；结束后 slot 重用；真实 checkpoint 和 GPU 实测。
小型随机配置测试只能证明局部逻辑，不能替代 27B checkpoint 验证。

**2026-10-17 用户决策：保持三天，Qwen3.8 采用独立 vLLM 部署；原生适配延期。**
执行入口：[Qwen3.8 独立部署手册](../deploy/qwen38/README.zh-CN.md)。

## 保留三天期限时的具体选择

- nano-vLLM 主线：已有 Top-k/Top-p、decode-first chunked prefill，先跑真实 Qwen3/Qwen2.5 验证与可复现实验。
- 新模型体验：可用成熟引擎部署 Qwen3.8-27B，测量 serving 延迟和资源；保持独立环境与结果。
- 简历归属：前者描述 engine implementation；后者描述 deployment/benchmarking。调用外部服务不算 nano-vLLM 原生适配。
- GLM-5.3 / DeepSeek V4.1 Flash 暂停原生开发计划，避免三天交付被多个新架构吞掉。

## 来源与保存的证据

1. [Qwen3.8-27B 官方配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
2. [Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
3. [Qwen3.8-27B 权重索引](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json)
4. [GLM-5.3 官方配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)
5. [NVIDIA GLM 模型结构与兼容说明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/glm/glm5.md)
6. [DeepSeek V4.1 Flash 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)
7. [DeepSeek V4.1 Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
8. [vLLM DeepSeek V4.1 Flash 部署与内存说明](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)

配置快照及 revision/SHA256 位于 `results/model-feasibility-2026-10-16/`。
评估完成不等于适配完成；当前模型 registry 仍只接受原有的 dense qwen3/qwen2/llama 路径。
