# 三天 nano-vLLM 推理项目：范围、计划与面试证据

评估日期：2026-10-16。执行窗口：2026-10-17（周六）至 10-19（周一）。
目标岗位：Cohere / Cerebras 的 inference infrastructure、model serving、platform；engine/runtime 可作为拓展。

**2026-10-17 已确认范围：保留三天，完成 nano-vLLM 改造；另用 vLLM 部署 Qwen3.8-27B。**
前者展示自己的引擎代码，后者展示新模型部署与测量。Qwen3.8 原生适配、GLM、DeepSeek 后移。
Qwen3.8 操作入口：[独立部署手册](../deploy/qwen38/README.zh-CN.md)。

原先 16–24GB GPU 的建议只适用于 nano-vLLM 小模型。两条线复用一台机器时，
优先规划单张 80GB GPU，顺序运行；Qwen3.8 BF16 checkpoint 约 55.6GB，实际余量待实测。
原生代码已有第一版，因此周末把时间优先给 GPU 验收和实验。总投入仍控制在 24–30 小时：
nano 主线约 16–20 小时，Qwen3.8 部署约 4–6 小时，报告/面试约 2–4 小时，具体按验收进展调整。

## 判断

**缩小范围后可行：24–30 小时，单张 Ampere 或更新的 NVIDIA GPU，优先保证正确性和测量闭环。**
这属于个人 inference-engine 项目，可以补充已有 Oracle 分布式系统经验；它本身不构成生产推理系统经验，也不能保证面试录取。
工时是包含 AI 辅助的工程估计，环境和 GPU 排障可能增加时间。

| 原目标 | 三天判断 | 本次采用的范围 |
|---|---|---|
| 支持目前最流行的开源模型 | 全面支持不可行；“最流行”也不是明确兼容标准 | Qwen3 dense 基线，新增明确的 Qwen2.5 / Llama dense 配置适配；先验收两个真实小模型 |
| Top-k / Top-p | 可行，约 3–5 小时含测试 | 每请求参数、温度、greedy、混合批次、过滤顺序与概率检查 |
| Chunked Prefill | 基础已经存在；有价值的是调度升级 | decode 优先、统一 token 预算、混合批次、prefix/preemption 边界，约 8–12 小时含调试 |
| 实验、报告、面试准备 | 必须留时间，约 7–10 小时 | 真实模型对照、TTFT/ITL/TPOT/吞吐、原始 JSON、贡献归属与取舍 |

这些工时有交叉，不是全部机械相加。环境与基线预留约 3–5 小时。

本地起点是 `bb823b3`（上游 chunked-prefill-refactor）。它已有 `num_cached_tokens`、
`num_scheduled_tokens`、首请求分块、prefix-cache KV attention，以及中间 chunk 不提交 token 的逻辑。
因此简历不能写“从零为缺少该功能的版本实现 Chunked Prefill”。本次增量是完整调度策略和验证。

“最热新模型”与“适合三天做正确”不同：Qwen3.5 已有 Gated DeltaNet / full-attention 混合结构，
不能复用这里的普通 GQA 路径就声称支持。MoE、MLA、视觉编码、量化、滑动窗口、YaRN 都不在本次范围。
参见 [Qwen3.5 官方结构说明](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)。

## 周六：GPU 验收与 Qwen3.8 服务启动（8–10 小时）

1. 约 1–2 小时：准备云端 Linux / 单 GPU / Python 3.12 和 Docker，记录 GPU、驱动、CUDA，固定模型 revision。
2. 约 2–3 小时：nano-vLLM 的 Qwen3-0.6B / Qwen2.5-0.5B 验证；检查采样、eager、CUDA Graph。
3. 约 3–4 小时：退出 nano 进程；独立 vLLM 容器加载 Qwen3.8-27B，完成 health、模型发现和真实 streaming 请求。
4. 剩余时间：理解采样过滤顺序、QKV bias / QK norm 和两个后端的边界。Llama 验收移出必交范围。

验收：CPU tests 通过；GPU 上至少 Qwen3 和 Qwen2.5 的验证脚本通过。GPU 环境排障超过两小时，优先换预装环境；
若当天 nano 仍不通过，周日先解决 GPU 基线。Qwen3.8 的下载/启动问题单独记录，不通过不算交付。

## 周日：调度与 KV 状态（9–11 小时）

1. 用例推演：运行中的短请求 + 新到达长 prompt，画出每一步 token 预算。
2. decode 每序列先取一个 token，余量给 prompt；有 prefill 的 mixed batch 用 paged varlen attention。
3. 中间 chunk 只更新 KV；末尾 chunk 才采样。核对 Q 长度、K 长度、绝对 position 和物理 slot。
4. 验证 255 / 256 / 257 token、共享前缀、EOS、max_tokens、KV 耗尽时抢占与重算。
5. eager 与 graph、整段与分块，逐 token 对照；真实模型数值差异必须定位，不能只看文本流畅。
6. 给 Qwen3.8 预留约 2 小时，完成并发 1/4 的 HTTP 探测；保存 usage、TTFT、E2E、错误和配置。

验收：每步 scheduled tokens 不超预算；无重复调度；引用计数与资源回收正确；GPU correctness 全通过。
若 mixed attention 在 GPU 上持续失败，保留 `prefill_first` 作为回退路径；将 mixed mode 标为实验实现，停止写性能提升结论。

## 周一：测量、分析与讲解（7–9 小时）

1. 固定同一 GPU、模型 revision、精度、输出长度、workload seed、并发上限、eager/graph 状态。
2. 同一 budget 下对比 prefill_first / decode_first；再扫描 budget=128/256/512。
3. 每个条件至少 3 次独立进程运行；关键配置反向顺序再跑一次，观察温度、竞争或频率漂移。
4. 保存每请求 TTFT、TPOT、每 token ITL、E2E、吞吐、混合步数量、峰值显存和版本。
5. 分别看短请求和长请求；报告长 prompt 的 TTFT 是否变差，以及 decode ITL 是否改善。
6. 阅读一份 profiler trace（若时间允许），区分 kernel 时间、CPU launch、采样排序和调度开销。
7. 写英文简历条目，准备 90 秒项目介绍与下面的追问。
8. Qwen3.8 的关键配置各复跑 3 次，记录部署体验和资源限制。它的 HTTP 指标独立报告，不与 nano 离线计时直接比较。

验收：正确性结果 + 原始 JSON + 至少一张对照表；结论即使是“无明显提速”也保留。
64 个请求只能形成探索性结果，p99 稳定性有限；时间允许再扩大样本。不要事先承诺降低多少百分比。

## 项目对两个目标公司的意义

Cohere 的 inference infrastructure 职位涉及模型部署与运行平台；Cerebras 的 inference platform 职位也包含分布式平台工程。
这个项目最有价值的是让你把已有控制平面、生命周期和故障处理经验连接到 token 调度、KV 容量、吞吐与尾延迟。
参见 [Cohere 官方职位](https://jobs.ashbyhq.com/cohere/41f23dad-9da2-451a-bd1e-a1800437cb64/)、
[Cerebras 官方职位](https://jobs.ashbyhq.com/cerebras/80775253-7fba-4a3b-87ee-0e70ce8e995c)。职位内容会变化，投递时复核。

- 对 Cohere：能解释模型接入正确性、长短请求竞争、缓存与延迟。
- 对 Cerebras：能解释与硬件相关的 compute/memory 瓶颈和调度取舍；本项目只有 NVIDIA 实验，不能声称有 WSE 开发经验。
- 对两个方向：三天项目用作个人技术证据；Oracle 经历仍负责证明生产 ownership、可靠性与跨团队交付。

## 面试必须能回答

1. 上游本来有什么？你的增量是什么？为什么没有重新写一套引擎？
2. 为什么只做 prompt chunking 还可能阻塞 decode？decode-first 的代价是什么？
3. 一条 cached=256、scheduled=128 的请求，Q 长度、K 长度、positions 分别是什么？
4. 为什么中间 chunk 不应消费一次采样、也不能提交 token？
5. prefix block 什么时候可以发布？部分计算的块为什么不能复用？
6. 混合批次为什么走 varlen eager，而纯 decode 能走 CUDA Graph？潜在性能损失在哪里？
7. chunk 小是否一定更快？长请求饥饿、head-of-line blocking、KV 全量预留有哪些限制？
8. top-k 后 top-p 的概率要不要重新归一化？温度在哪里应用？ties 怎么处理？
9. Qwen2.5 / Qwen3 / Llama 哪些结构相同，哪些必须显式区别？为什么不直接支持 Qwen3.5？
10. TTFT、TPOT、ITL 怎么定义？为什么只报总 tokens/s 不够？GPU 结果哪些条件受控？

## 简历写法与门槛

GPU 验证前：只能描述实现和 CPU/reference 验证，不能写真实模型或 CUDA 性能已验证。
GPU 验证通过后，可用以下英文骨架，并按实际模型、负载、结果修改：

> Extended nano-vLLM with per-request top-k/top-p sampling and decode-first chunked prefill scheduling;
> validated dense model compatibility against Transformers and benchmarked latency–throughput trade-offs under mixed-length workloads.

如果还没有 GPU 结果，去掉后半句，改为 “tested scheduler/KV invariants and model semantics using CPU reference tests”。
只有实测后，才添加 “reduced p95 inter-token latency by X% under [明确负载] on [GPU]”。

不可写：从零实现 vLLM、生产集群 serving、多模型全量兼容、上游 benchmark 数字、Cerebras 专有硬件优化。

## 当前交接状态

- 已完成：本地代码、CPU/reference correctness tests、GPU 验证入口、mixed-arrival benchmark。
- 本地验证：65 项测试通过（57 项引擎/reference 测试 + 8 项独立部署客户端测试）；模型测试使用小型随机权重和 CPU reference attention。
- 未完成：用户 10/16 反馈云 GPU 尚未创建，当前尚无可连接端点；真实 checkpoint、FlashAttention、CUDA Graph、TP 和性能均待验证。
- 下一步：按 [GPU 手册](GPU_RUNBOOK.md) 验收 Qwen3，再验收 Qwen2.5；不要先开始跑性能图。
- 独立 Qwen3.8：启动器、真实请求探测与手册已备好，云 GPU 部署待执行；入口见上面的独立部署手册。

## 最新模型候选的后续评估

用户提出的 GLM-5.3、Qwen3.8、DeepSeek V4.1 Flash 已按官方配置核实。
Qwen3.8-27B 采用成熟引擎部署；其混合注意力原生适配需要新的 recurrent/conv 状态管理，留待后续。
此前的原生适配取舍见 [最新模型适配评估](MODERN_MODEL_FEASIBILITY.zh-CN.md)。
