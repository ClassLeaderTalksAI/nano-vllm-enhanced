# Qwen3.8-27B：三天项目中的独立 vLLM 部署

决策日期：2026-10-17。用户选择保留三天期限，完成 nano-vLLM 改造，另用成熟引擎部署 Qwen3.8。

本目录提供部署启动器；`scripts/probe_openai.py` 提供真实请求验收和 HTTP 测量。
这部分贡献是部署、参数配置和实验；模型实现、调度和 kernels 由 vLLM 提供。
nano-vLLM 的模型 registry 没有增加 Qwen3.8。

**当前状态：启动参数和客户端已经本地检查、客户端经过假服务测试；GPU 部署尚未执行。**

## 1. 硬件与范围

- 优先规划单张 80GB NVIDIA GPU，BF16，最大上下文先限制为 4096，最多 4 条序列，纯文本。
- 发布 checkpoint 索引约 55.6GB。80GB 是留出 recurrent/KV/工作区余量的规划建议，具体容量需要实测。
- 16–24GB GPU 可做 nano-vLLM 小模型实验，不能直接全量驻留该 BF16 checkpoint。
- 两个服务轮流占用同一张 GPU；启动本服务前先退出 nano-vLLM 进程，避免显存竞争。
- 不增加 MTP、量化、多模态、Kubernetes 或双框架性能对比，先完成一个可测量的服务。

官方参考：[模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、
[vLLM 配方](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)、
[CLI 参数](https://docs.vllm.ai/en/stable/cli/serve/)。配方要求 Transformers >=5.8；启动器会检查容器版本。

## 2. 在云 VM 终端准备权重与镜像

要求 Docker 能识别 NVIDIA GPU，并安装 Hugging Face Hub CLI。模型保持固定 commit：

```bash
# 在仓库根目录执行。以下是 Linux bash 命令。
python3 -m venv .venv-download
.venv-download/bin/python -m pip install huggingface-hub
MODEL_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
.venv-download/bin/hf download Qwen/Qwen3.8-27B \
  --revision "$MODEL_REV" --local-dir models/Qwen3.8-27B

mkdir -p results/qwen38
# 首次选择当前官方镜像，再将它解析到不可变 digest。它仍需通过本机 GPU 验收。
docker pull vllm/vllm-openai:latest
VLLM_IMAGE=$(docker image inspect vllm/vllm-openai:latest --format '{{index .RepoDigests 0}}')
printf '%s\n' "$VLLM_IMAGE" > results/qwen38/image.txt
nvidia-smi -q > results/qwen38/nvidia-smi-before.txt
```

`latest` 只用于首次获取镜像；后续启动强制使用 digest。通过验收后，复跑直接读取 `image.txt`，
避免镜像更新改变实验。镜像中的依赖与 nano-vLLM 的本地 `.venv` 分离。

如果当前镜像不满足版本或模型兼容要求，应按官方配方选择镜像并重新记录 digest；
不要通过升级 nano-vLLM 的 Transformers 4.x 环境来解决另一个引擎的依赖问题。

## 3. 启动：终端 A

```bash
VLLM_IMAGE=$(cat results/qwen38/image.txt)
python3 -m deploy.qwen38.serve \
  --model-dir models/Qwen3.8-27B --image "$VLLM_IMAGE" \
  --manifest results/qwen38/launch.json
```

可先加 `--dry-run` 检查命令；dry-run 不检查 GPU、下载完整性或启动成功。
实际启动会检查模型配置、safetensors 文件、GPU 可见性和容器内依赖，并保存启动清单。
清单保存声明的模型 revision；应保留上面的固定 revision 下载过程，不能把该字段当作独立权重校验。

默认配置：单卡 TP=1、BF16、eager、4096 context、4 sequences、512 token budget；
启用 chunked prefill，关闭 prefix caching；只开放本机 `127.0.0.1:8000`。
这里启用的是 vLLM 的 chunked prefill，不能归为本项目自己实现的功能。

启动日志可能需要等待模型加载和 kernels 初始化。保存日志和实际可用的镜像版本。
不要仅以容器处于 Running 判断成功：继续执行下一节，直到真实请求通过。

## 4. 验收与测量：终端 B

仍在同一台云 VM 的仓库根目录执行，客户端只需要 Python 标准库：

```bash
python3 -m scripts.probe_openai --concurrency 1 --requests 12 \
  --manifest results/qwen38/launch.json --output results/qwen38/c1-r1.json
python3 -m scripts.probe_openai --concurrency 4 --requests 12 \
  --manifest results/qwen38/launch.json --output results/qwen38/c4-r1.json
```

程序按顺序检查 `/health`、`/v1/models`，再发送 `17×19`，要求可见答案为 `323`。
它显式关闭 thinking 和保留历史 reasoning，防止小输出预算全部花在思考内容里。
随后预热两种 prompt，再执行 3 短 : 1 长的文本负载，使用真实返回的 usage 统计 token 数。
输出提前 EOS 是允许的，不虚构固定生成量。失败时结果为 `passed: false`，命令返回非零状态。

测量采用 closed-loop 并发：完成一个请求后，该线程才发送下一个。请求排队时间从实际 HTTP 发起算起，
不包括线程池中尚未发出的任务等待；它不是固定 QPS 的开环压力测试。
复跑时换成 r2/r3 文件名，保留同一模型、镜像、参数和请求内容，至少做 3 次。
12 请求仅供验收和探索性测量，稳定的尾延迟结论需要扩大样本。

结果包括可见答案、usage、结束原因、TTFT、E2E、每次 SSE content event 的时间、错误和吞吐。
**SSE event 可能包含多个 token，因此 event 间隔不标为 token ITL，也不据此声称精确 TPOT。**
需要引擎 token 延迟分布时，再使用与镜像版本配套的 vLLM benchmark/服务端 metrics。
客户端 TTFT 包括 HTTP 和主机调度，不能与 nano-vLLM 的离线 step 计时直接比较。

重复 prompts 会出现在本负载中，因此初版显式关闭 prefix caching。
如果后续开启 cache 做实验，要重新保存启动清单并分别报告 cold/warm 条件。

在 Mac 上查看服务，可单独开终端建立转发：

```bash
ssh -L 8000:127.0.0.1:8000 your-cloud-ssh-alias
```

再访问本机端口。正式延迟实验优先在 VM 本地执行；通过 Mac 测得的结果包含跨网路延迟。

## 5. 停止、失败处理与交付

停止本服务：在终端 A 按 Ctrl-C，或在云 VM 执行 `docker stop qwen38-lab`。
确认 GPU 进程退出后，再运行 nano-vLLM。停止容器不等于停止计费 VM；实验结束后按云平台流程停止/释放实例。

常见失败的检查顺序：

1. GPU/Docker 不可见：先确认驱动及容器 GPU runtime。
2. 不支持模型或 processor：检查记录的 vLLM/Transformers 版本和官方模型配方。
3. OOM：确认没有其他模型进程，保持 4096/4/eager；必要时将 `--max-seqs 1`。
4. 上下文超限：减小测试 prompt，核对返回的 prompt usage。
5. 有 HTTP 返回但无答案：检查 thinking 参数、finish_reason 和 token 预算；空内容不能算通过。
6. 三天内启动排障累计超过约两小时仍未通过：记录失败与日志，先保住 nano-vLLM 主线；不要把失败部署写成成功。

验收完成应留下：image digest、GPU/驱动信息、launch.json、真实 smoke 输出和三次原始测量。
当前这些 GPU 产物尚未生成。本地假服务测试只验证 HTTP 客户端协议，不代表 Qwen3.8 已经部署。

简历应分别归属：nano-vLLM 的 engine implementation，以及 Qwen3.8 的 vLLM deployment/benchmarking。
