<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## This fork: nano-vLLM Enhanced

Personal inference-engine extension based on upstream commit `bb823b3`.
The upstream already implements basic prompt chunking. This fork adds:

- Per-request top-k / top-p sampling and temperature-zero greedy decoding.
- Decode-first scheduling with a shared token budget and mixed prefill/decode batches.
- Explicit dense Qwen2.5 / Qwen3 / Llama architecture validation, independent QK
  normalization and QKV bias, and Llama 3.1/3.2 RoPE scaling.
- CPU correctness tests, a real-checkpoint GPU validation script, and a mixed-arrival
  benchmark that saves per-token timing and experiment metadata.

**Validation:** CPU tests pass (reference attention in architecture tests).
Real checkpoints, FlashAttention, CUDA graphs, tensor parallel execution and performance
have **not yet been validated on a GPU in this fork**. Model support is conditional
on the documented dense configuration checks, not all variants of those families.

- [Three-day plan and interview positioning (中文)](docs/THREE_DAY_PLAN.zh-CN.md)
- [Implementation, support matrix and limitations](docs/IMPLEMENTATION.md)
- [Cloud GPU validation and benchmark runbook](docs/GPU_RUNBOOK.md)
- [Qwen3.8-27B deployment in a separate vLLM container (中文)](deploy/qwen38/README.zh-CN.md)

The three-day scope also includes a separately hosted Qwen3.8-27B experiment using
vLLM. Its launcher and HTTP probe are prepared; deployment is pending cloud GPU access.
This is deployment/measurement work, independent of native nano-vLLM model support.

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/models/Qwen3-0.6B", max_num_batched_tokens=256,
          scheduling_policy="decode_first", enforce_eager=True)
outputs = llm.generate(["Explain KV caching."],
                       SamplingParams(temperature=0.7, top_k=40, top_p=0.9))
```

For a controlled scheduler comparison, use `scheduling_policy="prefill_first"`
with the same token budget. That baseline retains upstream prompt chunking.

The features and performance table below describe the **upstream project**;
the published throughput numbers are not measurements of this fork.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
