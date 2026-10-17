# Cloud GPU acceptance runbook

Status on 2026-10-17: **not GPU executed**. No cloud machine or model directory is
connected to this workspace. These are the next acceptance steps, not completed results.

Scope update on 2026-10-17: also deploy Qwen3.8-27B in a **separate vLLM container**.
Use the [Qwen3.8 runbook](../deploy/qwen38/README.zh-CN.md). The 16–24 GB recommendation
below applies only to nano-vLLM's small models; plan an 80 GB GPU for the BF16 Qwen3.8
experiment and validate capacity. Run the two engines sequentially on a shared GPU.

## 1. Prepare one bounded environment

Use one Linux NVIDIA Ampere/Ada/Hopper GPU (compute capability >= 8.0), preferably
with at least 16–24 GB VRAM, and Python 3.12. Start with Qwen3-0.6B; then
Qwen2.5-0.5B-Instruct. A larger GPU is convenient, not a requirement for those models.
Check actual cloud capacity/quota before the weekend. Start with a single VM;
Kubernetes, distributed P/D, speculative decoding and WSE execution are outside scope.

The Mac is for editing and CPU reference tests. Do not install CUDA dependencies
there expecting native nano-vLLM GPU inference.

In the cloud VM terminal, from this repository root:

```bash
nvidia-smi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip packaging ninja psutil setuptools wheel
# Use a PyTorch CUDA wheel compatible with the VM driver. Prefer the provider's
# tested PyTorch image/environment when available; record its exact versions.
python -m pip install 'torch>=2.4'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())'
python -m pip install 'flash-attn>=2.5.8,<3' --no-build-isolation
python -m pip install -e '.[test]'
python -m pip check
mkdir -p results/environment
python -m pip freeze > results/environment/pip-freeze.txt
nvidia-smi -q > results/environment/nvidia-smi.txt
python -m pytest -q
```

If the CUDA check fails, resolve the wheel/driver combination first. If FlashAttention
compilation fails, use an environment matching its official requirements rather than
spending the weekend porting a backend. Tests run on CPU even on that VM; they do not
replace the GPU acceptance below. See [FlashAttention installation/support](https://github.com/Dao-AILab/flash-attention#installation-and-features).

These installation commands are a starting recipe, not a tested GPU lockfile. Once
one combination passes acceptance, preserve its freeze and use it for every experiment.
Do not compare across dependency changes.

## 2. Fix model revisions and download

Choose and record a specific Hugging Face commit revision before downloading. The
following interactive commands ask you for the actual revision, rather than inventing
one. `hf download` is provided by the installed Hugging Face Hub CLI.

```bash
read -r -p 'Qwen3 commit revision: ' QWEN3_REV
hf download Qwen/Qwen3-0.6B --revision "$QWEN3_REV" --local-dir models/Qwen3-0.6B
read -r -p 'Qwen2.5 commit revision: ' QWEN25_REV
hf download Qwen/Qwen2.5-0.5B-Instruct --revision "$QWEN25_REV" --local-dir models/Qwen2.5-0.5B-Instruct
printf 'Qwen3=%s\nQwen2.5=%s\n' "$QWEN3_REV" "$QWEN25_REV" > results/environment/model-revisions.txt
```

Run these in **bash on Linux** (`read -p` is bash syntax). Obtain revisions from the
models' Files and versions pages. Keep the local model directories out of Git.
Optional Llama validation needs authorized access to its weights; use the model's
normal access flow. Do not substitute a different architecture to bypass a gate.

## 3. Correctness before benchmarking

```bash
python -m scripts.validate_gpu --model models/Qwen3-0.6B --output results/qwen3-correctness.json
python -m scripts.validate_gpu --model models/Qwen2.5-0.5B-Instruct --output results/qwen25-correctness.json
# Optional, after obtaining the checkpoint through its normal access flow:
python -m scripts.validate_gpu --model models/Llama-3.2-1B-Instruct --output results/llama-correctness.json
```

Each command runs four **isolated processes**: HF greedy reference, full-prefill
baseline in eager mode, chunked/mixed eager, and chunked/mixed with decode CUDA graphs.
Prompt lengths are 7/255/256/257/513, output length 16, and EOS is ignored for fixed
work. It repeats a prompt after completion to exercise prefix reuse, then checks
mixed sampling parameters and final resource reclamation. The synthetic IDs are
valid vocabulary entries; this validates computation, not natural-language quality.

Pass criteria:

- Saved report says `passed: true`; every generated token matches HF and the repeat.
- No missing-weight, unsupported-config, CUDA, graph or KV error.
- Top-k/top-p mixed with greedy executes and returns the requested number of tokens.

A mismatch is a failure to investigate. Kernel precision or batch shapes can change
near-tied argmaxes; inspect the first divergent position and logits before deciding
whether the cause is a bug or numerical sensitivity. Do not loosen the result check
just to make the report green.

The script does not prove full TP compatibility. If two GPUs are available, separately
run `--tp 2` on a divisible configuration; otherwise keep TP labeled unverified.
It does not verify every checkpoint or tokenizer chat template. After passing the
synthetic tests, run `example.py` with the appropriate chat template for a human demo.

## 4. Controlled mixed-arrival experiments

Run each command in a fresh process, with the same model and runtime. First compare
both policies at the **same token budget**; changing budget and policy together
would confound the result. The baseline already chunks prompts.

```bash
for repeat in 1 2 3; do
  for policy in prefill_first decode_first; do
    python -m scripts.benchmark_mixed --model models/Qwen3-0.6B \
      --policy "$policy" --budget 256 --seed 7 \
      --output "results/qwen3-${policy}-256-r${repeat}.json"
  done
done
```

Then scan 128/256/512 for decode_first using the same seed and workload. For an eager
versus graph comparison, repeat **both** policies with `--eager`; do not compare eager
baseline with graph-enabled enhanced and attribute all gains to scheduling.

Default workload: 64 requests, 64 output tokens each, staggered at 20 ms intervals;
three short (64 input tokens) to one long (2048 input tokens), greedy, ignore EOS,
max 32 sequences per step. Independently generated synthetic prefixes avoid intentional
prefix hits. Warmup uses other prompts and is excluded. The model and CUDA graph
initialization are excluded; peak allocated VRAM includes the persistent KV pool.

The benchmark records engine-level events from `step()`. It is **not an HTTP service
or network benchmark**. Arrivals due during a GPU step are admitted at the next CPU
loop iteration. Saved scheduled-arrival and actual-admission times let you inspect
this load-generator lag. Overly slow steps can distort fine-grained arrival traffic.
It also saves individual token timestamps and IDs so comparisons remain inspectable.

Definitions:

- TTFT: scheduled arrival → first emitted token, including host-loop admission delay.
- Service TTFT: actual engine admission → first emitted token.
- ITL: each gap between emitted tokens, including pauses caused by other work.
- TPOT: (last token time − first token time) / (generated tokens − 1), per request.
- E2E: scheduled arrival → last token.
- Output throughput: all emitted tokens / elapsed workload time, including idle arrival gaps.
- `mixed_steps`: a step containing both prompt and decode tokens. If zero under the
  enhanced configuration, adjust the arrival interval/output length and report why.

Full raw JSON includes each request, all step counters/durations, config, package
versions, GPU, CUDA, model config, Git revision and dirty status. Save the model
revision and driver information alongside it. Record the final source diff/commit:
`revision` plus `dirty=true` alone does not identify the modified code.

Inspect both short/long TTFT, p95/p99 ITL, per-request TPOT and throughput across
repeats. Report sample counts. A 64-request p99 is exploratory, and pooled token-gap
p99 is different from the distribution of per-request TPOT. Do not mix those labels.
No fixed improvement threshold is assumed. If throughput falls or long TTFT rises,
explain the trade-off and the cost of eager mixed batches.

For the three-day deliverable, the upstream-policy ablation is sufficient. An external
vLLM comparison is optional and needs a separate version-locked environment; it must
hold model revision, precision, workload and output work fixed.

## 5. Evidence and completion gates

| Gate | Required artifact | Current status |
|---|---|---|
| CPU semantics / state invariants | pytest output and source | Passed locally |
| Real Qwen3 correctness | qwen3-correctness.json | Pending |
| Real Qwen2.5 correctness | qwen25-correctness.json | Pending |
| Llama checkpoint | llama-correctness.json | Optional, pending |
| GPU mixed batch + CUDA graph | validated runs in correctness report | Pending |
| Controlled latency / throughput | repeated raw experiment JSON | Pending |
| Résumé performance percentage | reproducible comparison and denominator | No claim yet |

CPU/reference test results cannot fill a GPU acceptance cell. The original upstream
RTX 4070 table cannot fill any result cell for this fork.
