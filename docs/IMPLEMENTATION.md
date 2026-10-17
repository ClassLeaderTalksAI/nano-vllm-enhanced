# Implementation and validation boundaries

This is a personal extension of GeeeekExplorer/nano-vLLM at `bb823b3`. Upstream
provides paged KV, prefix caching, continuous batching, CUDA graphs, tensor parallel
layers, Qwen3 model code, and **basic chunked prefill**. Those are upstream contributions.

## Dense text model matrix

| Checkpoint family | Implementation | Validation in this workspace |
|---|---|---|
| Qwen3 dense (e.g. Qwen3-0.6B) | Existing architecture; explicit QK norm and model validation | CPU random-weight logits and cached greedy generation vs HF pass; GPU pending |
| Qwen2 / Qwen2.5 dense (e.g. Qwen2.5-0.5B-Instruct) | QKV bias, no QK norm, shared packed weight layout | Same CPU tests pass; GPU pending |
| Llama dense, including Llama 3.1/3.2 RoPE | No QKV bias or QK norm; frequency-scaled RoPE | Same CPU tests pass; GPU/checkpoint access pending |

This matrix specifies compatible **configurations**, not arbitrary descendants or
all model sizes. Only FP16/BF16, SiLU gated MLP, full-context MHA/GQA, default or
`llama3` RoPE are accepted. TP requires divisible query/KV heads, vocabulary and MLP
width; KV-head replication when TP exceeds KV heads is not implemented. Quantized
checkpoints, hybrid attention, MoE, MLA, sliding windows, YaRN and multimodal models
are out of scope and must not be advertised as supported.

`models/registry.py` rejects unsupported types/configurations before GPU initialization.
Qwen2.5 and Llama reuse the existing transformer implementation because parameter
names and the core dense structure match. QKV bias and QK norm are independent;
using “no QKV bias implies QK norm” would silently implement the wrong Llama model.
`utils/loader.py` checks for missing checkpoint tensors, including packed Q/K/V and
MLP shards; tied LM-head weights may be omitted. Unrecognized weights fail loading.

CPU architecture tests use tiny random HF checkpoints and replace only attention
with a reference SDPA implementation. They check logits and cached greedy sequences,
including packed chunks and shared prefixes. These tests do not execute Triton,
FlashAttention, NCCL, GPU kernels or CUDA graphs.

## Sampling contract

`SamplingParams(temperature=1, top_k=0, top_p=1, max_tokens=64)` preserves the
unfiltered categorical default. `temperature=0` selects argmax. Invalid temperatures,
top-k/top-p and nonpositive output limits raise `ValueError`.

1. Convert logits to FP32 and divide each row by its temperature (greedy rows use
   a safe divisor and return the original argmax).
2. If filters are enabled, sort by descending logit and keep exactly k tokens.
   k=0 disables top-k; k >= vocabulary keeps all. Stable ordering breaks ties by
   vocabulary index; some other engines keep all ties at a threshold instead.
3. Apply top-p to the **renormalized top-k distribution**, keeping the token that
   crosses the probability threshold and always at least one candidate.
4. Normalize and use the upstream exponential-race sampler (`p / Exp(1)` argmax).

Per-request values survive sequence serialization. The runner only samples rows
whose scheduled tokens reach the end of the current sequence. Intermediate chunks
write KV and return `None`; they do not consume a random sample or emit a token.
The vocabulary projection still computes one logit row per scheduled sequence,
including intermediate chunks; avoiding that projection is a future optimization.
Unfiltered batches avoid vocabulary sorting. Sorting is a correctness-first PyTorch
implementation; no custom top-k kernel or performance claim is made.

## Decode-first chunked scheduling

`Config.scheduling_policy` accepts `decode_first` (new default) or `prefill_first`
(upstream-style baseline, which already chunks prompts).

For every step:

1. Schedule one token for each running decode, subject to sequence and token limits.
2. Check KV append capacity. If necessary, preempt a trailing running request,
   release its blocks and return it for recomputation. Protect any resident partial
   prompt from being blocked behind a newly preempted request.
3. Admit waiting prompts into the remaining budget. A long prompt keeps its place
   across chunks. `num_cached_tokens + num_scheduled_tokens` defines the valid KV end.
4. If the batch contains any prefill, pack decodes as length-one queries and run
   the existing paged variable-length attention path. Pure decode keeps the existing
   decode kernel and CUDA-graph path. Mixed batches run eagerly.
5. After a successful forward, hash only newly computed full blocks, advance the
   cached position, and append a sampled token only for completed prefills/decodes.
   EOS or the output limit releases all owned block references.

`engine/batch.py` constructs positions, query/key cumulative lengths and physical
slots on the CPU. For cached=256, chunk=128: Q=128, K=384, positions=256..383.
A decode in that batch has Q=1, K=its current sequence length. FlashAttention's
bottom-right causal alignment is required when Q and K differ.

`num_cached_tokens` includes computed prompt/completion tokens, not the latest
sampled token (which is computed on the next decode). Prefix reuse deliberately
leaves the final prompt block for recomputation to obtain logits. A block is
published only once its KV has actually been computed.

`LLMEngine.add_request` returns the sequence ID for tracing. `last_step_token_ids`
records emitted IDs and `last_step_stats` separates prompt/decode token counts.
The existing `step()` return shape and its signed legacy token count remain for
compatibility; use the new stats for mixed-batch measurements.

## Limits and deliberate trade-offs

- This chunks **compute**, not KV allocation. Full prompt KV is reserved on
  admission; memory pressure can still cause head-of-line blocking and preemption.
- Requests whose prompt plus maximum generated KV cannot fit in the pool are
  rejected conservatively. No KV offload, cancellation or distributed P/D exists.
- Sustained decode traffic that consumes the entire budget can delay prefills;
  this version does not guarantee prompt fairness or latency SLOs.
- More chunks mean more launches; mixed batches lose decode-only graph replay.
  Small budgets can worsen TTFT/throughput. Benefits must be measured.
- Pure decode scheduling is also bounded by the token budget. The original
  prefill-first policy is otherwise retained for controlled comparisons.
- GPU compilation, CUDA graph replay and TP behavior remain unverified locally.
- `get_rope` retains upstream's small cache. Non-default scaling uses a hashable
  tuple of scalar HF settings; only the documented RoPE variants are accepted.
- CPU tests exposed FP32 aliasing in the upstream RMSNorm in-place operations.
  Out-of-place normalization preserves residual semantics; compiled GPU performance
  of that adjustment has not been measured.

## Reproduce local checks

On a Mac, install CPU-compatible test dependencies without installing the CUDA-only
package dependency set:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch>=2.4' 'transformers>=4.57,<5' numpy xxhash safetensors tqdm pytest
.venv/bin/python -m pytest -q
```

Tests disable `torch.compile` to exercise CPU tensor semantics. GPU validation runs
in fresh processes outside that test environment, with compilation enabled.

Read [GPU_RUNBOOK.md](GPU_RUNBOOK.md) before claiming real-model compatibility or
speed improvements. The original README benchmark is upstream data, not our result.
