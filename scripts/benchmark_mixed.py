"""Measure engine-level latency under deterministic, staggered mixed arrivals.

This is an offline step-loop benchmark, not an HTTP serving benchmark.
Run one policy per fresh process. Use identical model, workload, and environment.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summary(values):
    return {"count": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "p99": percentile(values, 99)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=["prefill_first", "decode_first"], default="decode_first")
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--arrival-interval-ms", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests < 2 or args.output_tokens < 2 or args.budget < 1 or args.arrival_interval_ms < 0:
        parser.error("requests/output-tokens >= 2, budget >= 1, arrival interval >= 0 required")
    import torch
    from nanovllm import LLM, SamplingParams
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for performance evidence")
    llm = LLM(args.model, max_model_len=4096, max_num_seqs=32,
              max_num_batched_tokens=args.budget, scheduling_policy=args.policy,
              enforce_eager=args.eager, gpu_memory_utilization=.8)
    try:
        rng = random.Random(args.seed)
        vocab = llm.config.hf_config.vocab_size
        workload = []
        for i in range(args.requests):
            length = 2048 if i % 4 == 1 else 64
            # Independent prefixes avoid accidental reuse in the cold workload.
            tokens = [rng.randrange(1, min(vocab, 30000)) for _ in range(length)]
            workload.append(dict(index=i, prompt_tokens=length, tokens=tokens,
                                 arrival_s=i * args.arrival_interval_ms / 1000))
        # Warm up each stage with different tokens, including mixed batches and
        # random sampling compilation; compilation is excluded from measurements.
        sp = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        llm.generate([[0]*64, [0]*2048], sp, use_tqdm=False)
        llm.generate([[0]*64], SamplingParams(temperature=.6, top_k=20, top_p=.9,
                                            max_tokens=8, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        pending = 0
        records = {}
        steps = []
        while pending < len(workload) or not llm.is_finished():
            elapsed = perf_counter() - started
            while pending < len(workload) and workload[pending]["arrival_s"] <= elapsed:
                item = workload[pending]
                seq_id = llm.add_request(item["tokens"], sp)
                records[seq_id] = dict(index=item["index"], prompt_tokens=item["prompt_tokens"],
                    arrival_s=item["arrival_s"], admitted_s=perf_counter()-started, emissions=[])
                pending += 1
            if llm.is_finished():
                # No busy waiting; only wait until the next known arrival.
                from time import sleep
                sleep(min(.001, max(0., workload[pending]["arrival_s"] - (perf_counter()-started))))
                continue
            step_started = perf_counter()
            llm.step()
            # run() currently copies sampled IDs to CPU. Synchronize explicitly
            # so timing stays valid if that implementation detail changes.
            torch.cuda.synchronize()
            now = perf_counter() - started
            stats = llm.last_step_stats
            steps.append(dict(at_s=now, duration_ms=(perf_counter()-step_started)*1000, **stats))
            for seq_id, token_id in llm.last_step_token_ids:
                records[seq_id]["emissions"].append(dict(at_s=now, token_id=token_id))
        duration = perf_counter() - started
        rows = []
        all_gaps = []
        for row in records.values():
            times = [e["at_s"] for e in row["emissions"]]
            assert len(times) == args.output_tokens
            gaps = [(b-a)*1000 for a, b in zip(times, times[1:])]
            all_gaps.extend(gaps)
            row.update(ttft_ms=(times[0]-row["arrival_s"])*1000,
                       service_ttft_ms=(times[0]-row["admitted_s"])*1000,
                       e2e_ms=(times[-1]-row["arrival_s"])*1000,
                       tpot_ms=(times[-1]-times[0])*1000/(len(times)-1), itl_ms=gaps)
            rows.append(row)
        rows.sort(key=lambda row: row["index"])
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except subprocess.CalledProcessError:
            revision, dirty = None, None
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "triton", "flash-attn", "safetensors")}
        report = dict(config=vars(args), revision=revision, dirty=dirty, versions=versions,
            model=str(Path(args.model).resolve()), model_config=llm.config.hf_config.to_dict(),
            python=platform.python_version(), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
            duration_s=duration, output_tokens_per_second=args.requests*args.output_tokens/duration,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            ttft_ms=summary([r["ttft_ms"] for r in rows]),
            tpot_ms=summary([r["tpot_ms"] for r in rows]), itl_ms=summary(all_gaps),
            short_ttft_ms=summary([r["ttft_ms"] for r in rows if r["prompt_tokens"] == 64]),
            long_ttft_ms=summary([r["ttft_ms"] for r in rows if r["prompt_tokens"] == 2048]),
            mixed_steps=sum(s["prefill_tokens"] > 0 and s["decode_tokens"] > 0 for s in steps),
            requests=rows, steps=steps)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps({key: report[key] for key in ("output_tokens_per_second", "ttft_ms",
            "tpot_ms", "itl_ms", "mixed_steps")}, indent=2))
        print(f"Raw measurements: {path}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
