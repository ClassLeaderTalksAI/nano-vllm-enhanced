"""Compare real checkpoint greedy outputs in isolated processes; requires CUDA.

Run from the repository root: python -m scripts.validate_gpu --model /models/Qwen3-0.6B
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def prompts(vocab_size):
    # Synthetic token IDs make the block boundaries exact and reproducible.
    base = [1 + (i * 17) % (min(vocab_size, 1000) - 1) for i in range(513)]
    return [base[:n] for n in (7, 255, 256, 257, 513)]


def worker(args):
    import torch
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    inputs = prompts(config.vocab_size)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("an Ampere-or-newer CUDA GPU is required")
    if args.worker == "hf":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=config.dtype,
                                                    attn_implementation="eager").cuda().eval()
        outputs = []
        with torch.inference_mode():
            for tokens in inputs:
                ids = torch.tensor([tokens], device="cuda")
                result = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=args.output_tokens, min_new_tokens=args.output_tokens,
                    eos_token_id=None, pad_token_id=0, use_cache=True)
                outputs.append(result[0, len(tokens):].tolist())
        repeated = outputs[3]
        sampling_lengths = None
    else:
        from nanovllm import LLM, SamplingParams
        full = args.worker == "full_eager"
        llm = LLM(args.model, max_model_len=1024, max_num_seqs=8,
            max_num_batched_tokens=2048 if full else 128,
            scheduling_policy="prefill_first" if full else "decode_first",
            enforce_eager=args.worker != "chunked_graph", tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.memory_utilization)
        try:
            sp = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
            outputs = [o["token_ids"] for o in llm.generate(inputs, sp, use_tqdm=False)]
            repeated = llm.generate([inputs[3]], sp, use_tqdm=False)[0]["token_ids"]
            sampled = llm.generate([inputs[0], inputs[1]], [
                SamplingParams(temperature=.6, top_k=20, top_p=.8, max_tokens=8, ignore_eos=True),
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)], use_tqdm=False)
            sampling_lengths = [len(o["token_ids"]) for o in sampled]
            assert sampling_lengths == [8, 8]
            assert not llm.scheduler.block_manager.used_block_ids
        finally:
            llm.exit()
    Path(args.result).write_text(json.dumps(dict(mode=args.worker, outputs=outputs,
        repeated=repeated, sampling_lengths=sampling_lengths, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, dtype=str(config.dtype)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local unquantized safetensors checkpoint")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--output", default="results/gpu-correctness.json")
    parser.add_argument("--memory-utilization", type=float, default=.8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--worker", choices=["hf", "full_eager", "chunked_eager", "chunked_graph"])
    parser.add_argument("--result")
    args = parser.parse_args()
    if not 1 <= args.output_tokens <= 128:
        parser.error("output-tokens must be in [1, 128]")
    if args.worker:
        worker(args)
        return
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("hf", "full_eager", "chunked_eager", "chunked_graph"):
            result = str(Path(tmp) / (mode + ".json"))
            print(f"Validating {mode}", flush=True)
            subprocess.run([sys.executable, "-m", "scripts.validate_gpu", "--model", args.model,
                "--worker", mode, "--result", result, "--output-tokens", str(args.output_tokens),
                "--tp", str(args.tp), "--memory-utilization", str(args.memory_utilization)], check=True)
            results.append(json.loads(Path(result).read_text()))
    reference = results[0]
    for row in results:
        row["matches_hf"] = row["outputs"] == reference["outputs"]
        row["repeat_matches"] = row["repeated"] == reference["outputs"][3]
    passed = all(row["matches_hf"] and row["repeat_matches"] for row in results)
    report = dict(passed=passed, model=str(Path(args.model).resolve()), tp=args.tp,
                  prompt_lengths=[7, 255, 256, 257, 513], runs=results)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"{'PASS' if passed else 'FAIL'}: {path}")
    if not passed:
        raise SystemExit("greedy mismatch: inspect the saved token IDs before benchmarking")


if __name__ == "__main__":
    main()
