"""Smoke-test and measure a separately hosted OpenAI-compatible streaming server.

Standard library only. HTTP chunk timing is not GPU token timing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import urllib.request


def stream_result(lines, started):
    events, text, reasoning, usage, finish = [], [], [], None, None
    done = False
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        item = json.loads(payload)
        if item.get("error"):
            raise RuntimeError(f"server stream error: {item['error']}")
        if item.get("usage") is not None:
            usage = item["usage"]
        for choice in item.get("choices", []):
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            content = delta.get("content")
            if content:
                at_ms = (perf_counter() - started) * 1000
                text.append(content)
                events.append(dict(at_ms=at_ms, content=content))
    if not done or finish is None:
        raise RuntimeError("incomplete stream: finish_reason and [DONE] are required")
    if not text:
        raise RuntimeError("no visible answer was returned")
    if not usage or usage.get("completion_tokens", 0) <= 0:
        raise RuntimeError("server did not return completion token usage")
    gaps = [b["at_ms"] - a["at_ms"] for a, b in zip(events, events[1:])]
    return dict(text="".join(text), reasoning="".join(reasoning), usage=usage, finish_reason=finish,
        ttft_ms=events[0]["at_ms"], e2e_ms=(perf_counter()-started)*1000,
        content_events=events, inter_content_event_ms=gaps)


def post_stream(base_url, model, prompt, max_tokens, timeout):
    body = dict(model=model, messages=[dict(role="user", content=prompt)], temperature=0,
        seed=7, max_tokens=max_tokens, stream=True, stream_options=dict(include_usage=True),
        chat_template_kwargs=dict(enable_thinking=False, preserve_thinking=False))
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    started = perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return stream_result(response, started)


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def summary(values):
    def quantile(p):
        ordered = sorted(values)
        at = (len(ordered)-1)*p
        lo = int(at)
        hi = min(lo+1, len(ordered)-1)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(at-lo)
    return dict(count=len(values), p50=quantile(.5), p95=quantile(.95), p99=quantile(.99)) if values else None


def run(args):
    base = args.base_url.rstrip("/")
    # Health alone is insufficient: require the selected model and a real answer.
    with urllib.request.urlopen(base + "/health", timeout=args.timeout) as response:
        if response.status != 200:
            raise RuntimeError("server health check failed")
    models = get_json(base + "/v1/models", args.timeout)
    if args.model not in [m.get("id") for m in models.get("data", [])]:
        raise RuntimeError(f"requested served model {args.model!r} is absent from /v1/models")
    smoke = post_stream(base, args.model, "What is 17 times 19? Reply only with the integer.",
                        32, args.timeout)
    if smoke["text"].strip() != "323":
        raise RuntimeError(f"arithmetic smoke check failed: {smoke['text']!r}")
    short = "Explain the difference between LLM prefill and decode in two sentences."
    long = ("Read this synthetic context, then explain why KV caching helps decode in two sentences.\n"
            + "Cache pages store earlier keys and values so a request can continue without recomputing them.\n"*96)
    prompts = [long if i % 4 == 1 else short for i in range(args.requests)]
    # Warm up both prompt shapes. Warmup is separate from the measured workload.
    for prompt in (short, long):
        post_stream(base, args.model, prompt, args.max_tokens, args.timeout)
    def one(item):
        i, prompt = item
        try:
            row = post_stream(base, args.model, prompt, args.max_tokens, args.timeout)
            return dict(index=i, prompt_class="long" if i % 4 == 1 else "short", ok=True, **row)
        except Exception as exc:
            return dict(index=i, ok=False, error=f"{type(exc).__name__}: {exc}")
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, enumerate(prompts)))
    duration = perf_counter() - started
    successes = [row for row in rows if row["ok"]]
    tokens = sum(row["usage"]["completion_tokens"] for row in successes)
    return dict(passed=len(successes) == len(rows), smoke=smoke, server_models=models,
        workload="closed-loop HTTP concurrency; 3 short : 1 long repeating prompts; EOS allowed",
        duration_s=duration, completed_requests=len(successes), failed_requests=len(rows)-len(successes),
        successful_completion_tokens=tokens, output_tokens_per_second=tokens/duration,
        ttft_ms=summary([r["ttft_ms"] for r in successes]),
        e2e_ms=summary([r["e2e_ms"] for r in successes]), requests=rows,
        timing_note="TTFT is client-visible content arrival. SSE content event gaps are not token ITL. No TPOT/ITL claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen38-lab")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.)
    parser.add_argument("--manifest", help="launch manifest saved on the GPU VM")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens, args.timeout) <= 0:
        parser.error("request count, concurrency, token budget and timeout must be positive")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), config=vars(args))
    try:
        if args.manifest:
            report["server_manifest"] = json.loads(Path(args.manifest).read_text())
        report.update(run(args))
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["passed", "error", "ttft_ms", "e2e_ms",
        "output_tokens_per_second", "completed_requests", "failed_requests"] if k in report}, indent=2))
    print(f"Raw report: {path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
