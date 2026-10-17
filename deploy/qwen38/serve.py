"""Launch a separate, digest-pinned vLLM Qwen3.8 text server on a cloud VM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess

MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def command(args):
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.image):
        raise ValueError("--image must be an immutable Docker repository@sha256:digest")
    if not 1 <= args.port <= 65535 or min(args.budget, args.max_seqs) < 1 or args.gpu < 0:
        raise ValueError("valid port and positive budget/max-seqs required")
    return ["docker", "run", "--rm", "--gpus", f"device={args.gpu}",
        "--name", "qwen38-lab", "--shm-size", "8g",
        "-p", f"127.0.0.1:{args.port}:8000", "-v", f"{Path(args.model_dir).resolve()}:/model:ro",
        "--entrypoint", "vllm", args.image, "serve", "/model",
        "--served-model-name", "qwen38-lab", "--host", "0.0.0.0", "--port", "8000",
        "--dtype", "bfloat16", "--tensor-parallel-size", "1",
        "--max-model-len", "4096", "--max-num-seqs", str(args.max_seqs),
        "--max-num-batched-tokens", str(args.budget), "--gpu-memory-utilization", "0.85",
        "--generation-config", "vllm", "--language-model-only",
        "--reasoning-parser", "qwen3", "--enable-chunked-prefill",
        "--no-enable-prefix-caching", "--stream-interval", "1", "--enforce-eager"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-seqs", type=int, default=4)
    parser.add_argument("--manifest", default="results/qwen38/launch.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        cmd = command(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    path = Path(args.model_dir).resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("text_config", {}).get("num_hidden_layers") != 64:
        parser.error("model directory does not match the Qwen3.8-27B architecture")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be the HF commit used to download the checkpoint")
    if not list(path.glob("*.safetensors")):
        parser.error("checkpoint safetensors are missing")
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                         "--format=csv"], text=True)
    # Validate the container environment before committing to a long model load.
    version_check = (
        "import json,vllm,transformers; from packaging.version import Version; "
        "assert Version(vllm.__version__) >= Version('0.17.0'), 'vllm >=0.17 required'; "
        "assert Version(transformers.__version__) >= Version('5.8.0'), 'transformers >=5.8 required'; "
        "print(json.dumps({'vllm':vllm.__version__,'transformers':transformers.__version__}))")
    versions = subprocess.check_output(["docker", "run", "--rm", "--gpus", f"device={args.gpu}",
        "--entrypoint", "python3", args.image, "-c", version_check], text=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = dict(created_at=datetime.now(timezone.utc).isoformat(), model="Qwen/Qwen3.8-27B",
        declared_model_revision=args.revision, image=args.image, gpu_info=gpu_info,
        versions_output=versions, argv=cmd, status="launching; readiness not yet checked")
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    try:
        result = subprocess.run(cmd)
        record.update(status="container exited", returncode=result.returncode)
    except KeyboardInterrupt:
        record.update(status="launcher interrupted; inspect container status")
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
