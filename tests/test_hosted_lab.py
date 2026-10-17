"""Validate the deployment client's protocol behavior against a local fake server."""
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import perf_counter
from types import SimpleNamespace
import pytest
from deploy.qwen38.serve import command
from scripts.probe_openai import run, stream_result


def events(items):
    return io.BytesIO(b"".join(b"data: " + (item.encode() if isinstance(item, str) else json.dumps(item).encode())
                               + b"\n\n" for item in items))


def test_stream_content_usage_and_finish():
    data = events([{"choices":[{"delta":{"content":"hello "}}]},
        {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]},
        {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}, "[DONE]"])
    row = stream_result(data, perf_counter())
    assert row["text"] == "hello world"
    assert row["usage"]["completion_tokens"] == 2
    assert len(row["inter_content_event_ms"]) == 1
    assert row["ttft_ms"] >= 0 and row["e2e_ms"] >= row["ttft_ms"]


@pytest.mark.parametrize("items,match", [
    ([{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}], "incomplete stream"),
    ([{"error":{"message":"OOM"}}], "server stream error"),
    ([{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}, "[DONE]"], "usage"),
    ([{"choices":[{"delta":{},"finish_reason":"stop"}]}, "[DONE]"], "visible answer"),
])
def test_bad_or_incomplete_stream_fails(items, match):
    with pytest.raises(RuntimeError, match=match):
        stream_result(events(items), perf_counter())


@pytest.fixture
def server():
    payloads = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({"data":[{"id":"qwen38-lab"}]} if self.path == "/v1/models" else {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payloads.append(data)
            text = "323" if "17 times" in data["messages"][0]["content"] else "Cached state helps."
            body = events([{"choices":[{"delta":{"content":text},"finish_reason":"stop"}]},
                {"choices":[],"usage":{"prompt_tokens":16,"completion_tokens":4}}, "[DONE]"]).read()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", payloads
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_http_smoke_then_concurrent_measurement(server):
    base, payloads = server
    result = run(SimpleNamespace(base_url=base, model="qwen38-lab", requests=4,
                                  concurrency=2, max_tokens=16, timeout=5))
    assert result["passed"] and result["completed_requests"] == 4
    assert result["successful_completion_tokens"] == 16
    assert len(payloads) == 7  # smoke, two warmups, four measured requests
    assert all(p["chat_template_kwargs"]["enable_thinking"] is False for p in payloads)
    assert all(p["stream_options"]["include_usage"] for p in payloads)
    assert "inter_content_event_ms" in result["requests"][0]
    assert "itl_ms" not in result and "tpot_ms" not in result


def test_model_must_be_present_before_request(server):
    base, payloads = server
    with pytest.raises(RuntimeError, match="absent"):
        run(SimpleNamespace(base_url=base, model="wrong-model", requests=1,
                            concurrency=1, max_tokens=16, timeout=5))
    assert not payloads


def test_launcher_uses_pinned_image_and_separate_runtime(tmp_path):
    args = SimpleNamespace(image="vllm/vllm-openai@sha256:"+"a"*64, model_dir=str(tmp_path),
        gpu=0, port=8000, budget=512, max_seqs=4)
    cmd = command(args)
    assert "127.0.0.1:8000:8000" in cmd
    assert "--language-model-only" in cmd and "--enforce-eager" in cmd
    assert "--no-enable-prefix-caching" in cmd
    assert "bfloat16" in cmd and "4096" in cmd
    assert "nanovllm" not in " ".join(cmd)
    args.image = "vllm/vllm-openai:latest"
    with pytest.raises(ValueError, match="immutable"):
        command(args)
