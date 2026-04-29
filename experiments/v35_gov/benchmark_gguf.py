from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests


PROMPTS = [
    {"label": "warmup", "messages": [{"role": "user", "content": "Say one short sentence to confirm you are ready to begin the grounding protocol."}], "max_tokens": 32},
    {"label": "short", "messages": [{"role": "user", "content": "I used AI to help me write an email to my boss and the whole thing felt off."}], "max_tokens": 80},
    {"label": "medium", "messages": [{"role": "user", "content": "I use AI to help me manage my team's schedules every day, and last Tuesday it scheduled a meeting on top of my standing one-on-one. Ask the next grounding question."}], "max_tokens": 96},
    {"label": "long", "messages": [{"role": "user", "content": "I realized the AI was deciding what news I saw before I'd even finished my coffee, and I want to understand why that bothered me so much. Ask exactly one next grounding question in protocol."}], "max_tokens": 96},
]


def wait_for_server(base_url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.ok:
                return
        except Exception as exc:  # pragma: no cover - runtime polling
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Server did not become healthy: {last_error}")


def query_memory_used_mb() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[0]
        return int(line)
    except Exception:
        return None


def run_case(base_url: str, payload: dict, system_prompt: str | None) -> dict:
    messages = list(payload["messages"])
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": payload["max_tokens"],
            "temperature": 0.0,
        },
        timeout=180,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    data = response.json()
    message = data["choices"][0]["message"]["content"]
    completion_tokens = data.get("usage", {}).get("completion_tokens", max(1, len(message.split())))
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "label": payload["label"],
        "elapsed": round(elapsed, 2),
        "tokens": completion_tokens,
        "tps": round(tps, 1),
        "response": message,
    }


def benchmark_model(llama_server: Path, model: Path, port: int, out_path: Path, system_prompt: str | None) -> dict:
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            str(llama_server),
            "-m",
            str(model),
            "--jinja",
            "--reasoning",
            "off",
            "-ngl",
            "99",
            "-c",
            "4096",
            "-np",
            "2",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(base_url)
        memory_used = query_memory_used_mb()
        runs = [run_case(base_url, prompt, system_prompt) for prompt in PROMPTS]
        avg_tps = round(sum(run["tps"] for run in runs) / len(runs), 1)
        result = {
            "gguf": str(model),
            "gguf_size_gb": round(model.stat().st_size / 1_000_000_000, 2),
            "vram_used_mb": memory_used,
            "system_prompt": system_prompt,
            "avg_tps": avg_tps,
            "verdict": "PASS" if avg_tps >= 5.0 else "FAIL",
            "runs": runs,
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a GGUF with llama-server on BEAST.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=Path(os.environ.get("LLAMA_SERVER_PATH", "llama-server")),
    )
    parser.add_argument("--system-prompt-file", type=Path)
    args = parser.parse_args()

    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else None
    result = benchmark_model(args.llama_server, args.model, args.port, args.out, system_prompt)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
