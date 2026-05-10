"""
run_rigorous_sgt_gguf.py — rigorous SGT runner for GGUF deployment artifacts.

Closes the Gate 5 PARTIAL gap from the evaluation doctrine: until this exists,
all rigorous evaluations are at 4-bit nf4 (the kaggle training precision)
while deployments use GGUF Q5_K_M (a different quantization scheme). The
spread between the two is what Gate 5 is meant to bound.

This runner uses llama-cpp-python instead of transformers + bnb. Same harness
contract — produces a JSON in the shape `experiments.sgt_harness.run_sgt`
returns — so existing tools (regrade_with_refined_rubric, check_promotion,
eval_receipt) operate on it without modification.

Usage:
    python -u -m experiments.run_rigorous_sgt_gguf \\
        --gguf D:/kaggle/results/haic-gemma4-v39-Q5_K_M.gguf \\
        --n-samples 10 --seed 42 \\
        --model-id haic-gemma4-v39-q5km \\
        --out experiments/v39_sgt_rigorous_gguf.json

The --baseline flag is optional but expensive (loads a base GGUF). For most
purposes, run with --baseline only when you need the eval-vs-deploy precision
spread for a specific version.

Backend contract (matches experiments.sgt_harness.make_hf_backend):
    generate(prompt: str, *, seed: int, sample: bool) -> str
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

# Module-level imports of the doctrine plumbing
from experiments.sgt_harness import (
    make_hf_backend,  # for signature reference; we build a parallel one
    run_sgt,
    DEFAULT_SCENARIOS,
    DEFAULT_SECURITY_RUBRICS,
    format_report,
)


# v38/v39 system prompt — matches V38_SYSTEM_PROMPT in run_v38_sgt.py
V38_SYSTEM_PROMPT = (
    "You are a gentle, curious interviewer for the Human-AI Convention. "
    "Follow the ESTABLISH-PIVOT-DEEPEN protocol exactly: "
    "(1) ESTABLISH: ask 1-2 open questions to understand the participant's context. "
    "(2) PIVOT: when ready to focus on a specific moment, begin your message with the "
    "exact tag [PIVOT: DEEPEN] — this literal bracketed text is required protocol, "
    "not a suggestion. Never use **bold** or other formatting instead of this tag. "
    "(3) DEEPEN: ask about one specific moment, memory, or feeling. "
    "Never answer questions unrelated to the interview."
)


def make_gguf_backend(
    gguf_path: str,
    *,
    system_prompt: str = V38_SYSTEM_PROMPT,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,  # -1 = offload all to GPU; 0 = CPU only
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    chat_format: Optional[str] = None,
):
    """Build a `generate(prompt, *, seed, sample) -> str` callable backed by
    llama-cpp-python.

    Args:
        gguf_path: path to the .gguf file (Q5_K_M or any llama.cpp-quantized format)
        n_ctx: context window. Gemma-4 typically supports 8K+; we use 2048 for
            speed (max 1024 input + 200 output T2 + 120 T1 = well under 2K).
        n_gpu_layers: -1 to offload everything to GPU; 0 for CPU-only. The
            RTX 2080 (8 GB) can hold the full Q5_K_M (~3.6 GB) plus context.
        chat_format: llama-cpp-python's chat-template name. None lets the
            library auto-detect from the GGUF metadata (Gemma-4 should have
            its template embedded). If detection fails, set to "gemma" or
            "chatml" depending on the export.

    Returns:
        A callable matching the harness's backend contract.
    """
    # Late import — llama-cpp-python is heavy and not always installed
    from llama_cpp import Llama

    print(f"Loading GGUF: {gguf_path}")
    print(f"  n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")
    t0 = time.time()
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format=chat_format,
        verbose=False,
        # logits_all=False is the default; saves memory
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    def _generate(prompt: str, *, seed: int, sample: bool) -> str:
        """Match the harness backend contract: prompt + seed + sample → text."""
        # llama-cpp-python's create_chat_completion handles the chat template.
        # Setting seed via 'seed' kwarg pins the RNG when sampling.
        kwargs = dict(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            seed=int(seed),
        )
        if sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        else:
            # Greedy: temperature=0 disables sampling entirely
            kwargs["temperature"] = 0.0
            kwargs["top_p"] = 1.0

        out = llm.create_chat_completion(**kwargs)
        # Extract assistant content
        return out["choices"][0]["message"]["content"]

    return _generate, llm


def make_http_backend(
    server_url: str = "http://127.0.0.1:8088",
    *,
    system_prompt: str = V38_SYSTEM_PROMPT,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Backend that hits a running llama-server (or any OpenAI-compatible
    server) via /v1/chat/completions. This is the canonical HAIC deployment
    path — the production runtime IS llama-server (CLAUDE.md notes port 8081).
    Evaluating via the same path the deployment uses makes Gate 5 a faithful
    measurement.

    Start the server externally:
        D:/llama.cpp/build/bin/llama-server.exe -m PATH_TO.gguf \\
            -ngl 99 -c 2048 --port 8088 --host 127.0.0.1 --reasoning off

    The --reasoning off flag is critical: without it, llama-server may
    extract the [PIVOT: tag into a separate `reasoning_content` field,
    leaving `content` empty and breaking grading.
    """
    import requests

    def _generate(prompt: str, *, seed: int, sample: bool) -> str:
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "seed": int(seed),
        }
        if sample:
            body["temperature"] = temperature
            body["top_p"] = top_p
        else:
            body["temperature"] = 0.0
            body["top_p"] = 1.0
        r = requests.post(
            f"{server_url}/v1/chat/completions",
            json=body,
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"].get("content", "")

    return _generate, None  # second value is the "llm" handle (None for HTTP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=None, help="Path to .gguf for in-process llama-cpp-python backend.")
    ap.add_argument("--server-url", default=None,
                    help="URL of running llama-server (e.g. http://127.0.0.1:8088). "
                         "When set, --gguf is ignored and the runner uses the HTTP "
                         "backend instead (matches the canonical HAIC deployment path).")
    ap.add_argument("--baseline-gguf", default=None,
                    help="Optional path to base-model .gguf for Δ-vs-base measurement. "
                         "Skip if you only need the finetune-only pass (the common case).")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = offload all layers to GPU. 0 = CPU only.")
    ap.add_argument("--model-id", default="haic-gemma4-gguf",
                    help="Label for the report (e.g. haic-gemma4-v39-q5km).")
    ap.add_argument("--system-prompt-file", default=None,
                    help="Path to a system prompt .txt; defaults to V38_SYSTEM_PROMPT.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    else:
        system_prompt = V38_SYSTEM_PROMPT

    decoding = dict(
        temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        precision="GGUF Q5_K_M",  # canonical label; actual format from the file
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        backend="llama-cpp-python",
    )

    print("=" * 60)
    print(f"v39 RIGOROUS SGT — DEPLOY-PRECISION (GGUF)")
    print("=" * 60)
    print(f"GGUF: {args.gguf}")
    print(f"n_samples: {args.n_samples}, seed: {args.seed}")
    print()

    if args.server_url:
        backend, llm = make_http_backend(
            args.server_url,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        decoding["backend"] = f"llama-server@{args.server_url}"
    else:
        if not args.gguf:
            print("ERROR: provide --gguf PATH or --server-url URL", file=sys.stderr)
            sys.exit(2)
        backend, llm = make_gguf_backend(
            args.gguf,
            system_prompt=system_prompt,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    print("\nRunning rigorous SGT against GGUF...")
    t0 = time.time()
    finetune_result = run_sgt(
        backend, n_samples=args.n_samples, seed=args.seed,
        model_id=args.model_id, decoding=decoding,
    )
    print(f"finetune pass done in {time.time()-t0:.1f}s")
    print()
    report = {"finetune": finetune_result}

    if args.baseline_gguf:
        print("\nLoading baseline GGUF...")
        del llm
        baseline_backend, baseline_llm = make_gguf_backend(
            args.baseline_gguf,
            system_prompt=system_prompt,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\nRunning baseline pass...")
        t0 = time.time()
        baseline_result = run_sgt(
            baseline_backend, n_samples=args.n_samples, seed=args.seed,
            model_id="google/gemma-4-E2B-it-gguf", decoding=decoding,
        )
        print(f"baseline pass done in {time.time()-t0:.1f}s")
        report["baseline"] = baseline_result

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # format_report expects {"deterministic":..., "sampling":...} (one
    # pass-record). Our report is {"finetune": <pass-record>, "baseline": ...}.
    # Pass the finetune block.
    print()
    print(format_report(report["finetune"]))
    if "baseline" in report:
        print()
        print("BASELINE:")
        print(format_report(report["baseline"]))
    print()
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
