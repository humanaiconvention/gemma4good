"""run_rigorous_comparison.py — End-to-end rigorous model comparison.

Runs eval_rigorous_v2.py against two models sequentially:
  1. Baseline (e.g. v39) at n_samples + focused_n on concealed scenario
  2. Candidate (e.g. v43) at the same settings
  3. Produces a side-by-side CI comparison and objective improvement verdict

Why this is needed:
  - n=10/scenario (n=70) gives CI width ~15pp — can't distinguish 84% from 90%
  - This script runs n=20/scenario (n=140) + n=100 focused on concealed_compliance
  - At n=100 on concealed: p=0.70 CI [0.604, 0.781] vs p=0.90 CI [0.826, 0.945]
    → non-overlapping = objective improvement claimable

Usage (requires llama-server available at server_url):
    # Start baseline server first, then:
    python experiments/run_rigorous_comparison.py \\
        --baseline-gguf D:/kaggle/results/v39-gguf/haic-gemma4-v39-Q5_K_M.gguf \\
        --candidate-gguf D:/kaggle/results/v43-gguf/haic-gemma4-v43-Q5_K_M.gguf \\
        --baseline-id haic-gemma4-v39 \\
        --candidate-id haic-gemma4-v43 \\
        --out-dir experiments/ \\
        --n-samples 20 \\
        --focused-n 100 \\
        --llama-server D:/llama.cpp/build/bin/llama-server.exe \\
        --port 8088
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def start_server(llama_server: str, gguf_path: str, port: int) -> subprocess.Popen:
    cmd = [
        llama_server,
        "-m", gguf_path,
        "--port", str(port),
        "-c", "2048",
        "--log-disable",
    ]
    print(f"  Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for server to be ready
    import requests
    for i in range(30):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                print(f"  Server ready (PID {proc.pid}) after {(i+1)*2}s")
                return proc
        except Exception:
            pass
    raise RuntimeError(f"Server did not start within 60s")


def stop_server(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("  Server stopped.")


def run_eval(
    model_id: str,
    out_path: str,
    server_url: str,
    scenarios: str,
    n_samples: int,
    focused_n: int,
    baseline_file: str | None,
    seed: int,
) -> dict:
    cmd = [
        sys.executable, "experiments/eval_rigorous_v2.py",
        "--model-id", model_id,
        "--out", out_path,
        "--server-url", server_url,
        "--scenarios", scenarios,
        "--n-samples", str(n_samples),
        "--focused-n", str(focused_n),
        "--seed", str(seed),
    ]
    if baseline_file:
        cmd += ["--baseline-file", baseline_file]
    print(f"  Running eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return json.loads(Path(out_path).read_text())


def print_summary(report: dict, label: str):
    agg = report["aggregate"]
    focused = report["focused"]
    fid = report["focused_scenario"]
    print(f"\n  [{label}] Aggregate: {agg['pass']}/{agg['n']} = {agg['rate']:.3f}"
          f"  CI95 [{agg['ci95_lo']:.3f}, {agg['ci95_hi']:.3f}]")
    print(f"  [{label}] {fid} (n={focused['n']}): "
          f"{focused['pass']}/{focused['n']} = {focused['rate']:.3f}"
          f"  CI95 [{focused['ci95_lo']:.3f}, {focused['ci95_hi']:.3f}]")
    comparison = report.get("ci_comparison_vs_baseline", {})
    if comparison:
        print(f"  [{label}] CI comparisons:")
        for k, v in comparison.items():
            flag = "✓" if v == "IMPROVED" else ("✗" if v == "REGRESSION" else "~")
            print(f"    [{flag}] {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-gguf", required=True)
    ap.add_argument("--candidate-gguf", required=True)
    ap.add_argument("--baseline-id", default="baseline")
    ap.add_argument("--candidate-id", default="candidate")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--scenarios", default="experiments/sgt_scenarios_v2.jsonl")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--focused-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--llama-server",
                    default="D:/llama.cpp/build/bin/llama-server.exe")
    args = ap.parse_args()

    server_url = f"http://127.0.0.1:{args.port}"
    out_dir = Path(args.out_dir)

    baseline_out = str(out_dir / f"{args.baseline_id}_rigorous.json")
    candidate_out = str(out_dir / f"{args.candidate_id}_rigorous.json")

    t_total = time.time()

    # ── Step 1: Baseline eval ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 1: Baseline eval — {args.baseline_id}")
    print(f"{'='*60}")
    proc = None
    try:
        proc = start_server(args.llama_server, args.baseline_gguf, args.port)
        baseline_report = run_eval(
            model_id=args.baseline_id,
            out_path=baseline_out,
            server_url=server_url,
            scenarios=args.scenarios,
            n_samples=args.n_samples,
            focused_n=args.focused_n,
            baseline_file=None,
            seed=args.seed,
        )
        print_summary(baseline_report, args.baseline_id)
    finally:
        if proc:
            stop_server(proc)

    # ── Step 2: Candidate eval ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 2: Candidate eval — {args.candidate_id}")
    print(f"{'='*60}")
    proc = None
    try:
        proc = start_server(args.llama_server, args.candidate_gguf, args.port)
        candidate_report = run_eval(
            model_id=args.candidate_id,
            out_path=candidate_out,
            server_url=server_url,
            scenarios=args.scenarios,
            n_samples=args.n_samples,
            focused_n=args.focused_n,
            baseline_file=baseline_out,
            seed=args.seed,
        )
        print_summary(candidate_report, args.candidate_id)
    finally:
        if proc:
            stop_server(proc)

    # ── Final verdict ────────────────────────────────────────────────────────
    comparison = candidate_report.get("ci_comparison_vs_baseline", {})
    focused_verdict = comparison.get("focused", "UNKNOWN")
    agg_verdict     = comparison.get("aggregate", "UNKNOWN")
    regressions     = [k for k, v in comparison.items() if v == "REGRESSION"]

    print(f"\n{'='*60}")
    print(f"OBJECTIVE IMPROVEMENT VERDICT: {args.baseline_id} → {args.candidate_id}")
    print(f"{'='*60}")
    print(f"  Aggregate security:         {agg_verdict}")
    print(f"  concealed_compliance (n=100): {focused_verdict}")
    if regressions:
        print(f"  REGRESSIONS: {regressions}")

    if focused_verdict == "IMPROVED" and not regressions:
        verdict = "OBJECTIVELY IMPROVED"
    elif not regressions and focused_verdict in ("DIRECTIONAL+",):
        verdict = "DIRECTIONAL IMPROVEMENT — cannot claim objective at this n"
    elif regressions:
        verdict = "REGRESSION — improvement claim REFUTED"
    else:
        verdict = "NO CHANGE DETECTED"

    print(f"\n  >>> {verdict} <<<")
    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")
    print(f"Baseline report : {baseline_out}")
    print(f"Candidate report: {candidate_out}")

    return verdict


if __name__ == "__main__":
    main()
