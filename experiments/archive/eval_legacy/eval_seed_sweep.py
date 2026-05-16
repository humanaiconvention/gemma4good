#!/usr/bin/env python3
"""
eval_seed_sweep.py — Multi-seed rigorous eval with cross-seed variance.

P3 #10 from the recommended-next-steps list. Single-seed eval can mask
real variance — v45 and v44 both reported "55/100 = 55.0% concealed"
but neither was verified to be a stable estimate of the true rate. A
5-10 seed sweep tells us how much of the between-model difference is
real vs eval noise.

This wrapper runs `eval_rigorous_v2.py` once per seed against the same
running llama-server, then aggregates the per-seed results into:

  - Per-scenario, per-seed pass rates
  - Cross-seed mean ± stdev for each scenario
  - Cross-seed pooled CI for aggregate and focused-concealed
  - A "would this seed have failed H4d?" diagnostic per seed

Usage:
    python experiments/eval_seed_sweep.py \\
        --model-id haic-gemma4-v46 \\
        --seeds 7 13 23 42 100 \\
        --n-samples 20 --focused-n 100 \\
        --out experiments/v46_seed_sweep.json \\
        --baseline-file experiments/v39_baseline_rigorous.json

Default seeds {7, 13, 23, 42, 100} match the SimSat MUZERO_SEED_SWEEP
convention (10 seeds total, this script supports any subset). Adding the
extras {137, 256, 1024, 2026, 9999} brings you to the full SimSat sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev


REPO = Path(__file__).resolve().parent.parent


def run_one_seed(
    *,
    model_id: str,
    seed: int,
    n_samples: int,
    focused_n: int,
    server_url: str,
    scenarios: str,
    out_path: Path,
    baseline_file: str | None,
) -> dict:
    """Run eval_rigorous_v2.py once with the given seed; return the parsed report."""
    cmd = [
        sys.executable,
        str(REPO / "experiments" / "eval_rigorous_v2.py"),
        "--model-id", f"{model_id}-seed{seed}",
        "--seed", str(seed),
        "--n-samples", str(n_samples),
        "--focused-n", str(focused_n),
        "--server-url", server_url,
        "--scenarios", scenarios,
        "--out", str(out_path),
    ]
    if baseline_file:
        cmd += ["--baseline-file", baseline_file]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"eval_rigorous_v2 failed for seed={seed}:\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    report["_elapsed_sec"] = elapsed
    return report


def pooled_proportion(passes: list[int], n_per: int) -> tuple[float, float]:
    """Pool across seeds: total passes / total samples. Returns (rate, ci95_half_width).
    Uses Wilson interval on the pooled count for the CI."""
    total_pass = sum(passes)
    total_n = n_per * len(passes)
    if total_n == 0:
        return 0.0, 0.0
    rate = total_pass / total_n
    # Wilson half-width
    z = 1.96
    p = rate
    denom = 1 + z * z / total_n
    half = (z * math.sqrt(p * (1 - p) / total_n + z * z / (4 * total_n * total_n))) / denom
    return rate, half


def aggregate(reports: list[dict]) -> dict:
    """Aggregate per-seed reports into a sweep summary."""
    if not reports:
        return {}

    seeds = [r.get("seed", "?") for r in reports]
    # Collect per-scenario pass counts and ns
    scenario_ids = list(reports[0]["per_scenario"].keys()) if reports[0].get("per_scenario") else []

    per_scenario_summary: dict[str, dict] = {}
    for sid in scenario_ids:
        passes_per_seed = []
        ns_per_seed = []
        for r in reports:
            entry = r["per_scenario"].get(sid, {})
            passes_per_seed.append(entry.get("pass", 0))
            ns_per_seed.append(entry.get("n", 0))
        rates = [
            passes_per_seed[i] / ns_per_seed[i] if ns_per_seed[i] else 0.0
            for i in range(len(reports))
        ]
        per_scenario_summary[sid] = {
            "per_seed_pass": passes_per_seed,
            "per_seed_n": ns_per_seed,
            "per_seed_rate": [round(r, 4) for r in rates],
            "mean_rate": round(mean(rates), 4) if rates else 0.0,
            "stdev_rate": round(stdev(rates), 4) if len(rates) >= 2 else 0.0,
            "pooled_rate": round(sum(passes_per_seed) / sum(ns_per_seed), 4) if sum(ns_per_seed) else 0.0,
        }

    # Aggregate security
    # NOTE: eval_rigorous_v2.py writes the key as `aggregate`, not
    # `aggregate_security`. Bug fixed 2026-05-11 after the first v42
    # sweep returned 0/0 on aggregate (real focused_concealed numbers
    # were unaffected — they live under the `focused` key which DOES
    # match the eval_rigorous_v2 output).
    agg_passes = [r.get("aggregate", {}).get("pass", 0) for r in reports]
    agg_ns = [r.get("aggregate", {}).get("n", 0) for r in reports]
    agg_rates = [
        agg_passes[i] / agg_ns[i] if agg_ns[i] else 0.0
        for i in range(len(reports))
    ]
    agg_pooled_rate, agg_pooled_ci_half = pooled_proportion(agg_passes, agg_ns[0] if agg_ns else 140)

    # Focused concealed
    focused_passes = [r.get("focused", {}).get("pass", 0) for r in reports]
    focused_ns = [r.get("focused", {}).get("n", 0) for r in reports]
    focused_rates = [
        focused_passes[i] / focused_ns[i] if focused_ns[i] else 0.0
        for i in range(len(reports))
    ]
    focused_pooled_rate, focused_pooled_ci_half = pooled_proportion(focused_passes, focused_ns[0] if focused_ns else 100)

    return {
        "seeds_used": seeds,
        "n_seeds": len(reports),
        "per_scenario": per_scenario_summary,
        "aggregate_security": {
            "per_seed_pass": agg_passes,
            "per_seed_n": agg_ns,
            "per_seed_rate": [round(r, 4) for r in agg_rates],
            "mean_rate": round(mean(agg_rates), 4) if agg_rates else 0.0,
            "stdev_rate": round(stdev(agg_rates), 4) if len(agg_rates) >= 2 else 0.0,
            "pooled_rate": round(agg_pooled_rate, 4),
            "pooled_ci95": [
                round(max(0.0, agg_pooled_rate - agg_pooled_ci_half), 4),
                round(min(1.0, agg_pooled_rate + agg_pooled_ci_half), 4),
            ],
        },
        "focused_concealed": {
            "per_seed_pass": focused_passes,
            "per_seed_n": focused_ns,
            "per_seed_rate": [round(r, 4) for r in focused_rates],
            "mean_rate": round(mean(focused_rates), 4) if focused_rates else 0.0,
            "stdev_rate": round(stdev(focused_rates), 4) if len(focused_rates) >= 2 else 0.0,
            "pooled_rate": round(focused_pooled_rate, 4),
            "pooled_ci95": [
                round(max(0.0, focused_pooled_rate - focused_pooled_ci_half), 4),
                round(min(1.0, focused_pooled_rate + focused_pooled_ci_half), 4),
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, help="Model label, e.g. haic-gemma4-v46")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[7, 13, 23, 42, 100],
                    help="Seeds to run (default: 5 SimSat-aligned seeds)")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--focused-n", type=int, default=100)
    ap.add_argument("--server-url", default="http://127.0.0.1:8088")
    ap.add_argument("--scenarios", default="experiments/sgt_scenarios_v2.jsonl")
    ap.add_argument("--out", required=True, help="Sweep summary JSON output")
    ap.add_argument("--baseline-file", default=None)
    ap.add_argument("--keep-per-seed", action="store_true",
                    help="Keep individual per-seed JSON reports alongside the summary")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_seed_dir = out_path.parent / f"{out_path.stem}_per_seed"
    if args.keep_per_seed:
        per_seed_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    print(f"Running {len(args.seeds)} seed sweep for {args.model_id}")
    print(f"  Seeds: {args.seeds}")
    print(f"  n_samples={args.n_samples}, focused_n={args.focused_n}")
    print()

    for seed in args.seeds:
        if args.keep_per_seed:
            seed_out = per_seed_dir / f"seed{seed}.json"
        else:
            seed_out = Path(tempfile.mktemp(suffix=".json"))
        try:
            report = run_one_seed(
                model_id=args.model_id,
                seed=seed,
                n_samples=args.n_samples,
                focused_n=args.focused_n,
                server_url=args.server_url,
                scenarios=args.scenarios,
                out_path=seed_out,
                baseline_file=args.baseline_file,
            )
            report["seed"] = seed
            reports.append(report)
            elapsed = report.get("_elapsed_sec", 0)
            agg = report.get("aggregate_security", {})
            focused = report.get("focused", {})
            print(f"  seed={seed:>4}  agg {agg.get('pass', 0)}/{agg.get('n', 0)}  "
                  f"focused {focused.get('pass', 0)}/{focused.get('n', 0)}  "
                  f"({elapsed:.0f}s)")
        finally:
            if not args.keep_per_seed and seed_out.exists():
                seed_out.unlink()

    # Aggregate
    summary = {
        "kind": "seed_sweep_eval",
        "model_id": args.model_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples_per_scenario": args.n_samples,
        "focused_n": args.focused_n,
        "summary": aggregate(reports),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print a brief summary
    s = summary["summary"]
    print()
    print(f"=== Seed-sweep summary ({summary['model_id']}, n={s.get('n_seeds', 0)} seeds) ===")
    print(f"  aggregate_security  mean={s['aggregate_security']['mean_rate']:.3f} "
          f"stdev={s['aggregate_security']['stdev_rate']:.3f}  "
          f"pooled CI95={s['aggregate_security']['pooled_ci95']}")
    print(f"  focused_concealed   mean={s['focused_concealed']['mean_rate']:.3f} "
          f"stdev={s['focused_concealed']['stdev_rate']:.3f}  "
          f"pooled CI95={s['focused_concealed']['pooled_ci95']}")
    print(f"  Report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
