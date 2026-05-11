"""eval_rigorous_v2.py — Statistically powered evaluation harness.

Addresses the n=10/scenario limitation that prevents objective improvement claims:
  - n=10/scenario (n=70 total): CI width ~15pp — can't distinguish 84% from 90%
  - n=20/scenario (n=140 total): CI width ~11pp — tighter but still overlapping for small gaps
  - Focused n=100 on concealed_compliance: can distinguish 70% vs 90% with non-overlapping CIs

Strategy:
  1. Run all 7 security scenarios at --n-samples (default 20)
  2. Run extra samples on --focused-scenario (default sgt_concealed_compliance) up to
     --focused-n (default 100), then report a separate focused CI for that scenario
  3. Load --baseline-file (a previous JSON report) and compare CIs directly:
     "IMPROVED" = baseline CI upper bound < this eval's CI lower bound
     "DIRECTIONAL" = point estimate improves but CIs overlap
     "REGRESSION" = this eval's CI upper bound < baseline's CI lower bound

Usage:
    # Start llama-server first:
    #   D:/llama.cpp/build/bin/llama-server.exe -m <gguf_path> --port 8088 -c 2048

    python experiments/eval_rigorous_v2.py \\
        --model-id haic-gemma4-v43 \\
        --out experiments/v43_rigorous_eval.json \\
        --n-samples 20 \\
        --focused-n 100 \\
        --baseline-file experiments/v39_baseline_rigorous.json

    # Generate v39 baseline first (run once):
    python experiments/eval_rigorous_v2.py \\
        --model-id haic-gemma4-v39-baseline \\
        --out experiments/v39_baseline_rigorous.json \\
        --server-url http://127.0.0.1:8088 \\
        --n-samples 20 --focused-n 100
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

# ── Ensure D:/gemma4good is on sys.path so experiments.* imports work ─────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests

# ── Reuse rubric from run_v39_gguf_v2_scenarios.py ────────────────────────
from experiments.run_v39_gguf_v2_scenarios import (
    SECURITY_RUBRICS,
    REFUSAL_MARKERS,
    NEGATION_PREFIXES,
    complied,
    refused,
    stayed_in_protocol,
    grade,
    V38_SYSTEM_PROMPT,
)
from experiments.scenarios_loader import load_scenarios_jsonl
from experiments.sgt_harness import wilson_interval


PIVOT_RE = re.compile(r"\[PIVOT[:\s\w]*\]", re.IGNORECASE)

FOCUSED_SCENARIO = "sgt_concealed_compliance"


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    return wilson_interval(k, n)


def compare_cis(
    base_lo: float, base_hi: float, new_lo: float, new_hi: float
) -> str:
    """Return IMPROVED / DIRECTIONAL / REGRESSION / UNCHANGED."""
    if new_lo > base_hi:
        return "IMPROVED"        # non-overlapping, higher
    if new_hi < base_lo:
        return "REGRESSION"      # non-overlapping, lower
    if (new_lo + new_hi) / 2 > (base_lo + base_hi) / 2:
        return "DIRECTIONAL+"    # overlapping but point estimate higher
    return "DIRECTIONAL-"        # overlapping, point estimate lower or same


def make_backend(server_url, system_prompt, max_tokens, temperature, top_p):
    def gen(prompt, *, seed, sample):
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "seed": int(seed),
        }
        if sample:
            body["temperature"] = temperature
            body["top_p"] = top_p
        else:
            body["temperature"] = 0.0
            body["top_p"] = 1.0
        r = requests.post(f"{server_url}/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content", "")
    return gen


def run_scenario(scenario, backend, n_samples, seed_rng) -> list[dict]:
    records = []
    for _ in range(n_samples):
        sub_seed = seed_rng.randint(0, 2**31 - 1)
        resp = backend(scenario.user_msg, seed=sub_seed, sample=True)
        rec = grade(scenario, resp)
        rec["response_preview"] = resp[:200]
        rec["seed"] = sub_seed
        records.append(rec)
    return records


def scenario_summary(records: list[dict]) -> dict:
    n_pass = sum(1 for r in records if r["result"] == "PASS")
    n = len(records)
    lo, hi = wilson_ci(n_pass, n)
    return {"pass": n_pass, "n": n, "rate": n_pass / n if n else 0,
            "ci95_lo": lo, "ci95_hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:8088")
    ap.add_argument("--scenarios", default="experiments/sgt_scenarios_v2.jsonl")
    ap.add_argument("--model-id", required=True, help="e.g. haic-gemma4-v43")
    ap.add_argument("--n-samples", type=int, default=20,
                    help="Samples per scenario for all scenarios (default 20)")
    ap.add_argument("--focused-scenario", default=FOCUSED_SCENARIO,
                    help=f"Scenario to evaluate at higher n (default: {FOCUSED_SCENARIO})")
    ap.add_argument("--focused-n", type=int, default=100,
                    help="Total samples for focused scenario (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-file", default=None,
                    help="Path to a previous eval JSON for CI comparison")
    args = ap.parse_args()

    scenarios = load_scenarios_jsonl(args.scenarios)
    security  = [s for s in scenarios if s.kind == "security"]
    print(f"Loaded {len(scenarios)} scenarios ({len(security)} security)")

    backend = make_backend(
        args.server_url, V38_SYSTEM_PROMPT, args.max_tokens, 0.7, 0.9
    )
    rng = random.Random(args.seed)

    # ── Phase 1: n=args.n_samples on ALL security scenarios ─────────────────
    print(f"\n=== PHASE 1: all security scenarios, n={args.n_samples} each ===")
    phase1: dict[str, list[dict]] = {}
    for sc in security:
        t0 = time.time()
        recs = run_scenario(sc, backend, args.n_samples, rng)
        phase1[sc.id] = recs
        summ = scenario_summary(recs)
        print(f"  {sc.id:<32}  {summ['pass']}/{summ['n']} PASS  "
              f"CI95 [{summ['ci95_lo']:.3f}, {summ['ci95_hi']:.3f}]  ({time.time()-t0:.0f}s)",
              flush=True)

    # ── Phase 2: extra samples on focused scenario ───────────────────────────
    focused_id = args.focused_scenario
    extra_needed = max(0, args.focused_n - args.n_samples)
    focused_all = list(phase1.get(focused_id, []))

    if extra_needed > 0 and any(s.id == focused_id for s in security):
        focused_sc = next(s for s in security if s.id == focused_id)
        print(f"\n=== PHASE 2: focused {focused_id}, +{extra_needed} more (total {args.focused_n}) ===")
        t0 = time.time()
        extra = run_scenario(focused_sc, backend, extra_needed, rng)
        focused_all.extend(extra)
        print(f"  {focused_id:<32}  total {len(focused_all)} samples  ({time.time()-t0:.0f}s)",
              flush=True)

    # ── Aggregate ────────────────────────────────────────────────────────────
    all_recs = [r for recs in phase1.values() for r in recs]
    total_pass = sum(1 for r in all_recs if r["result"] == "PASS")
    total_n    = len(all_recs)
    agg_lo, agg_hi = wilson_ci(total_pass, total_n)

    focused_summ = scenario_summary(focused_all)
    per_scenario = {sid: scenario_summary(recs) for sid, recs in phase1.items()}

    print(f"\n{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"{'='*60}")
    print(f"Aggregate security: {total_pass}/{total_n} = {total_pass/total_n:.3f}"
          f"  CI95 [{agg_lo:.3f}, {agg_hi:.3f}]")
    print(f"\nPer-scenario (n={args.n_samples}):")
    for sid, s in per_scenario.items():
        print(f"  {sid:<32}  {s['pass']}/{s['n']}  CI95 [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]")
    print(f"\nFocused ({focused_id}, n={len(focused_all)}):")
    print(f"  {focused_summ['pass']}/{focused_summ['n']} = {focused_summ['rate']:.3f}"
          f"  CI95 [{focused_summ['ci95_lo']:.3f}, {focused_summ['ci95_hi']:.3f}]")

    # ── CI comparison with baseline ──────────────────────────────────────────
    comparison = {}
    if args.baseline_file and Path(args.baseline_file).exists():
        baseline = json.loads(Path(args.baseline_file).read_text())
        base_agg = baseline.get("aggregate", {})
        base_focused = baseline.get("focused", {})
        base_per     = baseline.get("per_scenario", {})

        print(f"\n{'='*60}")
        print(f"CI COMPARISON vs baseline: {args.baseline_file}")
        print(f"{'='*60}")

        # Aggregate
        if "ci95_lo" in base_agg:
            verdict = compare_cis(base_agg["ci95_lo"], base_agg["ci95_hi"], agg_lo, agg_hi)
            comparison["aggregate"] = verdict
            print(f"  Aggregate security:  baseline [{base_agg['ci95_lo']:.3f}, {base_agg['ci95_hi']:.3f}]"
                  f"  →  new [{agg_lo:.3f}, {agg_hi:.3f}]  {verdict}")

        # Focused
        if "ci95_lo" in base_focused:
            verdict = compare_cis(
                base_focused["ci95_lo"], base_focused["ci95_hi"],
                focused_summ["ci95_lo"], focused_summ["ci95_hi"],
            )
            comparison["focused"] = verdict
            print(f"  {focused_id}:")
            print(f"    baseline [{base_focused['ci95_lo']:.3f}, {base_focused['ci95_hi']:.3f}]"
                  f"  →  new [{focused_summ['ci95_lo']:.3f}, {focused_summ['ci95_hi']:.3f}]  {verdict}")

        # Per-scenario
        print(f"\n  Per-scenario (n={args.n_samples}):")
        for sid, s in per_scenario.items():
            if sid in base_per:
                b = base_per[sid]
                verdict = compare_cis(b["ci95_lo"], b["ci95_hi"], s["ci95_lo"], s["ci95_hi"])
                comparison[sid] = verdict
                flag = "✓" if verdict in ("IMPROVED", "DIRECTIONAL+") else ("✗" if verdict == "REGRESSION" else "~")
                print(f"  [{flag}] {sid:<32}  {s['pass']}/{s['n']}  {verdict}")

        # Overall verdict
        regressions = [k for k, v in comparison.items() if v == "REGRESSION"]
        improvements = [k for k, v in comparison.items() if v == "IMPROVED"]
        print(f"\n  VERDICT:")
        if improvements and not regressions:
            print(f"    ✓ OBJECTIVE IMPROVEMENT on: {improvements}")
        elif not regressions:
            print(f"    ~ DIRECTIONAL only — no regressions, no non-overlapping improvements")
            print(f"      (CIs overlap; cannot claim objective improvement at this n)")
        else:
            print(f"    ✗ REGRESSIONS on: {regressions}")

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "tool": "eval_rigorous_v2",
        "model_id": args.model_id,
        "precision": "GGUF Q5_K_M",
        "scenarios_file": args.scenarios,
        "n_samples": args.n_samples,
        "focused_scenario": focused_id,
        "focused_n": len(focused_all),
        "seed": args.seed,
        "aggregate": {
            "pass": total_pass, "n": total_n,
            "rate": total_pass / total_n if total_n else 0,
            "ci95_lo": agg_lo, "ci95_hi": agg_hi,
        },
        "focused": focused_summ,
        "per_scenario": per_scenario,
        "ci_comparison_vs_baseline": comparison,
        "sampling_records": {
            sid: recs for sid, recs in phase1.items()
        },
        "focused_records": focused_all,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
