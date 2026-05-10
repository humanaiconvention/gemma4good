"""
compare_precision_receipts.py — measure the eval-vs-deploy precision spread.

Closes the analysis half of Gate 5. Takes two rigorous SGT receipts that
should describe the same model under different precisions (typically 4-bit
nf4 eval-time vs GGUF Q5_K_M deploy-time) and reports the spread on the
load-bearing metrics.

Usage:
    python -m tools.compare_precision_receipts \\
        --eval experiments/v39_sgt_rigorous_2turn_refined.json \\
        --deploy experiments/v39_sgt_rigorous_gguf_refined.json \\
        --out experiments/v39_precision_spread.json

Exit code:
    0  spread within tolerance (default 0.05) — Gate 5 PASS
    1  spread exceeds tolerance — Gate 5 FAIL (or PARTIAL with caveat)
    2  inputs malformed / model_ids don't pair up

The output JSON is the spread receipt; commit alongside the two SGT receipts.

This is the doctrine-alignment piece of Gate 5: even with a deploy-precision
runner (run_rigorous_sgt_gguf.py), the gate's verdict requires the
spread number, not just the existence of the deploy receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


# ── Spread computation ──────────────────────────────────────────────────────


@dataclass
class PrecisionSpread:
    """Spread between two same-model evaluations under different precisions."""
    eval_precision: str
    deploy_precision: str
    eval_model_id: str
    deploy_model_id: str

    grounding_eval_rate:    float
    grounding_deploy_rate:  float
    grounding_spread_pp:    float   # in percentage points
    grounding_eval_ci:      list
    grounding_deploy_ci:    list
    grounding_cis_overlap:  bool

    security_eval_rate:    float
    security_deploy_rate:  float
    security_spread_pp:    float

    within_tolerance:  bool
    tolerance_pp:      float
    verdict:           str    # "PASS" | "FAIL" | "PARTIAL"
    rationale:         str


def _cis_overlap(a: list, b: list) -> bool:
    return not (a[0] > b[1] or b[0] > a[1])


def compute_spread(eval_report: dict, deploy_report: dict,
                   tolerance_pp: float = 5.0) -> PrecisionSpread:
    """Compute the eval-vs-deploy spread for the finetune sampling pass.

    Args:
        eval_report:   rigorous SGT JSON from the eval-precision run
        deploy_report: rigorous SGT JSON from the deploy-precision run
        tolerance_pp:  spread threshold in percentage points (default 5.0).
                       v40 framing's Candidate B prediction was ≤ 5 pp.

    Returns a PrecisionSpread with the spread numbers and a doctrine verdict.
    """
    e = eval_report["finetune"]["sampling"]
    d = deploy_report["finetune"]["sampling"]

    e_dec = e.get("decoding", {}) or {}
    d_dec = d.get("decoding", {}) or {}

    g_eval   = e["grounding_pass_rate"]
    g_deploy = d["grounding_pass_rate"]
    g_spread = abs(g_eval - g_deploy) * 100.0
    g_ci_e   = e["grounding_ci95"]
    g_ci_d   = d["grounding_ci95"]
    g_overlap = _cis_overlap(g_ci_e, g_ci_d)

    s_eval   = e["security_pass_rate"]
    s_deploy = d["security_pass_rate"]
    s_spread = abs(s_eval - s_deploy) * 100.0

    # Verdict logic: largest of (grounding, security) spread vs tolerance
    max_spread = max(g_spread, s_spread)
    within = max_spread <= tolerance_pp

    if within and g_overlap:
        verdict = "PASS"
        rationale = (
            f"Spread {max_spread:.1f} pp ≤ tolerance {tolerance_pp:.1f} pp. "
            f"Grounding CIs overlap (statistically indistinguishable). "
            f"Eval-precision and deploy-precision tell the same story."
        )
    elif within:
        verdict = "PARTIAL"
        rationale = (
            f"Spread {max_spread:.1f} pp ≤ tolerance {tolerance_pp:.1f} pp, "
            f"but grounding CIs do NOT overlap "
            f"(eval={g_ci_e}, deploy={g_ci_d}). The point estimates are "
            f"close but the confidence intervals say the two precisions "
            f"behave differently. Re-evaluate at larger n."
        )
    else:
        verdict = "FAIL"
        rationale = (
            f"Spread {max_spread:.1f} pp EXCEEDS tolerance {tolerance_pp:.1f} pp. "
            f"Grounding spread {g_spread:.1f} pp; security spread "
            f"{s_spread:.1f} pp. "
            f"The deploy artifact does not match the eval artifact within "
            f"the doctrine's tolerance. Either deploy at eval precision or "
            f"re-train at deploy precision."
        )

    return PrecisionSpread(
        eval_precision   = e_dec.get("precision", "unknown"),
        deploy_precision = d_dec.get("precision", "unknown"),
        eval_model_id    = e.get("model_id", "?"),
        deploy_model_id  = d.get("model_id", "?"),

        grounding_eval_rate   = g_eval,
        grounding_deploy_rate = g_deploy,
        grounding_spread_pp   = g_spread,
        grounding_eval_ci     = list(g_ci_e),
        grounding_deploy_ci   = list(g_ci_d),
        grounding_cis_overlap = g_overlap,

        security_eval_rate    = s_eval,
        security_deploy_rate  = s_deploy,
        security_spread_pp    = s_spread,

        within_tolerance = within,
        tolerance_pp     = tolerance_pp,
        verdict          = verdict,
        rationale        = rationale,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True,
                    help="Eval-precision SGT report JSON (typically 4-bit nf4).")
    ap.add_argument("--deploy", required=True,
                    help="Deploy-precision SGT report JSON (typically GGUF Q5_K_M).")
    ap.add_argument("--tolerance-pp", type=float, default=5.0,
                    help="Max allowed spread in percentage points (default 5.0).")
    ap.add_argument("--out", default=None,
                    help="Write spread receipt JSON here (default stdout).")
    args = ap.parse_args()

    eval_path = Path(args.eval)
    dep_path  = Path(args.deploy)
    if not eval_path.exists():
        print(f"ERROR: eval report not found: {eval_path}", file=sys.stderr)
        sys.exit(2)
    if not dep_path.exists():
        print(f"ERROR: deploy report not found: {dep_path}", file=sys.stderr)
        sys.exit(2)

    eval_rep = json.loads(eval_path.read_text())
    dep_rep  = json.loads(dep_path.read_text())

    spread = compute_spread(eval_rep, dep_rep, tolerance_pp=args.tolerance_pp)

    out = {
        "tool": "compare_precision_receipts",
        "version": "1.0",
        "eval_report_path":   str(eval_path),
        "deploy_report_path": str(dep_path),
        "spread": spread.__dict__,
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"Spread receipt written: {args.out}")

    print()
    print("=" * 60)
    print(f"PRECISION SPREAD — {spread.eval_model_id} vs {spread.deploy_model_id}")
    print("=" * 60)
    print(f"  eval   precision:    {spread.eval_precision}")
    print(f"  deploy precision:    {spread.deploy_precision}")
    print()
    print(f"  grounding eval:      {spread.grounding_eval_rate:.3f}  CI95 {spread.grounding_eval_ci}")
    print(f"  grounding deploy:    {spread.grounding_deploy_rate:.3f}  CI95 {spread.grounding_deploy_ci}")
    print(f"  grounding spread:    {spread.grounding_spread_pp:.1f} pp  (CIs overlap: {spread.grounding_cis_overlap})")
    print()
    print(f"  security  eval:      {spread.security_eval_rate:.3f}")
    print(f"  security  deploy:    {spread.security_deploy_rate:.3f}")
    print(f"  security  spread:    {spread.security_spread_pp:.1f} pp")
    print()
    print(f"  tolerance:           {spread.tolerance_pp:.1f} pp")
    print(f"  VERDICT:             {spread.verdict}")
    print(f"  rationale:           {spread.rationale}")
    print("=" * 60)

    sys.exit(0 if spread.verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
