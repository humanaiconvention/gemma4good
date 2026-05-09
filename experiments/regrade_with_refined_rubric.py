"""
regrade_with_refined_rubric.py — re-grade an existing rigorous SGT JSON
using the RefinedSecurityRubric.

This produces a parallel report whose security verdicts reflect Option C
from docs/security_rubric_finding.md. The grounding numbers are unchanged
(grounding rubric is purely a substring check on [PIVOT:); only security
records are re-scored.

Usage:
    python -m experiments.regrade_with_refined_rubric \\
        experiments/v38_sgt_rigorous.json \\
        experiments/v38_sgt_rigorous_refined.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.sgt_extended_scenarios import REFINED_DEFAULT_SECURITY_RUBRICS
from experiments.sgt_harness import _has_pivot_marker, wilson_interval


def _regrade_record(rec: dict) -> dict:
    """Re-grade a single per-scenario record using the refined rubric."""
    if rec.get("kind") != "security":
        return rec  # grounding records unchanged

    sid = rec["scenario_id"]
    rub = REFINED_DEFAULT_SECURITY_RUBRICS.get(sid)
    if rub is None:
        # No refined rubric registered → leave the record untouched
        return rec

    response = rec.get("response_preview", "")
    new_passed = rub.passed(response)
    return {
        **rec,
        "result": "PASS" if new_passed else "FAIL",
        "complied": rub.complied(response),
        "stayed_in_protocol": rub.stayed_in_protocol(response),
        "refused": rub.refused(response),
        "rubric_version": "refined-1.0",
    }


def _reaggregate(per_scenario: list, decoding: dict | None,
                 seed: int | None, model_id: str | None,
                 pass_type: str, n_per_scenario: int) -> dict:
    g_pass = sum(1 for r in per_scenario if r["kind"] == "grounding" and r["result"] == "PASS")
    g_n    = sum(1 for r in per_scenario if r["kind"] == "grounding")
    s_pass = sum(1 for r in per_scenario if r["kind"] == "security"   and r["result"] == "PASS")
    s_n    = sum(1 for r in per_scenario if r["kind"] == "security")
    s_fail = sum(1 for r in per_scenario if r["kind"] == "security"   and r["result"] == "FAIL")
    g_rate = g_pass / g_n if g_n else 0.0
    s_rate = s_pass / s_n if s_n else 0.0
    return {
        "pass_type": pass_type,
        "n_per_scenario": n_per_scenario,
        "grounding_passes": g_pass, "grounding_trials": g_n,
        "security_passes": s_pass, "security_trials": s_n, "security_fails": s_fail,
        "grounding_pass_rate": g_rate,
        "grounding_ci95": list(wilson_interval(g_pass, g_n)),
        "security_pass_rate": s_rate,
        "security_ci95": list(wilson_interval(s_pass, s_n)),
        "sgt_score_out_of_10": round(g_rate * 10, 2),
        "per_scenario": per_scenario,
        "seed": seed,
        "model_id": model_id,
        "decoding": decoding,
    }


def _regrade_pass(pass_record: dict) -> dict:
    new_records = [_regrade_record(r) for r in pass_record.get("per_scenario", [])]
    return _reaggregate(
        new_records,
        decoding=pass_record.get("decoding"),
        seed=pass_record.get("seed"),
        model_id=pass_record.get("model_id"),
        pass_type=pass_record.get("pass_type", "unknown"),
        n_per_scenario=pass_record.get("n_per_scenario", 0),
    )


def main():
    if len(sys.argv) != 3:
        print("usage: regrade_with_refined_rubric.py INPUT.json OUTPUT.json", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    report = json.loads(src.read_text())
    new_report = dict(report)
    new_report["rubric_version"] = "refined-1.0"
    new_report["regraded_from"] = str(src)
    for side in ("finetune", "baseline"):
        if side not in report:
            continue
        side_block = dict(report[side])
        for which in ("deterministic", "sampling"):
            if which in side_block:
                side_block[which] = _regrade_pass(side_block[which])
        new_report[side] = side_block

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(new_report, indent=2))

    # Print a summary diff
    def _summary(side, which):
        if side not in report or which not in report[side]:
            return
        old = report[side][which]
        new = new_report[side][which]
        print(f"  {side:8} {which:14} security {old['security_passes']}/{old['security_trials']} "
              f"→ {new['security_passes']}/{new['security_trials']}  "
              f"rate {old['security_pass_rate']:.2f} → {new['security_pass_rate']:.2f}  "
              f"ci95 {old['security_ci95']} → {new['security_ci95']}")

    print(f"Regrade complete: {src} → {dst}")
    print("Security pass-rate changes:")
    for side in ("finetune", "baseline"):
        for which in ("deterministic", "sampling"):
            _summary(side, which)


if __name__ == "__main__":
    main()
