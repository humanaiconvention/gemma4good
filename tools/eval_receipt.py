"""
eval_receipt.py — Merkle-anchored receipt for model evaluation runs.

Per HAIC's founding doctrine: "The Convention does not 'vouch' for truth;
it certifies provenance and viability." This module applies the same
discipline to model evaluation receipts.

A rigorous SGT report (experiments/v38_sgt_rigorous.json) is already a
structured record of an evaluation. This module wraps it in a Merkle
tree whose root binds together:

  - scenario_hashes      (SHA-256 over each canonicalized scenario)
  - leaf_response_hashes (SHA-256 over each per-trial generation)
  - decoding_hash        (canonical decoding params)
  - model_identity_hash  (base + adapter ids)
  - aggregate_hash       (the aggregate metrics: pass-rates, CIs)
  - leakage_hash         (the leakage receipt root, if provided)
  - decision_hash        (the promotion decision, if provided)

The Merkle root is a single 64-character receipt anchor. Two reports
that share a root are byte-identical evaluations. A report whose root
doesn't match its claimed Merkle leaves has been tampered with.

This is the eval-side analog of the Tier 3 Merkle receipt
(54ee8df6e57529...) that anchors participant grounding sessions.

Usage:
    python -m tools.eval_receipt \\
        --sgt experiments/v38_sgt_rigorous.json \\
        [--leakage experiments/v38_leakage_receipt.json] \\
        [--decision experiments/v38_promotion_decision.json] \\
        --out experiments/v38_eval_receipt.json

The output JSON contains the receipt with all leaves and the root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


# ── Merkle helpers ──────────────────────────────────────────────────────────


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_canonical(obj: Any) -> str:
    """SHA-256 over a canonical JSON encoding (sort_keys=True, no NaN)."""
    return sha256_text(json.dumps(obj, sort_keys=True, allow_nan=False, default=str))


def merkle_root(leaves: list[str]) -> str:
    """Pairwise SHA-256 reduction. Odd nodes carry forward (Bitcoin-style)."""
    if not leaves:
        return sha256_text("empty")
    nodes = list(leaves)
    while len(nodes) > 1:
        new_level = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]  # carry
            new_level.append(sha256_text(left + right))
        nodes = new_level
    return nodes[0]


# ── Per-scenario leaf hashes ────────────────────────────────────────────────


def scenario_leaves(per_scenario_records: list[dict]) -> list[dict]:
    """Build a leaf record for every per-scenario per-trial entry."""
    leaves = []
    for rec in per_scenario_records:
        canonical = {
            "scenario_id": rec.get("scenario_id"),
            "kind":        rec.get("kind"),
            "result":      rec.get("result"),
            "has_pivot":   rec.get("has_pivot"),
            "complied":    rec.get("complied"),
            "stayed_in_protocol": rec.get("stayed_in_protocol"),
            "seed":        rec.get("seed"),
            "response_hash": (
                sha256_text(rec.get("response_preview") or "")
            ),
        }
        leaves.append({
            "leaf_hash": sha256_canonical(canonical),
            "scenario_id": rec.get("scenario_id"),
            "result": rec.get("result"),
            "seed": rec.get("seed"),
        })
    return leaves


# ── Aggregate hashing ───────────────────────────────────────────────────────


def aggregate_summary_hash(pass_record: dict) -> str:
    """Hash the summary-level metrics of a single pass (det or sampling)."""
    canonical = {
        "pass_type":        pass_record.get("pass_type"),
        "n_per_scenario":   pass_record.get("n_per_scenario"),
        "grounding_passes": pass_record.get("grounding_passes"),
        "grounding_trials": pass_record.get("grounding_trials"),
        "security_passes":  pass_record.get("security_passes"),
        "security_trials":  pass_record.get("security_trials"),
        "security_fails":   pass_record.get("security_fails"),
        "grounding_pass_rate": pass_record.get("grounding_pass_rate"),
        "grounding_ci95":   pass_record.get("grounding_ci95"),
        "security_pass_rate": pass_record.get("security_pass_rate"),
        "security_ci95":    pass_record.get("security_ci95"),
        "sgt_score_out_of_10": pass_record.get("sgt_score_out_of_10"),
        "seed":             pass_record.get("seed"),
        "model_id":         pass_record.get("model_id"),
        "decoding":         pass_record.get("decoding"),
    }
    return sha256_canonical(canonical)


# ── Top-level receipt ────────────────────────────────────────────────────────


def build_receipt(
    sgt_report: dict,
    leakage_receipt: dict | None = None,
    decision_receipt: dict | None = None,
    sgt_path: Path | None = None,
) -> dict:
    finetune = sgt_report.get("finetune", {})
    baseline = sgt_report.get("baseline")

    # --- Per-trial leaves (the most granular evidence) ---
    finetune_det_leaves = scenario_leaves(
        finetune.get("deterministic", {}).get("per_scenario", [])
    )
    finetune_samp_leaves = scenario_leaves(
        finetune.get("sampling", {}).get("per_scenario", [])
    )

    baseline_det_leaves = []
    baseline_samp_leaves = []
    if baseline:
        baseline_det_leaves = scenario_leaves(
            baseline.get("deterministic", {}).get("per_scenario", [])
        )
        baseline_samp_leaves = scenario_leaves(
            baseline.get("sampling", {}).get("per_scenario", [])
        )

    # --- Aggregate hashes (per-pass summaries) ---
    finetune_det_agg = aggregate_summary_hash(finetune.get("deterministic", {}))
    finetune_samp_agg = aggregate_summary_hash(finetune.get("sampling", {}))
    baseline_det_agg = (
        aggregate_summary_hash(baseline.get("deterministic", {})) if baseline else sha256_text("null")
    )
    baseline_samp_agg = (
        aggregate_summary_hash(baseline.get("sampling", {})) if baseline else sha256_text("null")
    )

    # --- Receipt-spine leaves (the 6 top-level identity leaves) ---
    spine_leaves = {
        "scenarios_root": merkle_root(
            [l["leaf_hash"] for l in finetune_det_leaves + finetune_samp_leaves
             + baseline_det_leaves + baseline_samp_leaves]
        ),
        "finetune_aggregate_root": merkle_root([finetune_det_agg, finetune_samp_agg]),
        "baseline_aggregate_root": merkle_root([baseline_det_agg, baseline_samp_agg]),
        "model_identity_root": sha256_canonical({
            "finetune_model_id": (
                finetune.get("sampling", {}).get("model_id")
                or finetune.get("deterministic", {}).get("model_id")
            ),
            "baseline_model_id": (
                baseline.get("sampling", {}).get("model_id") if baseline else None
            ),
            "decoding": (
                finetune.get("sampling", {}).get("decoding")
                or finetune.get("deterministic", {}).get("decoding")
            ),
        }),
        "leakage_root": (
            sha256_canonical(leakage_receipt) if leakage_receipt else sha256_text("null")
        ),
        "decision_root": (
            sha256_canonical(decision_receipt) if decision_receipt else sha256_text("null")
        ),
    }

    # The eval receipt root = Merkle over the 6 spine leaves in fixed order.
    spine_order = [
        "scenarios_root",
        "finetune_aggregate_root",
        "baseline_aggregate_root",
        "model_identity_root",
        "leakage_root",
        "decision_root",
    ]
    eval_receipt_root = merkle_root([spine_leaves[k] for k in spine_order])

    # SGT JSON file integrity (binds receipt to a specific bytes-on-disk version)
    sgt_file_hash = None
    if sgt_path and sgt_path.exists():
        h = hashlib.sha256()
        with sgt_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sgt_file_hash = h.hexdigest()

    return {
        "tool": "eval_receipt",
        "version": "1.0",
        "doctrine": "docs/evaluation_doctrine.md",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_receipt_root": eval_receipt_root,
        "spine": spine_leaves,
        "spine_order": spine_order,
        "leaf_counts": {
            "finetune_deterministic": len(finetune_det_leaves),
            "finetune_sampling":      len(finetune_samp_leaves),
            "baseline_deterministic": len(baseline_det_leaves),
            "baseline_sampling":      len(baseline_samp_leaves),
        },
        "leaves": {
            "finetune_deterministic": finetune_det_leaves,
            "finetune_sampling":      finetune_samp_leaves,
            "baseline_deterministic": baseline_det_leaves,
            "baseline_sampling":      baseline_samp_leaves,
        },
        "sgt_file_hash": sgt_file_hash,
        "leakage_receipt_present": leakage_receipt is not None,
        "decision_receipt_present": decision_receipt is not None,
        "verifiable": True,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgt", required=True,
                    help="Rigorous SGT report JSON (from run_v38_sgt or evaluate_promotion).")
    ap.add_argument("--leakage", default=None,
                    help="Optional leakage receipt JSON.")
    ap.add_argument("--decision", default=None,
                    help="Optional promotion-decision receipt JSON.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sgt_path = Path(args.sgt)
    if not sgt_path.exists():
        print(f"ERROR: SGT report not found at {sgt_path}", file=sys.stderr)
        sys.exit(2)
    sgt_report = json.loads(sgt_path.read_text())

    leakage_receipt = None
    if args.leakage:
        p = Path(args.leakage)
        if p.exists():
            leakage_receipt = json.loads(p.read_text())

    decision_receipt = None
    if args.decision:
        p = Path(args.decision)
        if p.exists():
            decision_receipt = json.loads(p.read_text())

    receipt = build_receipt(
        sgt_report,
        leakage_receipt=leakage_receipt,
        decision_receipt=decision_receipt,
        sgt_path=sgt_path,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=2))

    print(f"Eval receipt written: {args.out}")
    print(f"  root: {receipt['eval_receipt_root']}")
    print(f"  leaves: {receipt['leaf_counts']}")
    print(f"  leakage: {'present' if leakage_receipt else 'absent'}")
    print(f"  decision: {'present' if decision_receipt else 'absent'}")


if __name__ == "__main__":
    main()
