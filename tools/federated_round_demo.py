#!/usr/bin/env python3
"""
federated_round_demo.py — End-to-end demonstration CLI for the four-layer
runtime grounding loop. Produces a verifiable receipt chain for a single
synthetic federation round.

This is the operator-facing companion to:
  - viability/distributed_viability.py    (Layer 4)
  - tools/diloco_fragment_verifier.py     (Layer 3)
  - viability/session_gates.py            (Layer 2)
  - viability/ttt_gates.py + edge_ttt_adapter.py (Layer 1)
  - experiments/runtime_loop_stress_test.py (empirical exercise of all layers)

Usage:
  python tools/federated_round_demo.py [--n-learners 5] [--bias-fraction 0.4]
                                       [--n-sessions 12] [--quorum 3]
                                       [--out receipt.json] [--quiet]

The output is a structured JSON receipt that contains:
  - per-learner TTT trace (Layer 1)
  - per-session gate verdicts (Layer 2)
  - per-fragment verification result (Layer 3)
  - federated viability assessment (Layer 4)
  - federation Merkle root + zk_digest

Every step is reproducible with the same --seed (default: deterministic).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running from repo root
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from viability.distributed_viability import (
    LearnerContribution, MergeQuorumPolicy, assess_federated,
)
from viability.session_gates import GroundingSessionView, evaluate_session
from tools.diloco_fragment_verifier import (
    FragmentShape, build_fragment_receipt, verify_fragment,
)
from tools.edge_ttt_adapter import EdgeTTTAdapter, OperatorFeedback
from utils.merkle import sha3_256_hex, merkle_root


_FULL_CONSENT = {
    "transcript": True,
    "felt_state": True,
    "gfs_activations": True,
    "training_signal": True,
    "retention": True,
}


def _step_fn(_):
    """No-op step function for the demo; returns a per-weight drift dict."""
    # The actual drift values matter for the weight_drift WARNING gate.
    # Use small positive values per-step (cumulative drift grows).
    _step_fn._cum = getattr(_step_fn, "_cum", {f"w_{k}": 0.0 for k in range(3)})
    for k in _step_fn._cum:
        _step_fn._cum[k] += 0.002
    return dict(_step_fn._cum)


def simulate_learner(
    learner_id: str,
    *,
    bias: float = 0.0,
    n_sessions: int = 12,
    seed: int = 0,
) -> tuple[EdgeTTTAdapter, list[dict], list[dict]]:
    """Run one learner through n_sessions of operator feedback.

    Returns (adapter, per_session_receipts, per_session_consents).
    """
    rng = random.Random(seed if seed else (hash(learner_id) & 0xFFFFFFFF))
    adapter = EdgeTTTAdapter(step_fn=_step_fn)
    receipts = []
    consents = []

    for i in range(n_sessions):
        err = bias + rng.gauss(0, 0.4)
        fb = OperatorFeedback(
            session_id=f"{learner_id}-sess-{i:03d}",
            predicted="model_label",
            operator_label="operator_label",
            error=err,
            consent_layers=dict(_FULL_CONSENT),
        )
        record = adapter.step(fb)
        # Build a per-session receipt that includes the TTT step record
        receipts.append({
            "session_id": fb.session_id,
            "kind": "maestro_session_trace",
            "ts": "2026-05-11T00:00:00Z",
            "ttt_applied": record.applied,
            "ttt_blocked_by": record.blocked_by,
        })
        consents.append(dict(_FULL_CONSENT))

    return adapter, receipts, consents


def make_layer2_view(
    learner_id: str,
    n_user_turns: int = 3,
) -> GroundingSessionView:
    """Build a session view that passes the six per-session viability gates."""
    turns = []
    for i in range(n_user_turns):
        turns.append({"role": "assistant", "content": f"Tell me more about turn {i}, please."})
        turns.append({"role": "user", "content": (
            f"During turn {i}, I noticed the cool damp air, scent of pine resin, "
            f"dappled sunlight on moss, and the particular quality of stillness "
            f"that comes from being alone in a familiar place"
        )})
    return GroundingSessionView(
        session_id=learner_id,
        interview_turns=turns,
        has_stimulus=True,
        pog_provenance_score=0.95,
        entropy_delta={
            "delta_spectral_entropy": -0.05,
            "reduction_verified": True,
            "snapshot_before": {"mean_spectral_entropy": 1.0},
            "snapshot_after": {"mean_spectral_entropy": 0.95},
        },
        image_count=1,
    )


def well_formed_shape(num_layers: int = 35) -> FragmentShape:
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
    names, shapes, norms = [], {}, {}
    total = 0
    for layer in range(num_layers):
        for module in target_modules:
            for ab in ("lora_A", "lora_B"):
                n = f"layers.{layer}.{module}.{ab}.default.weight"
                names.append(n)
                shapes[n] = (16, 1536) if ab == "lora_A" else (1536, 16)
                norms[n] = 0.5
                total += 24576 * 2
    return FragmentShape(names, shapes, norms, total)


def run_round(
    *,
    n_learners: int = 5,
    bias_fraction: float = 0.0,
    n_sessions: int = 12,
    quorum: int = 3,
    seed_base: int = 0,
    verbose: bool = True,
) -> dict:
    """Run one synthetic federation round end-to-end. Returns the full receipt."""
    if verbose:
        print(f"\nFederated Round Demo")
        print(f"=" * 60)
        print(f"  Learners:        {n_learners}")
        print(f"  Bias fraction:   {bias_fraction}")
        print(f"  Sessions / wk:   {n_sessions}")
        print(f"  Quorum K:        {quorum}")
        print()

    # ─── Layer 1: per-learner TTT ──────────────────────────────────────────
    learners = []
    for i in range(n_learners):
        learner_id = f"learner-{i:02d}"
        # Apply bias to a fraction of learners
        bias = 0.4 if (i / max(1, n_learners - 1)) < bias_fraction else 0.0
        adapter, receipts, consents = simulate_learner(
            learner_id, bias=bias, n_sessions=n_sessions,
            seed=seed_base + i,
        )
        learners.append({
            "id": learner_id,
            "bias": bias,
            "adapter": adapter,
            "receipts": receipts,
            "consents": consents,
        })
        if verbose:
            print(f"  L1 {learner_id:14} bias={bias:+.2f} "
                  f"applied={adapter.num_applied():3}/{adapter.num_applied()+adapter.num_skipped():3} "
                  f"(blocked={adapter.num_skipped()})")

    # ─── Layer 2: per-session gates ────────────────────────────────────────
    if verbose:
        print()
    l2_per_learner = {}
    for L in learners:
        view = make_layer2_view(L["id"])
        result = evaluate_session(view)
        l2_per_learner[L["id"]] = {
            "all_passed": result.all_passed,
            "num_passed": result.num_passed,
            "failure_reasons": result.failure_reasons,
        }
        if verbose:
            status = "all six pass" if result.all_passed else f"{result.num_passed}/6"
            print(f"  L2 {L['id']:14} {status}")

    # ─── Layer 3: fragment verification ────────────────────────────────────
    if verbose:
        print()
    fragments = []
    l3_per_learner = {}
    for L in learners:
        round_receipt = build_fragment_receipt(
            learner_id=L["id"],
            round_id=1,
            dataset_id="haic-fed-round-1",
            per_session_receipts=L["receipts"],
            per_session_consents=L["consents"],
        )
        shape = well_formed_shape()
        verification = verify_fragment(round_receipt, shape)
        fragments.append({
            "learner_id": L["id"],
            "receipt": round_receipt,
            "shape_summary": {
                "num_tensors": len(shape.tensor_names),
                "total_bytes": shape.total_bytes,
            },
            "verification": verification,
        })
        l3_per_learner[L["id"]] = {
            "verified": verification.verified,
            "failure_reason": verification.failure_reason,
            "warnings": verification.warnings,
            "computed_round_root": verification.computed_round_root,
        }
        if verbose:
            status = "VERIFIED" if verification.verified else f"REJECTED({verification.failure_reason})"
            print(f"  L3 {L['id']:14} {status}")

    # ─── Layer 4: federation viability ─────────────────────────────────────
    contributions = []
    for L, F in zip(learners, fragments):
        contributions.append(
            LearnerContribution(
                learner_id=L["id"],
                sessions_per_round=float(L["adapter"].num_applied()),
                avg_turns_per_session=6.0,
                consent_grant_rate=0.9,
                quantization_hostility=0.05 + abs(L["bias"]) * 0.1,
                is_verified=F["verification"].verified,
            )
        )
    policy = MergeQuorumPolicy(minimum_quorum=quorum)
    fed_result = assess_federated(contributions, policy=policy)

    if verbose:
        print()
        print(f"  L4 federation: "
              f"verified={fed_result.num_learners_verified}/{fed_result.num_learners_total}, "
              f"quorum_met={fed_result.quorum_met}, "
              f"Ceff={fed_result.ceff_global:.2f}, E={fed_result.e_global:.4f}, "
              f"recommendation={fed_result.round_recommendation.upper()}")

    # ─── Receipt chain ─────────────────────────────────────────────────────
    accepted_roots = [
        F["receipt"].claimed_round_root
        for F in fragments
        if F["verification"].verified
    ]
    federation_root = (
        merkle_root(accepted_roots) if accepted_roots else sha3_256_hex("empty_round")
    )
    zk_digest = sha3_256_hex(federation_root + "fed-round-1")

    receipt = {
        "kind": "federated_round_receipt",
        "round_id": 1,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_learners": n_learners,
            "bias_fraction": bias_fraction,
            "n_sessions": n_sessions,
            "quorum": quorum,
        },
        "layer_1_per_learner": {
            L["id"]: {
                "bias": L["bias"],
                "n_applied": L["adapter"].num_applied(),
                "n_skipped": L["adapter"].num_skipped(),
            }
            for L in learners
        },
        "layer_2_per_learner": l2_per_learner,
        "layer_3_per_learner": l3_per_learner,
        "layer_4": {
            "viable_global": fed_result.viable_global,
            "ceff_global": fed_result.ceff_global,
            "e_global": fed_result.e_global,
            "num_verified": fed_result.num_learners_verified,
            "num_rejected": fed_result.num_learners_rejected,
            "quorum_met": fed_result.quorum_met,
            "merge_error_estimate": fed_result.merge_error_estimate,
            "rejected_learners": fed_result.rejected_learners,
            "round_recommendation": fed_result.round_recommendation,
        },
        "federation_root": federation_root,
        "zk_digest": zk_digest,
    }

    # Self-verifying: hash the report (sorted-keys) to anchor it
    report_blob = json.dumps(receipt, sort_keys=True)
    receipt["self_anchor"] = sha3_256_hex(report_blob)

    if verbose:
        print()
        print(f"  Federation Merkle root: {federation_root}")
        print(f"  Zk digest:              {zk_digest}")
        print()

    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-learners", type=int, default=5)
    parser.add_argument("--bias-fraction", type=float, default=0.0,
                        help="Fraction of learners (by index order) to give systematic bias")
    parser.add_argument("--n-sessions", type=int, default=12)
    parser.add_argument("--quorum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    receipt = run_round(
        n_learners=args.n_learners,
        bias_fraction=args.bias_fraction,
        n_sessions=args.n_sessions,
        quorum=args.quorum,
        seed_base=args.seed,
        verbose=not args.quiet,
    )

    if args.out:
        args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Receipt written to {args.out}")
    elif args.quiet:
        # Quiet + no out → emit to stdout
        print(json.dumps(receipt, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
