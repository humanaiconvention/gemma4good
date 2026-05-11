#!/usr/bin/env python3
"""
runtime_loop_stress_test.py — End-to-end stress test of the four-layer
runtime grounding loop.

Mirrors the SimSat `scripts/viability_gates_exercise.py` pattern: drive the
new modules through several synthetic streams and tally how each gate /
layer responds. Produces a JSON report that documents the empirical
behaviour under known conditions, complementing the unit-test suite.

Streams:
  baseline_clean       — well-behaved federation; should commit
  systematic_bias      — one clinic biased; TTT BLOCKING should fire
  hostile_fragment     — one clinic's fragment is corrupted; L3 should reject
  cloud_blackout       — two stations report zero sessions; quorum still met
  consent_denial       — one session has denied consent; L2 should reject
  poisoning            — one fragment has out-of-bound tensor norms; L3 rejects
  federation_collapse  — only 2 of 5 verified, below K=3 quorum → rollback
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viability.distributed_viability import (
    LearnerContribution, MergeQuorumPolicy, assess_federated,
)
from viability.session_gates import GroundingSessionView, evaluate_session
from viability.ttt_gates import TTTTrustSnapshot, TTTUpdateRecord, evaluate_ttt
from tools.diloco_fragment_verifier import (
    FragmentExpectation, FragmentShape, build_fragment_receipt, verify_fragment,
)
from tools.edge_ttt_adapter import EdgeTTTAdapter, OperatorFeedback
from utils.merkle import sha3_256_hex, merkle_root


# ──────────────────────────────────────────────────────────────────────────
# Stream simulators
# ──────────────────────────────────────────────────────────────────────────

def _full_consent() -> dict:
    return {layer: True for layer in (
        "transcript", "felt_state", "gfs_activations",
        "training_signal", "retention"
    )}


def _denied_training_signal_consent() -> dict:
    c = _full_consent()
    c["training_signal"] = False
    return c


def _step_fn_factory(drift_per_step: float = 0.001):
    cumulative = {f"w_{k}": 0.0 for k in range(3)}
    def step_fn(feedback):
        for k in cumulative:
            cumulative[k] += drift_per_step
        return dict(cumulative)
    return step_fn


def simulate_clinic_l1_trace(*, clinic_id: str, n_sessions: int = 12,
                              bias: float = 0.0,
                              consent_override: dict | None = None,
                              step_fn_factory: Callable = _step_fn_factory,
                              seed: int = 0) -> EdgeTTTAdapter:
    """Layer 1 simulation: run an EdgeTTTAdapter for n_sessions."""
    adapter = EdgeTTTAdapter(step_fn=step_fn_factory())
    rng = random.Random(seed if seed else (hash(clinic_id) & 0xFFFFFFFF))
    consent = consent_override if consent_override is not None else _full_consent()
    for i in range(n_sessions):
        err = bias + rng.gauss(0, 0.4)
        fb = OperatorFeedback(
            session_id=f"{clinic_id}-sess-{i:03d}",
            predicted="x",
            operator_label="y",
            error=err,
            consent_layers=dict(consent),
        )
        adapter.step(fb)
    return adapter


def _make_l2_view_for_clinic(clinic_id: str) -> GroundingSessionView:
    """Layer 2 simulation: produce a session view that should pass all six gates."""
    return GroundingSessionView(
        session_id=clinic_id,
        interview_turns=[
            {"role": "assistant", "content": "What did you experience this morning?"},
            {"role": "user", "content": "I felt grounded after my walk in the forest"},
            {"role": "assistant", "content": "Can you describe the sensations?"},
            {"role": "user", "content": "Cool damp air, scent of pine resin, sunlight on moss"},
            {"role": "assistant", "content": "What memory does this surface?"},
            {"role": "user", "content": "Walking with my grandfather collecting mushrooms as a child"},
        ],
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


def _well_formed_shape(num_layers: int = 35) -> FragmentShape:
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
    return FragmentShape(
        tensor_names=names,
        tensor_shapes=shapes,
        tensor_norms=norms,
        total_bytes=total,
    )


def make_fragment(clinic_id: str, n_sessions: int = 12,
                  *, corrupt_root: bool = False,
                  null_tensors: bool = False,
                  denied_consent_in_session: int | None = None) -> tuple:
    """Build a fragment receipt + shape, with optional fault injection."""
    full_consent = _full_consent()
    receipts = []
    consents = []
    for i in range(n_sessions):
        receipts.append({
            "session_id": f"{clinic_id}-{i}",
            "kind": "maestro_session_trace",
            "ts": "2026-05-11T00:00:00Z",
        })
        if denied_consent_in_session is not None and i == denied_consent_in_session:
            consents.append(_denied_training_signal_consent())
        else:
            consents.append(dict(full_consent))

    receipt = build_fragment_receipt(
        learner_id=clinic_id,
        round_id=1,
        dataset_id=f"haic-clinic-week-1",
        per_session_receipts=receipts,
        per_session_consents=consents,
    )
    if corrupt_root:
        receipt.claimed_round_root = "deadbeef" * 8

    shape = _well_formed_shape()
    if null_tensors:
        for name in shape.tensor_norms:
            shape.tensor_norms[name] = 0.0

    return receipt, shape


# ──────────────────────────────────────────────────────────────────────────
# Stress runners (one per stream)
# ──────────────────────────────────────────────────────────────────────────

def run_baseline_clean() -> dict:
    """Five healthy clinics; all gates clean; round should commit cleanly."""
    adapters = {f"clinic-{i}": simulate_clinic_l1_trace(clinic_id=f"clinic-{i}", bias=0.0)
                for i in range(5)}
    l1_blocked_total = sum(a.num_skipped() for a in adapters.values())

    # L2: all sessions pass all six gates
    l2_results = {}
    for cid in adapters:
        view = _make_l2_view_for_clinic(cid)
        l2_results[cid] = evaluate_session(view)
    l2_all_passed = all(r.all_passed for r in l2_results.values())

    # L3: all fragments verify
    l3_results = {}
    fragments = {}
    for cid in adapters:
        receipt, shape = make_fragment(cid)
        fragments[cid] = (receipt, shape)
        l3_results[cid] = verify_fragment(receipt, shape)
    verified = [cid for cid, r in l3_results.items() if r.verified]

    # L4: federated assessment
    contributions = [
        LearnerContribution(
            learner_id=cid,
            sessions_per_round=float(adapters[cid].num_applied()),
            quantization_hostility=0.05,
            is_verified=(cid in verified),
        )
        for cid in adapters
    ]
    l4 = assess_federated(contributions, policy=MergeQuorumPolicy(minimum_quorum=3))

    return {
        "stream": "baseline_clean",
        "l1_blocked_steps": l1_blocked_total,
        "l2_all_passed": l2_all_passed,
        "l3_verified_count": len(verified),
        "l3_rejected_count": 5 - len(verified),
        "l4_recommendation": l4.round_recommendation,
        "l4_viable": l4.viable_global,
        "expected": "commit",
        "passed": l4.round_recommendation == "commit",
    }


def run_systematic_bias() -> dict:
    """One clinic biased; TTT BLOCKING should fire; other clinics still verify."""
    adapters = {}
    for i in range(5):
        bias = 0.4 if i == 2 else 0.0   # clinic-2 has systematic positive bias
        adapters[f"clinic-{i}"] = simulate_clinic_l1_trace(
            clinic_id=f"clinic-{i}", bias=bias, n_sessions=20,
        )
    # The biased clinic should have meaningful skipped steps
    biased_skipped = adapters["clinic-2"].num_skipped()
    healthy_skipped = sum(adapters[f"clinic-{i}"].num_skipped() for i in (0, 1, 3, 4))

    contributions = [
        LearnerContribution(
            learner_id=cid,
            sessions_per_round=float(adapters[cid].num_applied()),
            quantization_hostility=0.05,
            is_verified=True,
        )
        for cid in adapters
    ]
    l4 = assess_federated(contributions)

    return {
        "stream": "systematic_bias",
        "biased_clinic_skipped": biased_skipped,
        "healthy_clinic_skipped_total": healthy_skipped,
        "l4_recommendation": l4.round_recommendation,
        "expected": "commit",
        "passed": biased_skipped > healthy_skipped and l4.round_recommendation == "commit",
        "note": "BLOCKING gate fires more for biased clinic; federation still viable",
    }


def run_hostile_fragment() -> dict:
    """One clinic's fragment has a corrupted root; L3 should reject."""
    adapters = {f"clinic-{i}": simulate_clinic_l1_trace(clinic_id=f"clinic-{i}") for i in range(5)}
    fragments = {}
    l3_results = {}
    for i, cid in enumerate(adapters):
        corrupt = (i == 0)
        receipt, shape = make_fragment(cid, corrupt_root=corrupt)
        fragments[cid] = (receipt, shape)
        l3_results[cid] = verify_fragment(receipt, shape)

    verified = [cid for cid, r in l3_results.items() if r.verified]
    contributions = [
        LearnerContribution(
            learner_id=cid,
            sessions_per_round=12.0,
            quantization_hostility=0.05,
            is_verified=(cid in verified),
        )
        for cid in adapters
    ]
    l4 = assess_federated(contributions)

    return {
        "stream": "hostile_fragment",
        "l3_rejected_count": 5 - len(verified),
        "l4_recommendation": l4.round_recommendation,
        "expected": "alert_operator",
        "passed": l4.round_recommendation == "alert_operator" and 5 - len(verified) == 1,
    }


def run_cloud_blackout() -> dict:
    """Two monitoring stations report zero sessions (cloud cover);
    quorum K=3 still met from the remaining 3."""
    contributions = [
        LearnerContribution(
            learner_id=f"station-{i}",
            sessions_per_round=0.0 if i in (0, 1) else 10.0,
            quantization_hostility=0.08,
            is_verified=True,
        )
        for i in range(5)
    ]
    l4 = assess_federated(contributions, policy=MergeQuorumPolicy(minimum_quorum=3))

    return {
        "stream": "cloud_blackout",
        "zero_session_stations": 2,
        "active_stations": 3,
        "l4_recommendation": l4.round_recommendation,
        "expected": "commit",
        "passed": l4.round_recommendation == "commit" and l4.viable_global,
    }


def run_consent_denial() -> dict:
    """One session in clinic-0's fragment has denied training_signal;
    L3 should reject that fragment outright."""
    receipt, shape = make_fragment("clinic-0", denied_consent_in_session=3)
    result = verify_fragment(receipt, shape)

    return {
        "stream": "consent_denial",
        "l3_verified": result.verified,
        "l3_failure_reason": result.failure_reason,
        "expected": "REJECTED with consent failure",
        "passed": (not result.verified) and "denied_consent_layers" in (result.failure_reason or ""),
    }


def run_poisoning() -> dict:
    """One clinic's fragment has all-zero tensor norms (null-trained
    poisoning attempt); L3 should reject."""
    receipt, shape = make_fragment("clinic-0", null_tensors=True)
    result = verify_fragment(receipt, shape)

    return {
        "stream": "poisoning",
        "l3_verified": result.verified,
        "l3_failure_reason": result.failure_reason,
        "expected": "REJECTED with null_training_pattern",
        "passed": (not result.verified) and "null_training_pattern" in (result.failure_reason or ""),
    }


def run_federation_collapse() -> dict:
    """3 of 5 clinics fail verification; quorum K=3 NOT met → rollback."""
    contributions = []
    for i in range(5):
        is_verified = (i < 2)   # only first 2 verified, K=3 needed → fails quorum
        contributions.append(
            LearnerContribution(
                learner_id=f"clinic-{i}",
                sessions_per_round=12.0,
                quantization_hostility=0.5 if not is_verified else 0.05,
                is_verified=is_verified,
            )
        )
    l4 = assess_federated(contributions, policy=MergeQuorumPolicy(minimum_quorum=3))

    return {
        "stream": "federation_collapse",
        "verified_count": 2,
        "quorum_met": l4.quorum_met,
        "l4_recommendation": l4.round_recommendation,
        "expected": "rollback",
        "passed": l4.round_recommendation == "rollback" and not l4.quorum_met,
    }


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

STREAMS = [
    run_baseline_clean,
    run_systematic_bias,
    run_hostile_fragment,
    run_cloud_blackout,
    run_consent_denial,
    run_poisoning,
    run_federation_collapse,
]


def main() -> int:
    results = []
    for runner in STREAMS:
        t0 = time.time()
        result = runner()
        result["elapsed_sec"] = round(time.time() - t0, 4)
        results.append(result)

    report = {
        "kind": "runtime_loop_stress_test",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "module_versions": {
            "viability/ttt_gates.py": "1.0",
            "viability/session_gates.py": "1.0",
            "viability/distributed_viability.py": "1.0",
            "tools/diloco_fragment_verifier.py": "1.0",
            "tools/edge_ttt_adapter.py": "1.0",
            "tools/enforcement_evidence_contract.py": "1.0",
        },
        "streams": results,
        "all_passed": all(r["passed"] for r in results),
        "summary": {
            "n_streams": len(results),
            "n_passed": sum(1 for r in results if r["passed"]),
            "n_failed": sum(1 for r in results if not r["passed"]),
        },
    }

    # Stable receipt over the report (sorted keys)
    blob = json.dumps(report, sort_keys=True)
    report["receipt_root"] = sha3_256_hex(blob)

    out = Path("experiments/runtime_loop_stress_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Runtime Loop Stress Test")
    print("=" * 70)
    print(f"  Streams run:   {report['summary']['n_streams']}")
    print(f"  Passed:        {report['summary']['n_passed']}")
    print(f"  Failed:        {report['summary']['n_failed']}")
    print()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['stream']:30}  expected={r['expected']!r}")
    print()
    print(f"Report:    {out}")
    print(f"Receipt:   {report['receipt_root']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
