"""
Tests for tools/check_promotion.py.

Pure-Python — exercises the gate functions and the aggregation rule
without actually loading any model. Mirrors the discipline of
test_sgt_harness.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_promotion import (  # noqa: E402
    PROFILES,
    GateVerdict,
    gate_capability_gain,
    gate_leakage,
    gate_consistency,
    gate_covenant,
    gate_isolation,
    gate_epistemic,
    aggregate_decision,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _report(grounding_samp_rate=0.5, grounding_det_rate=0.5,
            grounding_ci=(0.3, 0.7), security_rate=1.0,
            security_trials=20, n_per_scenario=10,
            seed=42, model_id="haic-test", precision="4bit_nf4",
            with_baseline=False, baseline_rate=0.10,
            baseline_ci=(0.05, 0.20)):
    """Build a synthetic report dict matching run_sgt's output shape."""
    samp = {
        "pass_type": "sampling",
        "n_per_scenario": n_per_scenario,
        "grounding_passes": int(grounding_samp_rate * 30),
        "grounding_trials": 30,
        "security_passes": int(security_rate * security_trials),
        "security_trials": security_trials,
        "security_fails": security_trials - int(security_rate * security_trials),
        "grounding_pass_rate": grounding_samp_rate,
        "grounding_ci95": list(grounding_ci),
        "security_pass_rate": security_rate,
        "security_ci95": [security_rate, 1.0],  # placeholder
        "sgt_score_out_of_10": grounding_samp_rate * 10,
        "per_scenario": [],
        "seed": seed,
        "model_id": model_id,
        "decoding": {"temperature": 0.7, "top_p": 0.9, "precision": precision},
    }
    det = dict(samp)
    det.update({
        "pass_type": "deterministic",
        "n_per_scenario": 1,
        "grounding_passes": int(grounding_det_rate * 3),
        "grounding_trials": 3,
        "grounding_pass_rate": grounding_det_rate,
    })
    rep = {"finetune": {"deterministic": det, "sampling": samp}}
    if with_baseline:
        b_samp = dict(samp)
        b_samp["grounding_pass_rate"] = baseline_rate
        b_samp["grounding_ci95"] = list(baseline_ci)
        b_samp["model_id"] = "google/gemma-4-E2B-it"
        rep["baseline"] = {"deterministic": dict(samp), "sampling": b_samp}
    return rep


def _leakage_pass():
    return {"verdict": "PASS", "exact_hits": [], "near_hits": [],
            "training_shards": [{"path": "x.jsonl"}]}


def _leakage_fail_exact():
    return {"verdict": "BLOCKED_EXACT_MATCH",
            "exact_hits": [{"scenario_id": "sgt_emotional", "shard": "x.jsonl"}],
            "near_hits": [], "training_shards": [{"path": "x.jsonl"}]}


# ── Gate 1: capability gain ─────────────────────────────────────────────────


class TestGateCapabilityGain:

    def test_pass_when_delta_positive_and_cis_disjoint(self):
        rep = _report(grounding_samp_rate=0.7, grounding_ci=(0.55, 0.85),
                      with_baseline=True, baseline_rate=0.10, baseline_ci=(0.02, 0.30))
        v = gate_capability_gain(rep, PROFILES["default"])
        assert v.status == "PASS"

    def test_fail_when_delta_below_threshold(self):
        rep = _report(grounding_samp_rate=0.30, grounding_ci=(0.20, 0.40),
                      with_baseline=True, baseline_rate=0.25, baseline_ci=(0.15, 0.35))
        v = gate_capability_gain(rep, PROFILES["default"])
        assert v.status == "FAIL"
        assert "below threshold" in v.rationale.lower() or "Δ" in v.rationale

    def test_fail_when_cis_overlap_even_if_delta_above(self):
        rep = _report(grounding_samp_rate=0.50, grounding_ci=(0.30, 0.70),
                      with_baseline=True, baseline_rate=0.30, baseline_ci=(0.15, 0.50))
        v = gate_capability_gain(rep, PROFILES["default"])
        assert v.status == "FAIL"
        assert "overlap" in v.rationale.lower()

    def test_indeterminate_when_no_baseline(self):
        rep = _report(with_baseline=False)
        v = gate_capability_gain(rep, PROFILES["default"])
        assert v.status == "INDETERMINATE"


# ── Gate 2: leakage ─────────────────────────────────────────────────────────


class TestGateLeakage:

    def test_indeterminate_when_no_receipt(self):
        v = gate_leakage(None)
        assert v.status == "INDETERMINATE"

    def test_pass_when_receipt_pass(self):
        v = gate_leakage(_leakage_pass())
        assert v.status == "PASS"

    def test_fail_when_exact_hit(self):
        v = gate_leakage(_leakage_fail_exact())
        assert v.status == "FAIL"


# ── Gate 3: consistency ─────────────────────────────────────────────────────


class TestGateConsistency:

    def test_pass_when_within_tolerance(self):
        rep = _report(grounding_samp_rate=0.40, grounding_det_rate=0.33)
        v = gate_consistency(rep, PROFILES["default"])
        assert v.status == "PASS"

    def test_fail_when_outside_tolerance(self):
        rep = _report(grounding_samp_rate=0.40, grounding_det_rate=0.95)
        v = gate_consistency(rep, PROFILES["default"])
        assert v.status == "FAIL"


# ── Gate 4: covenant ────────────────────────────────────────────────────────


class TestGateCovenant:

    def test_pass_when_all_fields_present(self, tmp_path):
        rep = _report()
        p = tmp_path / "report.json"
        p.write_text(json.dumps(rep))
        v = gate_covenant(rep, p)
        assert v.status == "PASS"
        assert "report_sha256" in v.measured

    def test_partial_when_seed_missing(self, tmp_path):
        rep = _report()
        rep["finetune"]["sampling"]["seed"] = None
        p = tmp_path / "report.json"
        p.write_text(json.dumps(rep))
        v = gate_covenant(rep, p)
        assert v.status == "PARTIAL"
        assert "seed" in v.rationale


# ── Gate 5: isolation ───────────────────────────────────────────────────────


class TestGateIsolation:

    def test_pass_when_eval_matches_deploy_precision(self):
        rep = _report(precision="GGUF Q5_K_M")
        v = gate_isolation(rep)
        assert v.status == "PASS"

    def test_partial_when_eval_differs_from_deploy(self):
        rep = _report(precision="4bit_nf4")
        v = gate_isolation(rep)
        assert v.status == "PARTIAL"
        assert "differs" in v.rationale.lower() or "spread" in v.rationale.lower()

    def test_partial_when_decoding_is_none(self):
        rep = _report()
        rep["finetune"]["sampling"]["decoding"] = None
        v = gate_isolation(rep)
        assert v.status == "PARTIAL"
        assert "not recorded" in v.rationale.lower() or "unknown" in v.rationale.lower()

    def test_partial_when_decoding_missing(self):
        rep = _report()
        del rep["finetune"]["sampling"]["decoding"]
        v = gate_isolation(rep)
        assert v.status == "PARTIAL"


# ── Gate 6: epistemic alignment ─────────────────────────────────────────────


class TestGateEpistemic:

    def test_pass_when_lower_ci_high_and_security_perfect(self):
        rep = _report(grounding_ci=(0.65, 0.90), security_rate=1.0)
        v = gate_epistemic(rep, PROFILES["default"])
        assert v.status == "PASS"

    def test_fail_when_lower_ci_below_threshold(self):
        rep = _report(grounding_ci=(0.22, 0.54), security_rate=1.0)
        v = gate_epistemic(rep, PROFILES["default"])
        assert v.status == "FAIL"

    def test_fail_when_security_below_threshold(self):
        rep = _report(grounding_ci=(0.65, 0.90), security_rate=0.50)
        v = gate_epistemic(rep, PROFILES["default"])
        assert v.status == "FAIL"


# ── Aggregation ─────────────────────────────────────────────────────────────


def _v(name, status, rationale="x"):
    return GateVerdict(name=name, status=status, rationale=rationale)


class TestAggregateDecision:

    def test_all_pass_promoted(self):
        verdicts = [_v(f"g{i}", "PASS") for i in range(6)]
        decision, code, _ = aggregate_decision(verdicts)
        assert decision == "PROMOTED"
        assert code == 0

    def test_any_fail_blocks(self):
        verdicts = [_v("g1", "PASS"), _v("g6", "FAIL"), _v("g3", "PASS")]
        decision, code, rationale = aggregate_decision(verdicts)
        assert decision == "BLOCKED"
        assert code == 1
        assert "g6" in rationale

    def test_indeterminate_dominates_partial(self):
        verdicts = [_v("g1", "PARTIAL"), _v("g2", "INDETERMINATE")]
        decision, code, _ = aggregate_decision(verdicts)
        assert decision == "INDETERMINATE"
        assert code == 2

    def test_partial_alone_still_promoted_with_advisory(self):
        verdicts = [_v("g1", "PARTIAL"), _v("g2", "PASS"), _v("g3", "PASS")]
        decision, code, rationale = aggregate_decision(verdicts)
        assert decision == "PROMOTED"
        assert code == 0
        assert "advisory" in rationale.lower() or "review" in rationale.lower()

    def test_fail_dominates_indeterminate(self):
        verdicts = [_v("g1", "FAIL"), _v("g2", "INDETERMINATE")]
        decision, _, _ = aggregate_decision(verdicts)
        assert decision == "BLOCKED"


# ── Profile coverage ────────────────────────────────────────────────────────


class TestProfiles:

    def test_strict_more_demanding_than_default(self):
        assert PROFILES["strict"]["min_lower_ci_grounding"] > PROFILES["default"]["min_lower_ci_grounding"]
        assert PROFILES["strict"]["min_delta_grounding"]    > PROFILES["default"]["min_delta_grounding"]

    def test_loose_less_demanding_than_default(self):
        assert PROFILES["loose"]["min_lower_ci_grounding"] < PROFILES["default"]["min_lower_ci_grounding"]
        assert PROFILES["loose"]["min_delta_grounding"]    < PROFILES["default"]["min_delta_grounding"]
