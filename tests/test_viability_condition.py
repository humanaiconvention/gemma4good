"""
tests/test_viability_condition.py — Unit tests for viability/viability_condition.py.

Covers:
  - assess(): Viability Condition math, boundary cases, autophagy risk tiers,
    temporal signature detection, Prism cross-check notes
  - from_prism_metrics(): E(t) and Ceff(t) derivation from Prism geometry

Direct tests against the standalone module — no tool wrapper involved.
"""

import os
import sys
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from viability.viability_condition import assess, from_prism_metrics, ViabilityAssessment


# ── assess() ──────────────────────────────────────────────────────────────────

class TestAssess:
    """Core Viability Condition: Ceff(t) > E(t)."""

    # ── Basic viability logic ─────────────────────────────────────────────

    def test_viable_when_ceff_exceeds_error(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=5.0,
                    synthetic_data_ratio=0.0)
        assert r.viability_satisfied is True

    def test_not_viable_when_error_exceeds_ceff(self):
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=2.0,
                    synthetic_data_ratio=0.0)
        assert r.viability_satisfied is False

    def test_ratio_exactly_one_is_not_viable(self):
        """The condition is strict: Ceff > E, not >=."""
        r = assess(error_rate_estimate=5.0, verification_bandwidth_estimate=5.0,
                    synthetic_data_ratio=0.0)
        # ratio = 5.0/5.0 = 1.0, strict > means NOT satisfied
        assert r.viability_satisfied is False

    def test_ratio_just_above_one_is_viable(self):
        r = assess(error_rate_estimate=5.0, verification_bandwidth_estimate=5.001,
                    synthetic_data_ratio=0.0)
        assert r.viability_satisfied is True

    # ── Zero and near-zero error rate ─────────────────────────────────────

    def test_zero_error_rate_gives_huge_ratio(self):
        """E(t)=0 → ratio approaches infinity via the max(E, 1e-9) guard."""
        r = assess(error_rate_estimate=0.0, verification_bandwidth_estimate=1.0,
                    synthetic_data_ratio=0.0)
        assert r.viability_satisfied is True
        assert r.ceff_vs_e_ratio > 1e6

    def test_tiny_error_rate(self):
        r = assess(error_rate_estimate=1e-10, verification_bandwidth_estimate=1.0,
                    synthetic_data_ratio=0.0)
        assert r.viability_satisfied is True

    # ── Negative inputs ───────────────────────────────────────────────────

    def test_negative_error_rate_does_not_crash(self):
        """Negative E(t) is nonsensical but shouldn't crash."""
        r = assess(error_rate_estimate=-1.0, verification_bandwidth_estimate=5.0,
                    synthetic_data_ratio=0.0)
        # -1.0 < 1e-9 guard → ratio = 5.0/1e-9 → huge → viable
        assert isinstance(r, ViabilityAssessment)

    def test_negative_ceff_not_viable(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=-5.0,
                    synthetic_data_ratio=0.0)
        assert r.viability_satisfied is False

    # ── Synthetic data ratio ──────────────────────────────────────────────

    def test_synthetic_ratio_reduces_effective_ceff(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.5)
        # effective_ceff = 10.0 * (1 - 0.5) = 5.0
        assert r.effective_ceff == pytest.approx(5.0, abs=1e-4)

    def test_synthetic_ratio_one_zeroes_ceff(self):
        """100% synthetic data → no real corrections → critical."""
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=100.0,
                    synthetic_data_ratio=1.0)
        assert r.effective_ceff == pytest.approx(0.0, abs=1e-9)
        assert r.viability_satisfied is False
        assert r.autophagy_risk == "critical"

    def test_synthetic_ratio_zero_no_reduction(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.0)
        assert r.effective_ceff == pytest.approx(10.0, abs=1e-4)

    # ── Autophagy risk tiers ──────────────────────────────────────────────

    def test_autophagy_none_when_ratio_above_two(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=3.0,
                    synthetic_data_ratio=0.0)
        assert r.autophagy_risk == "none"

    def test_autophagy_low_ratio_between_one_and_two(self):
        r = assess(error_rate_estimate=2.0, verification_bandwidth_estimate=3.0,
                    synthetic_data_ratio=0.0)
        assert r.ceff_vs_e_ratio == pytest.approx(1.5, abs=1e-4)
        assert r.autophagy_risk == "low"

    def test_autophagy_medium_ratio_between_0p7_and_1p0(self):
        # ratio = 0.8 → "medium"
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=8.0,
                    synthetic_data_ratio=0.0)
        assert r.autophagy_risk == "medium"

    def test_autophagy_high_ratio_between_0p3_and_0p7(self):
        # ratio = 0.5
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=5.0,
                    synthetic_data_ratio=0.0)
        assert r.autophagy_risk == "high"

    def test_autophagy_critical_ratio_below_0p3(self):
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=2.0,
                    synthetic_data_ratio=0.0)
        assert r.autophagy_risk == "critical"

    # ── Boundary values for autophagy tiers ───────────────────────────────

    def test_ratio_exactly_2p0_is_low_not_none(self):
        """Threshold is > 2.0 for 'none', so exactly 2.0 → 'low'."""
        r = assess(error_rate_estimate=5.0, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.0)
        assert r.ceff_vs_e_ratio == pytest.approx(2.0, abs=1e-4)
        assert r.autophagy_risk == "low"

    def test_ratio_exactly_0p7_is_high_not_medium(self):
        """Threshold is > 0.7 for 'medium', so exactly 0.7 → 'high'."""
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=7.0,
                    synthetic_data_ratio=0.0)
        assert r.ceff_vs_e_ratio == pytest.approx(0.7, abs=1e-4)
        assert r.autophagy_risk == "high"

    def test_ratio_exactly_0p3_is_critical(self):
        """Threshold is > 0.3 for 'high', so exactly 0.3 → 'critical'."""
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=3.0,
                    synthetic_data_ratio=0.0)
        assert r.ceff_vs_e_ratio == pytest.approx(0.3, abs=1e-4)
        assert r.autophagy_risk == "critical"

    # ── Temporal signature detection ──────────────────────────────────────

    def test_temporal_signature_when_violated_and_high_synthetic(self):
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=1.0,
                    synthetic_data_ratio=0.5)
        assert r.temporal_signature_detected is True

    def test_no_temporal_signature_when_viable(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.5)
        assert r.temporal_signature_detected is False

    def test_no_temporal_signature_when_low_synthetic_even_if_violated(self):
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=2.0,
                    synthetic_data_ratio=0.1)
        assert r.viability_satisfied is False
        assert r.temporal_signature_detected is False

    def test_temporal_threshold_exactly_0p3_synthetic(self):
        """synthetic_data_ratio must be > 0.3, not >= 0.3."""
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=1.0,
                    synthetic_data_ratio=0.3)
        assert r.viability_satisfied is False
        assert r.temporal_signature_detected is False

    def test_temporal_threshold_just_above_0p3(self):
        r = assess(error_rate_estimate=10.0, verification_bandwidth_estimate=1.0,
                    synthetic_data_ratio=0.31)
        assert r.temporal_signature_detected is True

    # ── Prism hostility cross-check ───────────────────────────────────────

    def test_prism_note_when_hostility_exceeds_estimate(self):
        r = assess(error_rate_estimate=0.5, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.0, prism_hostility=0.9)
        assert "Prism hostility" in r.scaling_recommendation

    def test_no_prism_note_when_hostility_below_estimate(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.0, prism_hostility=0.5)
        assert "Prism hostility" not in r.scaling_recommendation

    def test_no_prism_note_when_hostility_none(self):
        r = assess(error_rate_estimate=1.0, verification_bandwidth_estimate=10.0,
                    synthetic_data_ratio=0.0, prism_hostility=None)
        assert "Prism hostility" not in r.scaling_recommendation

    # ── Return type and fields ────────────────────────────────────────────

    def test_returns_viability_assessment(self):
        r = assess(1.0, 5.0, 0.0)
        assert isinstance(r, ViabilityAssessment)

    def test_model_id_stored(self):
        r = assess(1.0, 5.0, 0.0, model_id="test-model")
        assert r.model_id == "test-model"

    def test_repr_contains_status(self):
        r = assess(1.0, 5.0, 0.0)
        assert "VIABLE" in repr(r)

    def test_repr_violated_contains_risk(self):
        r = assess(10.0, 1.0, 0.0)
        s = repr(r)
        assert "VIOLATED" in s
        assert "RISK" in s


# ── from_prism_metrics() ──────────────────────────────────────────────────────

class TestFromPrismMetrics:
    """Convenience constructor deriving E(t) and Ceff(t) from Prism+Maestro data."""

    def test_e_t_equals_hostility_times_scale(self):
        r = from_prism_metrics(
            outlier_ratio=50.0, activation_kurtosis=500.0,
            cardinal_proximity=0.5, quantization_hostility=0.8,
            sessions_per_day=10.0, deployment_scale_factor=2.0,
        )
        # E(t) = 0.8 * 2.0 = 1.6
        assert r.error_rate == pytest.approx(1.6, abs=1e-4)

    def test_ceff_t_from_sessions(self):
        r = from_prism_metrics(
            outlier_ratio=50.0, activation_kurtosis=500.0,
            cardinal_proximity=0.5, quantization_hostility=0.8,
            sessions_per_day=10.0,
            avg_turns_per_session=6.0,
            consent_grant_rate=0.85,
            synthetic_data_ratio=0.0,
        )
        # Ceff = 10 * 6 * 0.85 = 51.0
        expected_ceff = 10.0 * 6.0 * 0.85
        assert r.effective_ceff == pytest.approx(expected_ceff, abs=1e-2)

    def test_default_parameters(self):
        r = from_prism_metrics(
            outlier_ratio=50.0, activation_kurtosis=500.0,
            cardinal_proximity=0.5, quantization_hostility=0.8,
            sessions_per_day=10.0,
        )
        # defaults: avg_turns=6, consent_rate=0.85, synthetic=0.0, scale=1.0
        # E(t) = 0.8 * 1.0 = 0.8
        # Ceff = 10 * 6 * 0.85 = 51.0
        assert r.error_rate == pytest.approx(0.8, abs=1e-4)
        assert r.viability_satisfied is True

    def test_high_hostility_low_sessions_not_viable(self):
        r = from_prism_metrics(
            outlier_ratio=100.0, activation_kurtosis=1000.0,
            cardinal_proximity=0.9, quantization_hostility=0.95,
            sessions_per_day=0.1,  # very low throughput
            deployment_scale_factor=100.0,
        )
        # E(t) = 0.95 * 100 = 95.0
        # Ceff = 0.1 * 6 * 0.85 = 0.51
        assert r.viability_satisfied is False
        assert r.autophagy_risk in ("high", "critical")

    def test_model_id_passed_through(self):
        r = from_prism_metrics(
            outlier_ratio=50.0, activation_kurtosis=500.0,
            cardinal_proximity=0.5, quantization_hostility=0.5,
            sessions_per_day=10.0, model_id="test-v1",
        )
        assert r.model_id == "test-v1"

    def test_prism_hostility_stored(self):
        r = from_prism_metrics(
            outlier_ratio=50.0, activation_kurtosis=500.0,
            cardinal_proximity=0.5, quantization_hostility=0.75,
            sessions_per_day=10.0,
        )
        assert r.prism_hostility == pytest.approx(0.75)

    def test_synthetic_data_ratio_applied(self):
        r = from_prism_metrics(
            outlier_ratio=50.0, activation_kurtosis=500.0,
            cardinal_proximity=0.5, quantization_hostility=0.5,
            sessions_per_day=10.0,
            synthetic_data_ratio=0.5,
        )
        # Ceff_raw = 10 * 6 * 0.85 = 51.0
        # effective_ceff = 51.0 * (1 - 0.5) = 25.5
        assert r.effective_ceff == pytest.approx(25.5, abs=1e-2)
