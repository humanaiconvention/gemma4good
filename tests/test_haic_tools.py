"""
tests/test_haic_tools.py — Governance tool unit tests.

Covers the six functions in tools/haic_tools.py that previously had zero test
coverage (all existing tests targeted only incremental_grounding.py):

    assess_wellbeing       — wellbeing signal collection, mock fallback
    verify_consent         — 5-layer consent gate, hash determinism
    run_prism              — arena cache lookups, unknown-model fallback
    run_prism_analysis     — composite score math, None-safe _f() defaults
    check_viability_condition — Viability Condition math and arena cross-ref
    generate_receipt       — local Merkle root, determinism, structural fields

Also covers the _normalize_outlier_ratio() helper used by run_prism_analysis.

Gateway calls (requests.post / requests.get) are patched at the requests
module level so every test runs without a live Maestro server.
"""

import hashlib
import json
import math
import sys
import os
import pytest
from unittest.mock import patch

# Ensure the repo root is on sys.path so `from tools.haic_tools import ...`
# works regardless of how pytest is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.haic_tools import (          # noqa: E402
    _normalize_outlier_ratio,
    assess_wellbeing,
    verify_consent,
    run_prism,
    run_prism_analysis,
    generate_receipt,
    check_viability_condition,
    _ARENA_CACHE,
)


# ── _normalize_outlier_ratio ──────────────────────────────────────────────────

class TestNormalizeOutlierRatio:
    """Log-scale normalizer: outlier_ratio=1 → 0.0, =50 → 1.0, clamped."""

    def test_ratio_one_maps_to_zero(self):
        assert _normalize_outlier_ratio(1.0) == 0.0

    def test_ratio_fifty_maps_to_one(self):
        assert abs(_normalize_outlier_ratio(50.0) - 1.0) < 1e-9

    def test_below_one_clamped_via_max(self):
        # max(x, 1.0) is applied before log, so 0.5 acts like 1.0
        assert _normalize_outlier_ratio(0.5) == 0.0

    def test_above_fifty_clamped_to_one(self):
        assert _normalize_outlier_ratio(100.0) == 1.0
        assert _normalize_outlier_ratio(10_000.0) == 1.0

    def test_monotonically_increasing_one_to_fifty(self):
        prev = _normalize_outlier_ratio(1.0)
        for x in [2.0, 5.0, 10.0, 20.0, 49.9, 50.0]:
            curr = _normalize_outlier_ratio(x)
            assert curr >= prev, f"Not monotonic at {x}: {curr} < {prev}"
            prev = curr

    def test_geometric_midpoint_maps_to_half(self):
        # sqrt(50) is the geometric midpoint — log(sqrt(50))/log(50) = 0.5
        mid = _normalize_outlier_ratio(math.sqrt(50.0))
        assert abs(mid - 0.5) < 1e-9


# ── assess_wellbeing ──────────────────────────────────────────────────────────

class TestAssessWellbeing:
    """Wellbeing signal collection — fallback fires when gateway is down."""

    @staticmethod
    def _call(**kwargs):
        """Patch requests.post to raise, forcing the local fallback path."""
        defaults = dict(
            session_id="s-001",
            domain="economic_security",
            prompt_context="Tell me about your work situation.",
        )
        defaults.update(kwargs)
        with patch("requests.post", side_effect=ConnectionError("no server")):
            return assess_wellbeing(**defaults)

    def test_fallback_returns_required_keys(self):
        result = self._call()
        assert {"wellbeing_score", "domain", "narrative", "consent_given", "session_id"}.issubset(
            result.keys()
        )

    def test_fallback_score_is_0p5(self):
        assert self._call()["wellbeing_score"] == 0.5

    def test_fallback_consent_given_is_false(self):
        assert self._call()["consent_given"] is False

    def test_fallback_echoes_domain(self):
        result = self._call(domain="health")
        assert result["domain"] == "health"

    def test_fallback_echoes_session_id(self):
        result = self._call(session_id="ses-xyz")
        assert result["session_id"] == "ses-xyz"

    def test_fallback_narrative_is_string(self):
        assert isinstance(self._call()["narrative"], str)

    @pytest.mark.parametrize("domain", [
        "economic_security", "health", "autonomy",
        "social_connection", "meaning", "safety", "environment",
    ])
    def test_all_valid_domains_accepted(self, domain):
        result = self._call(domain=domain)
        assert result["domain"] == domain


# ── verify_consent ────────────────────────────────────────────────────────────

class TestVerifyConsent:
    """5-layer consent gate — hash determinism, layer filtering, validity."""

    ALL_GRANTED = {
        "transcript": "granted",
        "felt_state": "granted",
        "training_signal": "granted",
        "retention": "granted",
    }
    ALL_DENIED = {
        "transcript": "denied",
        "felt_state": "denied",
        "training_signal": "denied",
        "retention": "denied",
    }
    MIXED = {
        "transcript": "granted",
        "felt_state": "denied",
        "training_signal": "granted",
        "retention": "denied",
    }

    @staticmethod
    def _call(session_id="s-001", consent_layers=None):
        if consent_layers is None:
            consent_layers = {
                "transcript": "granted",
                "training_signal": "granted",
            }
        with patch("requests.post", side_effect=ConnectionError("no server")):
            return verify_consent(session_id=session_id, consent_layers=consent_layers)

    def test_hash_is_deterministic(self):
        r1 = self._call("s1", self.ALL_GRANTED)
        r2 = self._call("s1", self.ALL_GRANTED)
        assert r1["consent_hash"] == r2["consent_hash"]

    def test_hash_is_sha256_of_sorted_json(self):
        expected = hashlib.sha256(
            json.dumps(self.MIXED, sort_keys=True).encode()
        ).hexdigest()
        result = self._call("s1", self.MIXED)
        assert result["consent_hash"] == expected

    def test_hash_is_64_hex_chars(self):
        result = self._call()
        h = result["consent_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_all_granted_layers_listed(self):
        result = self._call("s1", self.ALL_GRANTED)
        assert set(result["layers_granted"]) == set(self.ALL_GRANTED.keys())

    def test_mixed_only_granted_layers_listed(self):
        result = self._call("s1", self.MIXED)
        assert set(result["layers_granted"]) == {"transcript", "training_signal"}

    def test_consent_valid_when_any_layer_granted(self):
        assert self._call("s1", self.MIXED)["consent_valid"] is True

    def test_consent_valid_when_all_granted(self):
        assert self._call("s1", self.ALL_GRANTED)["consent_valid"] is True

    def test_consent_invalid_when_all_denied(self):
        result = self._call("s1", self.ALL_DENIED)
        assert result["consent_valid"] is False
        assert result["layers_granted"] == []

    def test_echoes_session_id(self):
        assert self._call("ses-abc", self.MIXED)["session_id"] == "ses-abc"

    def test_different_inputs_produce_different_hashes(self):
        r1 = self._call("s1", self.ALL_GRANTED)
        r2 = self._call("s1", self.ALL_DENIED)
        assert r1["consent_hash"] != r2["consent_hash"]

    def test_session_id_does_not_affect_hash(self):
        # Hash is over consent_layers only (session_id is not included)
        r1 = self._call("session-A", self.MIXED)
        r2 = self._call("session-B", self.MIXED)
        assert r1["consent_hash"] == r2["consent_hash"]


# ── run_prism ────────────────────────────────────────────────────────────────

class TestRunPrism:
    """Arena cache hits and unknown-model fallback."""

    def test_known_model_source_is_arena_cache(self):
        result = run_prism("gemma4-e2b", "test prompt")
        assert result["source"] == "arena_cache"

    def test_known_model_id_echoed(self):
        result = run_prism("gemma4-e2b", "test prompt")
        assert result["model_id"] == "gemma4-e2b"

    def test_known_model_quantization_hostility(self):
        result = run_prism("gemma4-e2b", "test prompt")
        assert result["quantization_hostility"] == pytest.approx(0.9145)

    def test_layer_range_echoed(self):
        result = run_prism("haic-v6", "probe", layer_range="mid")
        assert result["layer_range"] == "mid"

    def test_verified_partial_model_in_cache(self):
        result = run_prism("haic-gemma4-v34", "probe")
        assert result["source"] == "arena_cache"
        assert result["data_status"] == "verified_partial"

    def test_verified_partial_has_none_fields(self):
        result = run_prism("haic-gemma4-v34", "probe")
        assert result["outlier_ratio"] is None
        assert result["cardinal_proximity"] is None

    def test_all_cached_models_present(self):
        # All 12 entries in the arena should be reachable
        for model_id in _ARENA_CACHE:
            result = run_prism(model_id, "probe")
            assert result["source"] == "arena_cache"

    def test_unknown_model_falls_back(self):
        with patch("requests.get", side_effect=ConnectionError("no server")):
            result = run_prism("nonexistent-model-xyz", "probe")
        assert result["source"] == "fallback"
        assert result["data_status"] == "placeholder"

    def test_unknown_model_fallback_has_required_keys(self):
        with patch("requests.get", side_effect=ConnectionError("no server")):
            result = run_prism("nonexistent-xyz", "probe")
        assert {
            "model_id", "outlier_ratio", "activation_kurtosis",
            "cardinal_proximity", "quantization_hostility",
        }.issubset(result.keys())


# ── run_prism_analysis ────────────────────────────────────────────────────────

class TestRunPrismAnalysis:
    """Composite score math, None-safe defaults, alignment_risk thresholds."""

    def test_returns_required_keys(self):
        result = run_prism_analysis("haic-v6", "probe")
        assert {
            "model_id", "transparency_score", "quantization_hostility",
            "alignment_risk", "dimensions", "composite_alignment",
            "data_status", "source",
        }.issubset(result.keys())

    def test_dimensions_subkeys_present(self):
        result = run_prism_analysis("haic-v6", "probe")
        assert set(result["dimensions"].keys()) == {
            "semantic_fidelity", "drift_detection", "info_density", "context_anxiety"
        }

    def test_haic_v6_composite_math(self):
        """Verify the exact composite formula against known haic-v6 values."""
        # haic-v6 arena: or=23.82, kurtosis=347.49, cp=0.3632, qh=0.7179
        or_, k, cp, qh = 23.82, 347.49, 0.3632, 0.7179
        sf  = max(0.0, 1.0 - min(math.log(max(or_, 1.0)) / math.log(50.0), 1.0))
        dd  = max(0.0, 1.0 - min(k / 1000.0, 1.0))
        id_ = max(0.0, 1.0 - cp)
        ca  = max(0.0, 1.0 - qh)
        expected_composite = 0.35 * sf + 0.30 * dd + 0.20 * id_ + 0.15 * ca

        result = run_prism_analysis("haic-v6", "probe")
        assert result["composite_alignment"] == pytest.approx(expected_composite, abs=1e-4)
        assert result["transparency_score"] == pytest.approx(expected_composite * 100.0, abs=0.02)

    def test_transparency_score_is_composite_times_100(self):
        result = run_prism_analysis("haic-v7", "probe")
        assert result["transparency_score"] == pytest.approx(
            result["composite_alignment"] * 100.0, abs=1e-3
        )

    def test_verified_partial_uses_none_safe_defaults(self):
        """haic-gemma4-v34 has outlier_ratio=None → default=50 → sf=0.0
        and cardinal_proximity=None → default=0.60 → id=0.40."""
        result = run_prism_analysis("haic-gemma4-v34", "probe")
        assert result["dimensions"]["semantic_fidelity"] == pytest.approx(0.0, abs=1e-9)
        assert result["dimensions"]["info_density"] == pytest.approx(0.40, abs=1e-4)
        assert result["data_status"] == "verified_partial"

    def test_verified_partial_v35_gov_uses_defaults(self):
        result = run_prism_analysis("haic-gemma4-v35-gov", "probe")
        assert result["dimensions"]["semantic_fidelity"] == pytest.approx(0.0, abs=1e-9)

    def test_alignment_risk_low_when_composite_above_0p8(self):
        # or=1.0 (sf=1.0), kurtosis=0 (dd=1.0), cp=0.0 (id=1.0), qh=0.0 (ca=1.0)
        # composite = 1.0 → "low"
        with patch("tools.haic_tools.run_prism") as mock_prism:
            mock_prism.return_value = {
                "outlier_ratio": 1.0, "activation_kurtosis": 0.0,
                "cardinal_proximity": 0.0, "quantization_hostility": 0.0,
                "data_status": "mock", "source": "mock",
            }
            result = run_prism_analysis("mock-model", "probe")
        assert result["alignment_risk"] == "low"
        assert result["composite_alignment"] == pytest.approx(1.0, abs=1e-9)

    def test_alignment_risk_medium_between_0p6_and_0p8(self):
        # Build inputs where each dimension = 0.7 → composite = 0.7 → "medium"
        # sf=0.7: _normalize(or) = 0.3 → or = 50^0.3
        # dd=0.7: kurtosis = 300
        # id=0.7: cp = 0.3
        # ca=0.7: qh = 0.3
        or_val = 50.0 ** 0.3
        with patch("tools.haic_tools.run_prism") as mock_prism:
            mock_prism.return_value = {
                "outlier_ratio": or_val, "activation_kurtosis": 300.0,
                "cardinal_proximity": 0.30, "quantization_hostility": 0.30,
                "data_status": "mock", "source": "mock",
            }
            result = run_prism_analysis("mock-model", "probe")
        assert result["alignment_risk"] == "medium"
        assert result["composite_alignment"] == pytest.approx(0.70, abs=1e-4)

    def test_alignment_risk_high_for_hostile_arena_models(self):
        # gemma4-e2b: qh=0.9145, high outlier_ratio → composite well below 0.6
        result = run_prism_analysis("gemma4-e2b", "probe")
        assert result["alignment_risk"] == "high"

    def test_transparency_score_in_valid_range(self):
        for model_id in ["haic-v6", "haic-v7", "haic-v8", "gemma4-v1",
                         "haic-gemma4-v34", "haic-gemma4-v35-gov"]:
            result = run_prism_analysis(model_id, "probe")
            assert 0.0 <= result["transparency_score"] <= 100.0, (
                f"{model_id} transparency_score={result['transparency_score']} out of range"
            )

    def test_all_dimensions_in_0_1_range(self):
        for model_id in _ARENA_CACHE:
            result = run_prism_analysis(model_id, "probe")
            for dim, val in result["dimensions"].items():
                assert 0.0 <= val <= 1.0, (
                    f"{model_id}.{dim}={val} out of [0,1]"
                )


# ── check_viability_condition ─────────────────────────────────────────────────

class TestCheckViabilityCondition:
    """Viability Condition math: Ceff(t) > E(t), autophagy thresholds, arena cross-ref."""

    @staticmethod
    def _call(**overrides):
        defaults = dict(
            model_id="haic-v6",
            deployment_context="test deployment",
            error_rate_estimate=1.0,
            verification_bandwidth_estimate=10.0,
            synthetic_data_ratio=0.0,
        )
        defaults.update(overrides)
        return check_viability_condition(**defaults)

    def test_viable_when_ceff_exceeds_error(self):
        result = self._call(error_rate_estimate=1.0, verification_bandwidth_estimate=5.0)
        assert result["viability_satisfied"] is True

    def test_not_viable_when_error_exceeds_ceff(self):
        result = self._call(error_rate_estimate=10.0, verification_bandwidth_estimate=2.0)
        assert result["viability_satisfied"] is False

    def test_exact_ratio_math(self):
        # effective_ceff = 10.0 * (1 - 0.2) = 8.0, ratio = 8.0 / 2.0 = 4.0
        result = self._call(
            error_rate_estimate=2.0,
            verification_bandwidth_estimate=10.0,
            synthetic_data_ratio=0.2,
        )
        assert result["ceff_vs_e_ratio"] == pytest.approx(4.0, abs=1e-4)

    def test_synthetic_ratio_reduces_effective_ceff(self):
        r_full = self._call(verification_bandwidth_estimate=10.0, synthetic_data_ratio=0.0)
        r_half = self._call(verification_bandwidth_estimate=10.0, synthetic_data_ratio=0.5)
        assert r_full["ceff_vs_e_ratio"] > r_half["ceff_vs_e_ratio"]

    def test_effective_ceff_in_inputs(self):
        # verification_bandwidth=10, synthetic_ratio=0.3 → effective_ceff=7.0
        result = self._call(
            verification_bandwidth_estimate=10.0,
            synthetic_data_ratio=0.3,
            error_rate_estimate=1.0,
        )
        assert result["inputs"]["effective_ceff"] == pytest.approx(7.0, abs=1e-4)

    def test_autophagy_risk_none_ratio_above_two(self):
        result = self._call(error_rate_estimate=1.0, verification_bandwidth_estimate=5.0)
        assert result["autophagy_risk"] == "none"

    def test_autophagy_risk_low_ratio_just_above_one(self):
        # ratio = 1.5 → "low"
        result = self._call(error_rate_estimate=2.0, verification_bandwidth_estimate=3.0)
        assert result["autophagy_risk"] == "low"

    def test_autophagy_risk_critical_ratio_below_0p3(self):
        # effective_ceff = 1.0 * 0.5 = 0.5, error=10 → ratio=0.05 < 0.3
        result = self._call(
            error_rate_estimate=10.0,
            verification_bandwidth_estimate=1.0,
            synthetic_data_ratio=0.5,
        )
        assert result["autophagy_risk"] == "critical"

    def test_temporal_signature_detected_when_violated_and_high_synthetic(self):
        result = self._call(
            error_rate_estimate=10.0,
            verification_bandwidth_estimate=1.0,
            synthetic_data_ratio=0.5,
        )
        assert result["temporal_signature_detected"] is True

    def test_temporal_signature_not_detected_when_viable(self):
        # effective_ceff = 10.0 * (1 - 0.2) = 8.0, ratio = 8.0 > 1.0 → viable
        # synthetic_data_ratio=0.2 < 0.3 → temporal signature false even if violated
        result = self._call(
            error_rate_estimate=1.0,
            verification_bandwidth_estimate=10.0,
            synthetic_data_ratio=0.2,
        )
        assert result["viability_satisfied"] is True
        assert result["temporal_signature_detected"] is False

    def test_temporal_signature_not_detected_when_low_synthetic_even_if_violated(self):
        # violated but synthetic_ratio <= 0.3
        result = self._call(
            error_rate_estimate=10.0,
            verification_bandwidth_estimate=2.0,
            synthetic_data_ratio=0.1,
        )
        assert result["viability_satisfied"] is False
        assert result["temporal_signature_detected"] is False

    def test_prism_cross_reference_added_for_cached_model(self):
        result = self._call(model_id="haic-v6")
        rec = result["scaling_recommendation"]
        assert "Prism" in rec
        assert "haic-v6" in rec

    def test_prism_cross_ref_includes_hostility_value(self):
        result = self._call(model_id="haic-v7")
        assert "0.7177" in result["scaling_recommendation"]

    def test_unknown_model_does_not_crash(self):
        result = self._call(model_id="nonexistent-model-xyz")
        assert "viability_satisfied" in result
        assert "Prism" not in result["scaling_recommendation"]

    def test_returns_required_keys(self):
        result = self._call()
        assert {
            "viability_satisfied", "ceff_vs_e_ratio", "autophagy_risk",
            "temporal_signature_detected", "scaling_recommendation", "inputs",
        }.issubset(result.keys())

    def test_inputs_echo_parameters(self):
        result = self._call(
            model_id="haic-v8",
            error_rate_estimate=2.0,
            verification_bandwidth_estimate=6.0,
            synthetic_data_ratio=0.1,
        )
        assert result["inputs"]["model_id"] == "haic-v8"
        assert result["inputs"]["error_rate_estimate"] == pytest.approx(2.0)
        assert result["inputs"]["synthetic_data_ratio"] == pytest.approx(0.1)


# ── generate_receipt ──────────────────────────────────────────────────────────

_MSGS = [
    {"role": "user", "content": "Hello, I work in agriculture."},
    {"role": "assistant", "content": "[PIVOT: SENSORY] What does the soil feel like?"},
]
_CONSENT = {"transcript": "granted", "training_signal": "granted"}


class TestGenerateReceipt:
    """Local Merkle receipt: determinism, structure, edge cases."""

    @staticmethod
    def _call(session_id="ses-001", messages=None, consent=None):
        if messages is None:
            messages = _MSGS
        if consent is None:
            consent = _CONSENT
        with patch("requests.post", side_effect=ConnectionError("no server")):
            return generate_receipt(
                session_id=session_id,
                messages=messages,
                consent=consent,
            )

    def test_returns_required_keys(self):
        result = self._call()
        assert {"merkle_root", "verifiable", "node_count", "source"}.issubset(
            result.keys()
        )

    def test_source_is_local_fallback(self):
        assert self._call()["source"] == "local_fallback"

    def test_verifiable_is_true(self):
        assert self._call()["verifiable"] is True

    def test_node_count_matches_messages_length(self):
        assert self._call(messages=_MSGS)["node_count"] == len(_MSGS)

    def test_merkle_root_is_deterministic(self):
        r1 = self._call()
        r2 = self._call()
        assert r1["merkle_root"] == r2["merkle_root"]

    def test_different_messages_produce_different_root(self):
        r1 = self._call(messages=[{"role": "user", "content": "Alpha"}])
        r2 = self._call(messages=[{"role": "user", "content": "Beta"}])
        assert r1["merkle_root"] != r2["merkle_root"]

    def test_different_consent_produces_different_root(self):
        r1 = self._call(consent={"transcript": "granted"})
        r2 = self._call(consent={"transcript": "denied"})
        assert r1["merkle_root"] != r2["merkle_root"]

    def test_merkle_root_is_64_char_hex(self):
        root = self._call()["merkle_root"]
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)

    def test_empty_messages_still_returns_valid_root(self):
        result = self._call(messages=[])
        assert len(result["merkle_root"]) == 64

    def test_single_message_deterministic(self):
        single = [{"role": "user", "content": "one"}]
        r1 = self._call(messages=single)
        r2 = self._call(messages=single)
        assert r1["merkle_root"] == r2["merkle_root"]

    def test_merkle_root_manual_verification(self):
        """Reproduce the local Merkle algorithm and compare to tool output."""
        messages = [{"role": "user", "content": "X"}]
        consent = {"k": "v"}

        # Replicate exactly what generate_receipt() does in the fallback
        nodes = [
            hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
            for m in messages
        ]
        nodes.append(
            hashlib.sha256(json.dumps(consent, sort_keys=True).encode()).hexdigest()
        )
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            nodes = [
                hashlib.sha256((nodes[i] + nodes[i + 1]).encode()).hexdigest()
                for i in range(0, len(nodes), 2)
            ]
        expected_root = nodes[0]

        result = self._call(messages=messages, consent=consent)
        assert result["merkle_root"] == expected_root

    def test_session_id_does_not_affect_merkle_root(self):
        # Receipt root is computed over messages + consent, not session_id
        r1 = self._call(session_id="session-A")
        r2 = self._call(session_id="session-B")
        assert r1["merkle_root"] == r2["merkle_root"]

    def test_odd_message_count_duplicates_last_node(self):
        # With 3 messages + 1 consent = 4 nodes → clean pairs
        # With 2 messages + 1 consent = 3 nodes → last duplicated before pairing
        # Just verify no crash and root is valid 64-char hex
        three_msgs = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = self._call(messages=three_msgs)
        assert len(result["merkle_root"]) == 64
