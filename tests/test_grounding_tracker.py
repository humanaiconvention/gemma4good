"""
Tests for viability/grounding_tracker.py

Validates the grounding trajectory tracker:
  - Session accumulation
  - Cumulative C(t) monotonicity
  - Viability trend computation
  - JSON serialization round-trip
"""

import json
import pytest

from viability.grounding_tracker import GroundingTracker, GroundingSession


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_session(session_id: str, sft_pairs: int = 3,
                  training_executed: bool = False,
                  consent: str = "granted",
                  loss_before: float = None,
                  loss_after: float = None) -> GroundingSession:
    return GroundingSession(
        session_id=session_id,
        timestamp="2026-04-10T12:00:00Z",
        sft_pair_count=sft_pairs,
        training_executed=training_executed,
        loss_before=loss_before,
        loss_after=loss_after,
        session_receipt_root=f"receipt_{session_id}",
        training_receipt_root=f"training_{session_id}" if training_executed else None,
        consent_training_signal=consent,
        steps_executed=5 if training_executed else 0,
    )


# ── Basic accumulation ────────────────────────────────────────────────────────

class TestGroundingTrackerBasics:

    def test_empty_tracker(self):
        tracker = GroundingTracker()
        assert tracker.session_count == 0
        assert tracker.cumulative_ceff() == 0.0
        assert tracker.monotonically_improving() is True

    def test_add_single_session(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1", sft_pairs=3))
        assert tracker.session_count == 1
        assert tracker.cumulative_ceff() == 3.0

    def test_add_multiple_sessions(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1", sft_pairs=3))
        tracker.add_session(_make_session("s2", sft_pairs=6))
        tracker.add_session(_make_session("s3", sft_pairs=2))
        assert tracker.session_count == 3
        assert tracker.cumulative_ceff() == 11.0

    def test_denied_sessions_dont_count(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1", sft_pairs=3, consent="granted"))
        tracker.add_session(_make_session("s2", sft_pairs=6, consent="denied"))
        tracker.add_session(_make_session("s3", sft_pairs=2, consent="granted"))
        assert tracker.cumulative_ceff() == 5.0  # 3 + 0 + 2
        assert tracker.consented_session_count == 2
        assert tracker.session_count == 3  # all logged, even denied

    def test_executed_count(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1", training_executed=False))
        tracker.add_session(_make_session("s2", training_executed=True))
        tracker.add_session(_make_session("s3", training_executed=True))
        assert tracker.executed_session_count == 2


# ── Monotonicity ──────────────────────────────────────────────────────────────

class TestMonotonicity:

    def test_always_true_for_valid_sessions(self):
        tracker = GroundingTracker()
        for i in range(10):
            tracker.add_session(_make_session(f"s{i}", sft_pairs=i + 1))
        assert tracker.monotonically_improving() is True

    def test_denied_sessions_dont_break_monotonicity(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1", sft_pairs=5, consent="granted"))
        tracker.add_session(_make_session("s2", sft_pairs=0, consent="denied"))
        tracker.add_session(_make_session("s3", sft_pairs=3, consent="granted"))
        assert tracker.monotonically_improving() is True


# ── Viability trend ───────────────────────────────────────────────────────────

class TestViabilityTrend:

    def test_trend_length_matches_sessions(self):
        tracker = GroundingTracker(e_t=0.9146)
        tracker.add_session(_make_session("s1", sft_pairs=3))
        tracker.add_session(_make_session("s2", sft_pairs=6))
        trend = tracker.viability_trend()
        assert len(trend) == 2

    def test_trend_cumulative_ceff_increases(self):
        tracker = GroundingTracker(e_t=0.9146)
        tracker.add_session(_make_session("s1", sft_pairs=3))
        tracker.add_session(_make_session("s2", sft_pairs=6))
        tracker.add_session(_make_session("s3", sft_pairs=2))
        trend = tracker.viability_trend()
        ceffs = [t["cumulative_ceff"] for t in trend]
        assert ceffs == [3.0, 9.0, 11.0]

    def test_trend_ratio_uses_e_t(self):
        tracker = GroundingTracker(e_t=1.0)
        tracker.add_session(_make_session("s1", sft_pairs=5))
        trend = tracker.viability_trend()
        assert trend[0]["ratio_ceff_e"] == 5.0

    def test_trend_denied_session_doesnt_increment(self):
        tracker = GroundingTracker(e_t=0.9146)
        tracker.add_session(_make_session("s1", sft_pairs=3, consent="granted"))
        tracker.add_session(_make_session("s2", sft_pairs=6, consent="denied"))
        trend = tracker.viability_trend()
        assert trend[0]["cumulative_ceff"] == 3.0
        assert trend[1]["cumulative_ceff"] == 3.0  # denied: no increment

    def test_trend_includes_loss_values(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session(
            "s1", training_executed=True,
            loss_before=2.5, loss_after=1.8,
        ))
        trend = tracker.viability_trend()
        assert trend[0]["loss_before"] == 2.5
        assert trend[0]["loss_after"] == 1.8

    def test_trend_dry_run_loss_is_none(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1", training_executed=False))
        trend = tracker.viability_trend()
        assert trend[0]["loss_before"] is None
        assert trend[0]["loss_after"] is None


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_has_required_fields(self):
        tracker = GroundingTracker(model_id="gemma-4-e2b", e_t=0.9146)
        tracker.add_session(_make_session("s1"))
        s = tracker.summary()
        assert s["model_id"] == "gemma-4-e2b"
        assert s["e_t"] == 0.9146
        assert s["total_sessions"] == 1
        assert s["monotonically_improving"] is True
        assert "cumulative_ceff" in s
        assert "cumulative_ratio" in s


# ── JSON serialization ────────────────────────────────────────────────────────

class TestJsonRoundTrip:

    def test_empty_tracker_round_trip(self):
        tracker = GroundingTracker(model_id="test-model", e_t=0.5)
        json_str = tracker.to_json()
        restored = GroundingTracker.from_json(json_str)
        assert restored.model_id == "test-model"
        assert restored.e_t == 0.5
        assert restored.session_count == 0

    def test_populated_tracker_round_trip(self):
        tracker = GroundingTracker(model_id="gemma-4-e2b", e_t=0.9146)
        tracker.add_session(_make_session("s1", sft_pairs=3, training_executed=True))
        tracker.add_session(_make_session("s2", sft_pairs=6, consent="denied"))

        json_str = tracker.to_json()
        restored = GroundingTracker.from_json(json_str)

        assert restored.session_count == 2
        assert restored.cumulative_ceff() == 3.0  # s2 denied
        assert restored.sessions[0].training_executed is True
        assert restored.sessions[1].consent_training_signal == "denied"

    def test_json_is_valid(self):
        tracker = GroundingTracker()
        tracker.add_session(_make_session("s1"))
        json_str = tracker.to_json()
        parsed = json.loads(json_str)  # should not raise
        assert "sessions" in parsed
        assert "model_id" in parsed

    def test_round_trip_preserves_summary(self):
        tracker = GroundingTracker(e_t=0.9146)
        tracker.add_session(_make_session("s1", sft_pairs=5))
        tracker.add_session(_make_session("s2", sft_pairs=3))

        restored = GroundingTracker.from_json(tracker.to_json())
        assert tracker.summary() == restored.summary()
