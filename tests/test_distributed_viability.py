"""Tests for viability/distributed_viability.py — federated Viability Condition."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viability.distributed_viability import (
    LearnerContribution,
    MergeQuorumPolicy,
    assess_federated,
)


def _healthy_learner(learner_id: str, sessions: float = 50.0) -> LearnerContribution:
    """A learner that should always verify and contribute positively."""
    return LearnerContribution(
        learner_id=learner_id,
        sessions_per_round=sessions,
        avg_turns_per_session=6.0,
        consent_grant_rate=0.9,
        quantization_hostility=0.05,
        deployment_scale_factor=1.0,
        is_verified=True,
    )


def _rejected_learner(learner_id: str, reason: str = "test") -> LearnerContribution:
    return LearnerContribution(
        learner_id=learner_id,
        sessions_per_round=20.0,
        quantization_hostility=0.5,   # rejected learners contribute their hostility to E
        is_verified=False,
        rejection_reason=reason,
    )


# ── Basic single-learner-style smoke ────────────────────────────────────────

def test_single_healthy_learner_below_quorum_fails():
    """A single verified learner cannot meet the default K=3 quorum."""
    result = assess_federated([_healthy_learner("solo")])
    assert result.num_learners_verified == 1
    assert not result.quorum_met
    assert result.round_recommendation == "rollback"


def test_three_healthy_learners_meet_quorum_and_commit():
    """Three healthy learners hit default K=3 and the round should commit."""
    learners = [_healthy_learner(f"clinic-{i}") for i in range(3)]
    result = assess_federated(learners)
    assert result.num_learners_verified == 3
    assert result.num_learners_rejected == 0
    assert result.quorum_met
    assert result.viable_global
    assert result.round_recommendation == "commit"


# ── Rejection handling ──────────────────────────────────────────────────────

def test_rejection_excludes_ceff_but_keeps_e():
    """A rejected learner should reduce Ceff_global but its E still counts toward E_global."""
    learners = [
        _healthy_learner("c1"),
        _healthy_learner("c2"),
        _healthy_learner("c3"),
        _rejected_learner("c4-bad", reason="merkle_root_mismatch"),
    ]
    result = assess_federated(learners)
    assert result.num_learners_verified == 3
    assert result.num_learners_rejected == 1
    assert "c4-bad" in result.rejected_learners
    # E_global should be dominated by the rejected learner's E (0.5) plus merge error
    assert result.e_global >= 0.5
    # Recommendation should be alert_operator even though quorum is met and Ceff > E
    assert result.round_recommendation == "alert_operator"


def test_all_learners_rejected_triggers_rollback():
    learners = [_rejected_learner(f"bad-{i}") for i in range(5)]
    result = assess_federated(learners)
    assert result.num_learners_verified == 0
    assert not result.quorum_met
    assert result.round_recommendation == "rollback"


# ── Quorum policy ───────────────────────────────────────────────────────────

def test_custom_quorum_minimum_enforced():
    """A stricter quorum (K=10) should reject a 5-learner round."""
    policy = MergeQuorumPolicy(minimum_quorum=10)
    learners = [_healthy_learner(f"c-{i}") for i in range(5)]
    result = assess_federated(learners, policy=policy)
    assert result.num_learners_verified == 5
    assert not result.quorum_met
    assert result.round_recommendation == "rollback"


def test_merge_error_scales_inverse_sqrt_k():
    """merge_error should decrease as K grows (1/√K)."""
    policy = MergeQuorumPolicy(minimum_quorum=2, merge_error_floor=0.0)
    err_k4 = policy.merge_error(4)
    err_k16 = policy.merge_error(16)
    err_k100 = policy.merge_error(100)
    # √4 = 2 → 0.5; √16 = 4 → 0.25; √100 = 10 → 0.1
    assert abs(err_k4 - 0.5) < 1e-9
    assert abs(err_k16 - 0.25) < 1e-9
    assert abs(err_k100 - 0.1) < 1e-9


def test_merge_error_floor_enforced():
    """At very high K, the floor should clamp merge_error from going below floor."""
    policy = MergeQuorumPolicy(minimum_quorum=2, merge_error_floor=0.05)
    # √10000 = 100 → 0.01, but floor = 0.05 should clamp
    assert policy.merge_error(10000) == 0.05


# ── Synthetic data ratio ────────────────────────────────────────────────────

def test_synthetic_data_ratio_reduces_ceff():
    """Synthetic data should reduce effective Ceff."""
    learners = [_healthy_learner(f"c-{i}") for i in range(5)]
    clean = assess_federated(learners, synthetic_data_ratio=0.0)
    polluted = assess_federated(learners, synthetic_data_ratio=0.5)
    assert polluted.ceff_global == clean.ceff_global * 0.5


# ── Per-learner report ──────────────────────────────────────────────────────

def test_per_learner_assessment_includes_all_learners():
    """The per-learner report must include verified AND rejected learners."""
    learners = [
        _healthy_learner("good-1"),
        _healthy_learner("good-2"),
        _healthy_learner("good-3"),
        _rejected_learner("bad-1"),
    ]
    result = assess_federated(learners)
    assert set(result.per_learner_assessment.keys()) == {"good-1", "good-2", "good-3", "bad-1"}


# ── Scenario: deforestation 20 stations with 2 failures ─────────────────────

def test_deforestation_scenario_handles_two_station_failures():
    """20 Amazon monitoring stations, 2 fail (one cloud-blackout, one compromised).
    The remaining 18 should clear the K=3 quorum easily, and the round should
    commit even though one was rejected (alert_operator)."""
    learners = []
    for i in range(18):
        learners.append(
            LearnerContribution(
                learner_id=f"station-{i:02d}",
                sessions_per_round=10.0,  # 10 classifications per day per station
                quantization_hostility=0.08,
                is_verified=True,
            )
        )
    # Two failures: one rejected (compromised), one with no Ceff contribution
    # (cloud blackout — still verified but zero sessions)
    learners.append(
        LearnerContribution(
            learner_id="station-18-cloudout",
            sessions_per_round=0.0,
            quantization_hostility=0.08,
            is_verified=True,
        )
    )
    learners.append(
        _rejected_learner("station-19-compromised", reason="shape_mismatch")
    )

    result = assess_federated(learners)
    assert result.num_learners_total == 20
    assert result.num_learners_verified == 19   # 18 + the cloudout still counts as verified
    assert result.num_learners_rejected == 1
    assert result.quorum_met
    # Should alert_operator because there was a rejection
    assert result.round_recommendation == "alert_operator"
    # Merge error at K=19 should be modest (~0.23)
    assert 0.20 < result.merge_error_estimate < 0.25
