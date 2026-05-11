"""Tests for viability/session_gates.py — six convention-session viability gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viability.session_gates import (
    ENTROPY_EPSILON,
    EXTRACTION_RISK_LIMIT,
    MAX_TURN_CHAR_LENGTH,
    MIN_PROVENANCE_SCORE,
    MIN_USER_TURNS,
    MIN_VOCAB_DIVERSITY,
    MIN_WORD_COUNT,
    GroundingSessionView,
    evaluate_session,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _good_session(**overrides) -> GroundingSessionView:
    """A session that should pass all six gates."""
    base = dict(
        session_id="s-good-1",
        interview_turns=[
            {"role": "assistant", "content": "What did you experience this morning?"},
            {"role": "user", "content": "I felt grounded after my walk in the forest"},
            {"role": "assistant", "content": "Can you describe the specific sensations?"},
            {"role": "user", "content": "There was cool damp air, scent of pine resin, dappled sunlight on moss"},
            {"role": "assistant", "content": "What memory does this surface?"},
            {"role": "user", "content": "Walking with my grandfather as a child collecting mushrooms"},
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
    base.update(overrides)
    return GroundingSessionView(**base)


# ── Happy path ──────────────────────────────────────────────────────────────

def test_good_session_passes_all_six_gates():
    result = evaluate_session(_good_session())
    assert result.all_passed
    assert result.num_passed == 6
    assert result.failure_reasons == []


# ── Gate 1: entropy_reduction ───────────────────────────────────────────────

def test_no_prism_delta_passes_gate_1_vacuously():
    session = _good_session(entropy_delta=None)
    result = evaluate_session(session)
    assert result.entropy_reduction


def test_insufficient_entropy_reduction_fails_gate_1():
    session = _good_session(entropy_delta={
        "delta_spectral_entropy": -0.005,  # below ENTROPY_EPSILON in magnitude
        "reduction_verified": False,
        "snapshot_before": {"mean_spectral_entropy": 1.0},
        "snapshot_after": {"mean_spectral_entropy": 0.995},
    })
    result = evaluate_session(session)
    assert not result.entropy_reduction
    assert any("entropy_reduction" in r for r in result.failure_reasons)


def test_reduction_verified_flag_overrides_threshold():
    """If Prism flags reduction_verified=True, the gate passes even if delta is small."""
    session = _good_session(entropy_delta={
        "delta_spectral_entropy": -0.001,
        "reduction_verified": True,
        "snapshot_before": {"mean_spectral_entropy": 1.0},
        "snapshot_after": {"mean_spectral_entropy": 0.999},
    })
    result = evaluate_session(session)
    assert result.entropy_reduction


# ── Gate 2: extraction_risk ─────────────────────────────────────────────────

def test_normal_session_below_extraction_risk_limit():
    result = evaluate_session(_good_session())
    assert result.extraction_risk
    assert result.extraction_risk_score < EXTRACTION_RISK_LIMIT


def test_oversized_transcript_increases_extraction_risk():
    """Long transcripts + image scraping pattern + no PoG + extra factor → risk > limit."""
    long_content = "word " * 30000  # ~150k chars
    session = _good_session(
        interview_turns=[
            {"role": "user", "content": long_content},  # 1 user turn (image-scrape pattern)
        ],
        image_count=5,
        pog_provenance_score=None,
        extra_risk_factors={"repeat_offender": 0.05},
    )
    result = evaluate_session(session)
    # 0.05 (>10k chars) + 0.05 (>50k chars) + 0.08 (image_count>2 AND user_turn_count<2)
    #   + 0.02 (no PoG) + 0.05 (extra) = 0.25 > 0.15 threshold
    assert not result.extraction_risk


def test_image_scraping_pattern_increases_risk():
    """5 images + only 1 user turn → image-scraping pattern → risk bumped."""
    session = _good_session(
        interview_turns=[
            {"role": "user", "content": "show me everything you have on this"},
        ],
        image_count=5,
        pog_provenance_score=None,   # additional 0.02 risk
        extra_risk_factors={"suspicious_ip": 0.10},
    )
    result = evaluate_session(session)
    assert not result.extraction_risk


# ── Gate 3: prism_consistency ───────────────────────────────────────────────

def test_prism_consistency_fails_on_claim_actual_mismatch():
    """If claimed delta doesn't match the snapshot arithmetic, gate fails."""
    session = _good_session(entropy_delta={
        "delta_spectral_entropy": -0.5,
        "reduction_verified": False,
        "snapshot_before": {"mean_spectral_entropy": 1.0},
        "snapshot_after": {"mean_spectral_entropy": 0.95},  # actual delta = -0.05, claimed = -0.5
    })
    result = evaluate_session(session)
    assert not result.prism_consistency


def test_prism_consistency_missing_snapshots_fails():
    session = _good_session(entropy_delta={
        "delta_spectral_entropy": -0.05,
        # no snapshots
    })
    result = evaluate_session(session)
    assert not result.prism_consistency


# ── Gate 4: participation_covenant ──────────────────────────────────────────

def test_missing_stimulus_fails_gate_4():
    session = _good_session(has_stimulus=False)
    result = evaluate_session(session)
    assert not result.participation_covenant


def test_low_provenance_score_fails_gate_4():
    session = _good_session(pog_provenance_score=0.80)
    result = evaluate_session(session)
    assert not result.participation_covenant
    assert any("provenance" in r for r in result.failure_reasons)


def test_too_few_user_turns_fails_gate_4():
    session = _good_session(interview_turns=[
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Hi sufficient words here for the count"},
    ])
    result = evaluate_session(session)
    assert not result.participation_covenant
    assert any("user_turns" in r for r in result.failure_reasons)


def test_too_few_words_fails_gate_4():
    session = _good_session(interview_turns=[
        {"role": "assistant", "content": "?"},
        {"role": "user", "content": "yes"},
        {"role": "assistant", "content": "?"},
        {"role": "user", "content": "ok"},
    ])
    result = evaluate_session(session)
    assert not result.participation_covenant


def test_no_pog_score_does_not_fail_gate_4():
    """A None provenance score should NOT fail the gate — it just isn't checked."""
    session = _good_session(pog_provenance_score=None)
    result = evaluate_session(session)
    # Gate 4 should still pass (turns and words are good)
    assert result.participation_covenant
    # But gate 2 (extraction_risk) is bumped by 0.02 for missing PoG


# ── Gate 5: federated_exchange ──────────────────────────────────────────────

def test_image_blob_in_turn_fails_gate_5():
    session = _good_session(interview_turns=[
        {"role": "assistant", "content": "What did you see?"},
        {"role": "user", "content": "data:image/png;base64,iVBOR..."},
        {"role": "assistant", "content": "Anything else?"},
        {"role": "user", "content": "Just felt grounded after my walk in the forest with grandfather"},
    ])
    result = evaluate_session(session)
    assert not result.federated_exchange


def test_oversized_turn_fails_gate_5():
    session = _good_session(interview_turns=[
        {"role": "assistant", "content": "?"},
        {"role": "user", "content": "x" * (MAX_TURN_CHAR_LENGTH + 1)},
        {"role": "assistant", "content": "?"},
        {"role": "user", "content": "valid response with several distinct words"},
    ])
    result = evaluate_session(session)
    assert not result.federated_exchange


# ── Gate 6: epistemic_alignment ─────────────────────────────────────────────

def test_repeated_assistant_turns_fail_gate_6():
    """Half or more of assistant turns identical → fails epistemic_alignment."""
    session = _good_session(interview_turns=[
        {"role": "assistant", "content": "Tell me more."},
        {"role": "user", "content": "first response with diverse vocabulary indeed"},
        {"role": "assistant", "content": "Tell me more."},
        {"role": "user", "content": "second response with even more distinct words here"},
        {"role": "assistant", "content": "Tell me more."},
        {"role": "user", "content": "third response continues the diversity unique words"},
        {"role": "assistant", "content": "Tell me more."},
        {"role": "user", "content": "fourth response adding additional novel terms always"},
    ])
    result = evaluate_session(session)
    assert not result.epistemic_alignment


def test_low_vocab_diversity_fails_gate_6():
    """User repeats the same words → fails epistemic_alignment."""
    session = _good_session(interview_turns=[
        {"role": "assistant", "content": "What did you see?"},
        {"role": "user", "content": "the the the the the the the the the the the the the the the the"},
        {"role": "assistant", "content": "What else?"},
        {"role": "user", "content": "the the the the the the the the the the the the the the the the"},
    ])
    result = evaluate_session(session)
    assert not result.epistemic_alignment
    assert result.vocab_diversity < MIN_VOCAB_DIVERSITY


def test_vocab_diversity_reported_in_result():
    """Even on passing sessions, vocab_diversity should be populated for the audit log."""
    result = evaluate_session(_good_session())
    assert 0.0 < result.vocab_diversity <= 1.0


# ── Composite ───────────────────────────────────────────────────────────────

def test_multiple_gates_can_fail_simultaneously_with_full_report():
    """A degenerate session can fail several gates; all reasons should appear."""
    session = GroundingSessionView(
        session_id="s-bad",
        interview_turns=[
            {"role": "user", "content": "yes"},
        ],
        has_stimulus=False,
        pog_provenance_score=0.5,
        entropy_delta={
            "delta_spectral_entropy": -0.5,
            "snapshot_before": {"mean_spectral_entropy": 1.0},
            "snapshot_after": {"mean_spectral_entropy": 0.99},  # delta mismatch
        },
        image_count=5,
    )
    result = evaluate_session(session)
    assert not result.all_passed
    assert result.num_passed < 6
    # Multiple reasons should be present
    assert len(result.failure_reasons) >= 2


def test_thresholds_match_simsat_reference():
    assert ENTROPY_EPSILON == 0.01
    assert EXTRACTION_RISK_LIMIT == 0.15
    assert MIN_PROVENANCE_SCORE == 0.90
    assert MIN_USER_TURNS == 2
    assert MIN_WORD_COUNT == 10
    assert MIN_VOCAB_DIVERSITY == 0.30
