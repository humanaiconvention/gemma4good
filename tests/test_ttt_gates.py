"""Tests for viability/ttt_gates.py — three runtime adaptation gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viability.ttt_gates import (
    MAX_TTT_UPDATE_COUNT,
    MAX_TTT_WEIGHT_DRIFT,
    TTT_BIAS_THRESHOLD,
    TTT_BIAS_WINDOW,
    TTTTrustSnapshot,
    TTTUpdateRecord,
    evaluate_ttt,
)


def _record(error: float, applied: bool = True) -> TTTUpdateRecord:
    return TTTUpdateRecord(error=error, applied=applied)


# ── Empty / warm-up state ───────────────────────────────────────────────────

def test_empty_snapshot_passes_all_gates_vacuously():
    snap = TTTTrustSnapshot()
    result = evaluate_ttt(snap)
    assert result.all_passed
    assert not result.blocked


def test_partial_window_passes_error_bias_vacuously():
    """Less than 10 entries → error_bias passes (warm-up)."""
    snap = TTTTrustSnapshot(recent_updates=[_record(1.0) for _ in range(9)])
    result = evaluate_ttt(snap)
    assert result.error_bias_passed
    assert not result.blocked


# ── error_bias (BLOCKING) ───────────────────────────────────────────────────

def test_error_bias_fires_when_all_recent_same_sign():
    """10 positive errors in a row → error_bias fires, step is blocked."""
    snap = TTTTrustSnapshot(recent_updates=[_record(0.5) for _ in range(10)])
    result = evaluate_ttt(snap)
    assert not result.error_bias_passed
    assert result.blocked
    assert result.blocked_by == "error_bias"


def test_error_bias_fires_at_threshold_exactly():
    """7 of 10 same-sign hits the 70% threshold exactly → fires."""
    errors = [0.5] * 7 + [-0.5] * 3
    snap = TTTTrustSnapshot(recent_updates=[_record(e) for e in errors])
    result = evaluate_ttt(snap)
    assert not result.error_bias_passed


def test_error_bias_passes_at_60_percent():
    """6 of 10 same-sign is below 70% threshold → passes."""
    errors = [0.5] * 6 + [-0.5] * 4
    snap = TTTTrustSnapshot(recent_updates=[_record(e) for e in errors])
    result = evaluate_ttt(snap)
    assert result.error_bias_passed
    assert not result.blocked


def test_error_bias_only_inspects_last_window_entries():
    """An old streak of same-sign errors should not fire if the recent window
    has diversified."""
    # 20 positive, then 10 with 4 negative → recent window passes
    old = [_record(0.5) for _ in range(20)]
    recent = [_record(0.5)] * 6 + [_record(-0.5)] * 4
    snap = TTTTrustSnapshot(recent_updates=old + recent)
    result = evaluate_ttt(snap)
    assert result.error_bias_passed


def test_error_bias_window_advances_on_blocked_records():
    """Blocked records should also contribute to the window (so the gate clears
    after diverse feedback). Apply 10 same-sign → blocked, then 5 opposite →
    window now 5-5 split, gate clears."""
    snap = TTTTrustSnapshot()
    # Fill window with positive errors → block
    for _ in range(10):
        snap.recent_updates.append(_record(0.5, applied=False))
    assert evaluate_ttt(snap).blocked
    # Now add 6 negative errors — window slides to mix
    for _ in range(6):
        snap.recent_updates.append(_record(-0.5))
    # Last 10: 4 positive + 6 negative = 60% same-sign, below threshold
    assert evaluate_ttt(snap).error_bias_passed


# ── weight_drift (WARNING) ──────────────────────────────────────────────────

def test_weight_drift_passes_when_under_threshold():
    snap = TTTTrustSnapshot(drift_from_baseline={"w0": 0.1, "w1": -0.2, "w2": 0.05})
    result = evaluate_ttt(snap)
    assert result.weight_drift_passed
    assert "weight_drift" not in result.warnings


def test_weight_drift_fires_when_any_weight_over_threshold():
    snap = TTTTrustSnapshot(drift_from_baseline={"w0": 0.1, "w1": 0.31, "w2": 0.05})
    result = evaluate_ttt(snap)
    assert not result.weight_drift_passed
    assert "weight_drift" in result.warnings
    # But does NOT block — weight_drift is a warning, not blocking
    assert not result.blocked


def test_weight_drift_negative_delta_treated_as_absolute():
    snap = TTTTrustSnapshot(drift_from_baseline={"w0": -0.35})
    result = evaluate_ttt(snap)
    assert not result.weight_drift_passed


def test_weight_drift_empty_passes_vacuously():
    snap = TTTTrustSnapshot()
    result = evaluate_ttt(snap)
    assert result.weight_drift_passed


# ── update_rate (WARNING) ───────────────────────────────────────────────────

def test_update_rate_passes_under_max():
    snap = TTTTrustSnapshot(update_count=999)
    result = evaluate_ttt(snap)
    assert result.update_rate_passed


def test_update_rate_passes_at_max():
    snap = TTTTrustSnapshot(update_count=MAX_TTT_UPDATE_COUNT)
    result = evaluate_ttt(snap)
    assert result.update_rate_passed


def test_update_rate_fires_over_max():
    snap = TTTTrustSnapshot(update_count=MAX_TTT_UPDATE_COUNT + 1)
    result = evaluate_ttt(snap)
    assert not result.update_rate_passed
    assert "update_rate" in result.warnings
    assert not result.blocked   # warning, not blocking


# ── Combined behaviour ──────────────────────────────────────────────────────

def test_all_three_gates_can_fire_simultaneously():
    snap = TTTTrustSnapshot(
        update_count=2000,
        drift_from_baseline={"w0": 0.5},
        recent_updates=[_record(0.5) for _ in range(10)],
    )
    result = evaluate_ttt(snap)
    assert not result.weight_drift_passed
    assert not result.update_rate_passed
    assert not result.error_bias_passed
    assert result.blocked
    assert result.blocked_by == "error_bias"
    # Warnings still surface both non-blocking failures
    assert set(result.warnings) == {"weight_drift", "update_rate"}


def test_thresholds_match_simsat_reference():
    """Sanity-check the threshold constants match the SimSat reference."""
    assert MAX_TTT_WEIGHT_DRIFT == 0.30
    assert MAX_TTT_UPDATE_COUNT == 1000
    assert TTT_BIAS_WINDOW == 10
    assert TTT_BIAS_THRESHOLD == 0.70
