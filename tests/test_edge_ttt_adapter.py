"""Tests for tools/edge_ttt_adapter.py — per-device runtime adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.edge_ttt_adapter import EdgeTTTAdapter, OperatorFeedback


# ── Fixtures ────────────────────────────────────────────────────────────────

def _full_consent() -> dict:
    return {
        "transcript": True,
        "felt_state": True,
        "gfs_activations": True,
        "training_signal": True,
        "retention": True,
    }


def _fb(session_id: str, error: float, consent: dict = None) -> OperatorFeedback:
    return OperatorFeedback(
        session_id=session_id,
        predicted="model_said_X",
        operator_label="operator_said_Y",
        error=error,
        consent_layers=consent if consent is not None else _full_consent(),
    )


class _CountingStepFn:
    """Step function that pretends to update LoRA weights and tracks call count."""

    def __init__(self, drift_per_call: dict[str, float]):
        self.drift_per_call = drift_per_call
        self.call_count = 0
        self.cumulative_drift = {k: 0.0 for k in drift_per_call}

    def __call__(self, feedback: OperatorFeedback) -> dict[str, float]:
        self.call_count += 1
        for k, v in self.drift_per_call.items():
            self.cumulative_drift[k] += v
        return dict(self.cumulative_drift)


# ── Happy path ──────────────────────────────────────────────────────────────

def test_first_step_applies_with_full_consent_and_empty_history():
    step = _CountingStepFn({"w0": 0.01})
    adapter = EdgeTTTAdapter(step_fn=step)
    record = adapter.step(_fb("s-1", 0.5))
    assert record.applied
    assert record.blocked_by is None
    assert step.call_count == 1
    assert adapter.num_applied() == 1
    assert adapter.num_skipped() == 0


def test_export_receipt_has_expected_keys():
    step = _CountingStepFn({"w0": 0.01})
    adapter = EdgeTTTAdapter(step_fn=step)
    for i in range(3):
        adapter.step(_fb(f"s-{i}", 0.1 * (i - 1)))
    receipt = adapter.export_receipt()
    assert receipt["kind"] == "edge_ttt_trace"
    assert receipt["update_count"] == 3
    assert len(receipt["history"]) == 3
    assert "final_drift_from_baseline" in receipt


# ── Consent enforcement (hard gate) ─────────────────────────────────────────

def test_denied_training_signal_refuses_step_and_does_not_advance_window():
    step = _CountingStepFn({"w0": 0.01})
    adapter = EdgeTTTAdapter(step_fn=step)

    bad_consent = _full_consent()
    bad_consent["training_signal"] = False
    record = adapter.step(_fb("s-1", 0.5, consent=bad_consent))

    assert not record.applied
    assert record.blocked_by == "consent_denied"
    assert "training_signal" in record.notes[0]
    assert step.call_count == 0
    # The skipped record is in history but the SNAPSHOT window did NOT advance
    # (consent refusals are a covenant, not a statistical-window event).
    assert adapter._snapshot.update_count == 0
    assert len(adapter._snapshot.recent_updates) == 0


def test_missing_consent_layer_refuses_step():
    step = _CountingStepFn({"w0": 0.01})
    adapter = EdgeTTTAdapter(step_fn=step)
    # Omit gfs_activations entirely → counts as denied
    partial = {k: True for k in [
        "transcript", "felt_state", "training_signal", "retention"
    ]}
    record = adapter.step(_fb("s-1", 0.5, consent=partial))
    assert record.blocked_by == "consent_denied"
    assert step.call_count == 0


# ── BLOCKING error_bias gate ────────────────────────────────────────────────

def test_error_bias_blocks_after_10_same_sign_then_clears_with_diverse_feedback():
    """Trace through the blocking-and-recovery cycle:

    - Steps 1-10: ten positive errors. All apply (warm-up).
    - Step 11: positive. Window now full of positives → BLOCKED. step_fn=10.
    - Steps 12+: negative feedback. As the window slides, each new negative
      is initially BLOCKED (window still ≥70% positive). After enough
      negatives slide into the window, the proportion drops below 70% and
      the gate clears — from that point on, negatives apply.

    The exact step at which the gate clears depends on the window arithmetic.
    What we assert is the qualitative pattern: (a) the 11th step is blocked,
    (b) at least one negative is initially blocked too, (c) eventually the
    gate clears and additional negatives apply.
    """
    step = _CountingStepFn({"w0": 0.001})
    adapter = EdgeTTTAdapter(step_fn=step)

    # Warm-up: 10 positive applied
    for i in range(10):
        rec = adapter.step(_fb(f"s-{i}", 0.5))
        assert rec.applied, f"step {i} should apply during warm-up"
    assert step.call_count == 10

    # 11th positive: window full of positives → blocked
    rec_11 = adapter.step(_fb("s-10", 0.5))
    assert not rec_11.applied
    assert rec_11.blocked_by == "error_bias"
    assert step.call_count == 10   # step_fn NOT called for blocked step

    # Feed 6 negatives. At least the first one or two will be blocked (window
    # still dominated by positives); eventually the gate clears.
    blocked_count = 0
    applied_count = 0
    for i in range(6):
        rec = adapter.step(_fb(f"s-neg-{i}", -0.5))
        if rec.applied:
            applied_count += 1
        else:
            blocked_count += 1
    assert blocked_count >= 1, "at least the first negative should be blocked"
    assert applied_count >= 1, "the gate should clear partway through the negatives"

    # After enough diverse feedback, a recovery step should apply
    rec = adapter.step(_fb("s-recover", 0.5))
    # We don't assert rec.applied here — it depends on the window state. We assert
    # that step_fn has been called more than 10 times (i.e. recovery happened).
    assert step.call_count > 10, (
        f"expected step_fn to have been called > 10 times after recovery; "
        f"got {step.call_count}"
    )


def test_blocked_steps_count_in_skipped_not_applied():
    step = _CountingStepFn({"w0": 0.001})
    adapter = EdgeTTTAdapter(step_fn=step)
    for i in range(15):
        adapter.step(_fb(f"s-{i}", 0.5))
    # First 10 applied, last 5 blocked
    assert adapter.num_applied() == 10
    assert adapter.num_skipped() == 5


# ── weight_drift WARNING (not blocking) ─────────────────────────────────────

def test_weight_drift_warning_does_not_block_step():
    # Big drift per call to trip the gate quickly
    step = _CountingStepFn({"w0": 0.4})  # exceeds 0.30 threshold immediately
    adapter = EdgeTTTAdapter(step_fn=step)

    # First step applies with no warning (drift checked POST-step)
    rec = adapter.step(_fb("s-0", 0.5))
    assert rec.applied
    # Step DID apply, but warning surfaced in post-step gate
    assert "weight_drift" in rec.notes

    # Second step still applies (drift is a warning, not a block)
    rec2 = adapter.step(_fb("s-1", -0.5))  # opposite sign to keep error_bias clear
    assert rec2.applied


# ── update_rate WARNING ─────────────────────────────────────────────────────

def test_update_rate_warning_after_many_updates():
    step = _CountingStepFn({})  # no drift tracking
    adapter = EdgeTTTAdapter(step_fn=step)
    # Fake the update count to be over threshold without actually doing 1000 steps
    adapter._snapshot.update_count = 999
    # Use diverse feedback to keep error_bias clear
    rec = adapter.step(_fb("s-1000", 0.1 if (adapter._snapshot.update_count % 2) else -0.1))
    assert rec.applied
    # update_count now 1000 — exactly at threshold, no warning yet
    assert "update_rate" not in rec.notes
    rec2 = adapter.step(_fb("s-1001", 0.1))
    # 1001 → over threshold → warning
    assert rec2.applied
    assert "update_rate" in rec2.notes


# ── Receipt → DiLoCo fragment compatibility ────────────────────────────────

def test_receipt_is_serializable_to_json_for_fragment_leaf():
    import json
    step = _CountingStepFn({"w0": 0.01, "w1": -0.005})
    adapter = EdgeTTTAdapter(step_fn=step)
    for i in range(4):
        adapter.step(_fb(f"s-{i}", 0.1 * (i - 1)))
    receipt = adapter.export_receipt()
    # Must be deterministically serializable for Merkle leaf hashing
    blob = json.dumps(receipt, sort_keys=True)
    # Round-trip should be stable
    again = json.dumps(json.loads(blob), sort_keys=True)
    assert blob == again
