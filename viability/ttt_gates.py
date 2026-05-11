"""
ttt_gates.py — Three non-compensatory viability gates for runtime adaptation (TTT).

This is the per-adaptation-step filter that complements the Viability Condition
(per-system invariant) and the DiLoCo fragment verifier (per-fragment gate).
Together, they form the runtime grounding loop:

    operator feedback  →  ttt_gates  →  local LoRA step
                             ↓
                         (skipped if error_bias fires)
                             ↓
    accumulated round  →  diloco_fragment_verifier  →  global merge
                             ↓
                         (rejected if shape/consent/merkle fail)
                             ↓
    federated state    →  distributed_viability      →  commit / rollback
                             ↓
                         (rolled back if Ceff_global ≤ E_global)

Ported from SimSat `src/sim/haic/viability.py` (`evaluate_ttt_viability`) and
generalised for any LoRA-adapted edge model in the Gemma4Good scenarios
(clinic laptop, classroom tablet, monitoring station). Three gates:

  - error_bias    (BLOCKING)  — skip update if recent errors share a sign
  - weight_drift  (WARNING)   — log if any weight drifts > 0.30 from baseline
  - update_rate   (WARNING)   — log if cumulative updates exceed 1000

The blocking gate is what prevents systematic operator-feedback bias from
compounding into a runaway adaptation. The warning gates surface drift and
saturation for operator review without halting the loop.

Source: HumanAI Convention SimSat track, viability gates exercise (N=1100
per stream over 3 streams: baseline_clean / drift_one_class / saturation;
gate-fire rates 38.5% / 99.1% / 9.1% respectively).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Thresholds mirror SimSat. They are deliberately conservative — most edge
# deployments should never approach them. If you find yourself relaxing them,
# you almost certainly have a deeper issue with the feedback signal.
MAX_TTT_WEIGHT_DRIFT = 0.30      # max absolute per-weight drift from baseline
MAX_TTT_UPDATE_COUNT = 1000      # recommend manual reset above this count
TTT_BIAS_WINDOW = 10             # number of recent updates to inspect for bias
TTT_BIAS_THRESHOLD = 0.70        # fraction of same-sign errors that trips the gate


@dataclass
class TTTUpdateRecord:
    """One operator-feedback step in the TTT history.

    The `error` is a signed scalar — positive means the model over-predicted
    (operator label was lower), negative means under-predicted. For the
    BLOCKING error_bias gate, the *sign distribution* in the recent window
    is what matters; the magnitude only matters for downstream logging.

    `applied` flags whether the step actually mutated weights (False if a
    gate fired and the update was skipped). Skipped records still advance
    the window — that's the recovery mechanism that lets the gate clear
    when feedback diversifies.
    """

    error: float
    applied: bool = True
    blocked_by: Optional[str] = None
    note: Optional[str] = None


@dataclass
class TTTTrustSnapshot:
    """Snapshot of the runtime adaptation state, fed into the gates."""

    update_count: int = 0
    drift_from_baseline: dict[str, float] = field(default_factory=dict)
    recent_updates: list[TTTUpdateRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "update_count": self.update_count,
            "drift_from_baseline": dict(self.drift_from_baseline),
            "recent_updates": [
                {"error": u.error, "applied": u.applied, "blocked_by": u.blocked_by}
                for u in self.recent_updates
            ],
        }


@dataclass
class TTTGateResult:
    """Per-step gate evaluation result."""

    weight_drift_passed: bool
    update_rate_passed: bool
    error_bias_passed: bool
    blocked: bool                # True if any BLOCKING gate failed
    blocked_by: Optional[str]    # which BLOCKING gate (only error_bias is blocking)
    warnings: list[str]          # which WARNING gates failed
    snapshot_at_eval: dict       # the snapshot used for evaluation, for the receipt

    @property
    def all_passed(self) -> bool:
        return (
            self.weight_drift_passed
            and self.update_rate_passed
            and self.error_bias_passed
        )

    def __repr__(self) -> str:
        if self.blocked:
            return f"TTTGate(BLOCKED by {self.blocked_by!r})"
        if self.warnings:
            return f"TTTGate(passed, warnings={self.warnings!r})"
        return "TTTGate(passed, clean)"


def evaluate_ttt(snapshot: TTTTrustSnapshot) -> TTTGateResult:
    """Evaluate all three TTT gates on the current adaptation snapshot.

    `error_bias` is the BLOCKING gate: when it fires, the caller MUST skip
    the pending adaptation step (don't reinforce the bias direction). The
    skipped record should still be added to `recent_updates` so the window
    advances and the gate eventually clears when feedback diversifies.

    `weight_drift` and `update_rate` are WARNING gates: log them for
    operator review but proceed with the step.
    """
    weight_ok = _gate_weight_drift(snapshot)
    rate_ok = _gate_update_rate(snapshot)
    bias_ok = _gate_error_bias(snapshot)

    warnings: list[str] = []
    if not weight_ok:
        warnings.append("weight_drift")
    if not rate_ok:
        warnings.append("update_rate")

    blocked = not bias_ok
    blocked_by = "error_bias" if blocked else None

    return TTTGateResult(
        weight_drift_passed=weight_ok,
        update_rate_passed=rate_ok,
        error_bias_passed=bias_ok,
        blocked=blocked,
        blocked_by=blocked_by,
        warnings=warnings,
        snapshot_at_eval=snapshot.as_dict(),
    )


# ── Individual gates ────────────────────────────────────────────────────────

def _gate_weight_drift(snapshot: TTTTrustSnapshot) -> bool:
    """WARNING gate: no single weight has drifted > 0.30 from baseline.

    Drift is reported per-weight (e.g. LoRA-delta L2 vs initial-zero state).
    Excessive drift indicates the adaptation has overfit to a short stream
    of operator labels and abandoned the baseline.
    """
    drift = snapshot.drift_from_baseline or {}
    if not drift:
        return True   # no drift data yet → vacuously passes
    for _, delta in drift.items():
        if abs(delta) > MAX_TTT_WEIGHT_DRIFT:
            return False
    return True


def _gate_update_rate(snapshot: TTTTrustSnapshot) -> bool:
    """WARNING gate: cumulative update count ≤ 1000.

    Above 1000 updates between resets, the local model has been adapted
    through many operator sessions and a manual snapshot review is due.
    """
    return snapshot.update_count <= MAX_TTT_UPDATE_COUNT


def _gate_error_bias(snapshot: TTTTrustSnapshot) -> bool:
    """BLOCKING gate: < 70% of the last 10 errors share the same sign.

    Systematic same-sign error means the model is consistently over- or
    under-predicting in one direction. Continuing to update under this
    condition reinforces the bias instead of correcting it. The blocking
    behaviour skips the pending step, advances the window, and gives the
    error distribution a chance to diversify.

    Passes vacuously until the window has 10 entries (warm-up). Also
    passes vacuously if fewer than 3 parseable error values are available,
    to avoid divide-by-zero on garbled feedback streams.
    """
    window = snapshot.recent_updates[-TTT_BIAS_WINDOW:]
    if len(window) < TTT_BIAS_WINDOW:
        return True  # warm-up
    errors = [u.error for u in window]
    if len(errors) < 3:
        return True
    positive = sum(1 for e in errors if e > 0)
    negative = sum(1 for e in errors if e < 0)
    total = len(errors)
    frac_same_sign = max(positive, negative) / total
    return frac_same_sign < TTT_BIAS_THRESHOLD
