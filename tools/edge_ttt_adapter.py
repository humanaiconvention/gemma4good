"""
edge_ttt_adapter.py — Per-device runtime adaptation under viability gates.

Wraps the local LoRA-step loop on an edge device (clinic laptop, classroom
tablet, monitoring station) so that:

  1. Operator feedback enters through a structured signed-error contract.
  2. The three TTT gates from `viability/ttt_gates.py` are evaluated PRE-step.
  3. If the BLOCKING `error_bias` gate fires, the step is skipped and the
     window advances (so the gate can clear on subsequent diverse feedback).
  4. After the step, drift and rate gates are evaluated POST-step as warnings.
  5. The accumulated runtime trace is exportable as a per-session receipt,
     which gets folded into the next DiLoCo round receipt.

This is the runtime grounding loop's per-step layer. It composes with:
  - tools/diloco_fragment_verifier.py  (per-round filter on accumulated deltas)
  - viability/distributed_viability.py (per-federation viability assessment)
  - viability/viability_condition.py   (per-system invariant Ceff > E)

The actual gradient step is performed by an injected `step_fn` callback — this
module is gradient-framework agnostic, so it tests cleanly without torch/peft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from viability.ttt_gates import (
    TTTGateResult,
    TTTTrustSnapshot,
    TTTUpdateRecord,
    evaluate_ttt,
)


@dataclass
class OperatorFeedback:
    """One operator-labelled adaptation request.

    Attributes
    ----------
    session_id : str
        Maestro session that produced this feedback.
    predicted : str
        Model's prediction (action label, score, or free text).
    operator_label : str
        Operator's corrected label.
    error : float
        Signed scalar mapping prediction → operator. Positive = model
        over-predicted; negative = model under-predicted. Zero = matched.
    consent_layers : dict[str, bool]
        Five-layer consent dict (transcript, felt_state, gfs_activations,
        training_signal, retention). All must be True for the feedback to
        enter the adaptation loop.
    """

    session_id: str
    predicted: str
    operator_label: str
    error: float
    consent_layers: dict[str, bool] = field(default_factory=dict)


# Step function signature: takes the operator feedback, runs forward+backward+
# step on the local model, returns a dict of per-weight L2 deltas (the
# weight_drift gate consumes these). Implementations live downstream
# (e.g. in maestro/legacy_mvp or a notebook cell that wraps peft).
StepFn = Callable[[OperatorFeedback], dict[str, float]]


@dataclass
class StepRecord:
    """One adaptation step's full receipt — what happened, why, and the gate state."""

    session_id: str
    error: float
    applied: bool
    blocked_by: Optional[str]
    gate_result: TTTGateResult
    weight_deltas_after: dict[str, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class EdgeTTTAdapter:
    """Per-edge-device runtime adaptation loop.

    Usage:
        adapter = EdgeTTTAdapter(step_fn=my_lora_step, baseline_weights=...)
        record = adapter.step(operator_feedback)
        if record.blocked_by:
            log.warning(f"step skipped: {record.blocked_by}")

    The adapter maintains state across calls (update count, recent updates
    window, cumulative drift). Reset by constructing a new instance — at
    which point you also typically snapshot weights as the new baseline.
    """

    step_fn: StepFn
    baseline_weights: dict[str, float] = field(default_factory=dict)
    required_consent_layers: tuple[str, ...] = (
        "transcript", "felt_state", "gfs_activations",
        "training_signal", "retention",
    )

    # ── runtime state (private) ────────────────────────────────────────────
    _snapshot: TTTTrustSnapshot = field(default_factory=TTTTrustSnapshot)
    _history: list[StepRecord] = field(default_factory=list)

    def step(self, feedback: OperatorFeedback) -> StepRecord:
        """Process one operator-feedback step under the viability gates.

        Order of operations:
          1. Consent check (hard gate — if any required layer is denied,
             refuse the step entirely; this is a separate, deterministic
             rejection from the statistical TTT gates).
          2. Evaluate TTT gates PRE-step on the current snapshot.
          3. If error_bias is blocked: record a skipped update, advance the
             window, do NOT call step_fn.
          4. If clear: call step_fn, update drift from returned deltas,
             increment counter, record applied update.
          5. Re-evaluate gates POST-step for the receipt (warnings only).
        """
        # 1. Consent check
        denied = [
            layer for layer in self.required_consent_layers
            if not feedback.consent_layers.get(layer, False)
        ]
        if denied:
            return self._refuse_for_consent(feedback, denied)

        # 2. PRE-step gate evaluation
        pre_result = evaluate_ttt(self._snapshot)

        if pre_result.blocked:
            # 3. Skipped step — still record so the window advances
            skipped = TTTUpdateRecord(
                error=feedback.error,
                applied=False,
                blocked_by=pre_result.blocked_by,
            )
            self._snapshot.recent_updates.append(skipped)
            record = StepRecord(
                session_id=feedback.session_id,
                error=feedback.error,
                applied=False,
                blocked_by=pre_result.blocked_by,
                gate_result=pre_result,
                weight_deltas_after=dict(self._snapshot.drift_from_baseline),
                notes=[f"step skipped by BLOCKING gate {pre_result.blocked_by}"],
            )
            self._history.append(record)
            return record

        # 4. Apply the gradient step
        new_drifts = self.step_fn(feedback)
        # Update the rolling drift state — for each reported weight, store
        # its absolute delta from baseline.
        for weight_id, drift in new_drifts.items():
            self._snapshot.drift_from_baseline[weight_id] = drift

        applied = TTTUpdateRecord(
            error=feedback.error,
            applied=True,
            blocked_by=None,
        )
        self._snapshot.recent_updates.append(applied)
        self._snapshot.update_count += 1

        # 5. POST-step gate re-eval for the receipt (warnings only)
        post_result = evaluate_ttt(self._snapshot)
        record = StepRecord(
            session_id=feedback.session_id,
            error=feedback.error,
            applied=True,
            blocked_by=None,
            gate_result=post_result,
            weight_deltas_after=dict(self._snapshot.drift_from_baseline),
            notes=list(post_result.warnings),
        )
        self._history.append(record)
        return record

    def export_receipt(self) -> dict:
        """Export the full TTT trace for this session/round for the next DiLoCo fragment receipt.

        Shape is designed to be a Merkle leaf in a fragment receipt — sorted-keys
        JSON of this dict produces a stable hash.
        """
        return {
            "kind": "edge_ttt_trace",
            "update_count": self._snapshot.update_count,
            "history": [
                {
                    "session_id": r.session_id,
                    "error": r.error,
                    "applied": r.applied,
                    "blocked_by": r.blocked_by,
                    "warnings": list(r.gate_result.warnings),
                    "weight_drift_max": (
                        max((abs(v) for v in r.weight_deltas_after.values()),
                            default=0.0)
                    ),
                }
                for r in self._history
            ],
            "final_drift_from_baseline": dict(self._snapshot.drift_from_baseline),
        }

    def num_applied(self) -> int:
        return sum(1 for r in self._history if r.applied)

    def num_skipped(self) -> int:
        return sum(1 for r in self._history if not r.applied)

    # ── private helpers ────────────────────────────────────────────────────

    def _refuse_for_consent(self, feedback: OperatorFeedback, denied: list[str]) -> StepRecord:
        """Hard refusal: consent denied means the operator feedback should
        never enter the adaptation loop. This is distinct from the BLOCKING
        TTT gate — consent is not a statistical filter, it's a covenant.
        We do NOT advance the window for consent refusals.
        """
        sorted_denied = sorted(denied)
        # Produce a synthetic gate result reflecting the refusal — gates not
        # evaluated, since we never even got to the step.
        synthetic = TTTGateResult(
            weight_drift_passed=True,
            update_rate_passed=True,
            error_bias_passed=True,
            blocked=True,
            blocked_by="consent_denied",
            warnings=[],
            snapshot_at_eval=self._snapshot.as_dict(),
        )
        record = StepRecord(
            session_id=feedback.session_id,
            error=feedback.error,
            applied=False,
            blocked_by="consent_denied",
            gate_result=synthetic,
            weight_deltas_after=dict(self._snapshot.drift_from_baseline),
            notes=[f"consent denied for layers: {sorted_denied}"],
        )
        self._history.append(record)
        return record
