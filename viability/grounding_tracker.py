"""
grounding_tracker.py — Track incremental grounding trajectory over time.

Maintains an append-only log of grounding sessions, computes cumulative
C(t), and monitors the Viability Condition trajectory.

Stateless design: fully JSON-serializable, no database dependency.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GroundingSession:
    """Record of one completed grounding session."""
    session_id: str
    timestamp: str                          # ISO 8601
    sft_pair_count: int                     # real extracted pairs
    training_executed: bool                 # True only if gradients ran
    loss_before: Optional[float] = None     # real or None (dry_run)
    loss_after: Optional[float] = None      # real or None (dry_run)
    session_receipt_root: Optional[str] = None
    training_receipt_root: Optional[str] = None
    consent_training_signal: str = "granted"
    steps_executed: int = 0
    error: Optional[str] = None


class GroundingTracker:
    """
    Append-only tracker for incremental grounding sessions.

    Computes cumulative C(t) and Viability Condition trajectory.
    All data comes from real computations — no simulated values.
    """

    def __init__(self, model_id: str = "gemma-4-e2b", e_t: float = 0.9146):
        """
        Args:
            model_id:  identifier for the model being grounded
            e_t:       E(t) estimate from PRISM quantization_hostility
        """
        self.model_id = model_id
        self.e_t = e_t
        self.sessions: list[GroundingSession] = []
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def add_session(self, session: GroundingSession) -> None:
        """
        Append a completed grounding session.

        Only sessions with consent_training_signal == "granted" contribute
        to C(t).  Sessions with denied consent are logged but do not
        increment the corrective bandwidth.
        """
        self.sessions.append(session)

    @property
    def session_count(self) -> int:
        """Total sessions logged (including denied/failed)."""
        return len(self.sessions)

    @property
    def consented_session_count(self) -> int:
        """Sessions where training_signal was granted."""
        return sum(
            1 for s in self.sessions
            if s.consent_training_signal == "granted"
        )

    @property
    def executed_session_count(self) -> int:
        """Sessions where gradient steps actually ran."""
        return sum(1 for s in self.sessions if s.training_executed)

    def cumulative_ceff(self) -> float:
        """
        Total consented SFT pairs contributed so far.

        This is the raw C(t) — the cumulative corrective bandwidth
        from all consented sessions.  Only counts pairs from sessions
        where training_signal was granted.
        """
        return sum(
            s.sft_pair_count
            for s in self.sessions
            if s.consent_training_signal == "granted"
        )

    def viability_trend(self) -> list:
        """
        Per-session Viability Condition trajectory.

        Returns a list of dicts with cumulative C(t) and the ratio
        C(t)/E(t) at each session boundary.  E(t) is held constant
        (justified because PRISM geometry is stable under LoRA updates —
        confirmed by 4 independent adapter audits).

        Note: C(t)/E(t) here represents the cumulative grounding
        investment, not the per-day rate.  For rate-based V(t), divide
        by the time span in days.
        """
        trend = []
        cumulative = 0.0

        for i, s in enumerate(self.sessions):
            if s.consent_training_signal == "granted":
                cumulative += s.sft_pair_count

            ratio = cumulative / max(self.e_t, 1e-9)

            entry = {
                "session_idx": i,
                "session_id": s.session_id,
                "timestamp": s.timestamp,
                "cumulative_ceff": cumulative,
                "ratio_ceff_e": round(ratio, 4),
                "training_executed": s.training_executed,
                "loss_before": s.loss_before,
                "loss_after": s.loss_after,
            }
            trend.append(entry)

        return trend

    def monotonically_improving(self) -> bool:
        """
        Check if cumulative C(t) is strictly non-decreasing.

        This must always be True by construction — sessions only add
        to C(t), never subtract.  If this returns False, something is
        wrong with the tracking logic.
        """
        cumulative = 0.0
        for s in self.sessions:
            if s.consent_training_signal == "granted":
                cumulative += s.sft_pair_count
            if s.sft_pair_count < 0:
                return False
        return True

    def summary(self) -> dict:
        """Human-readable summary of grounding state."""
        cum_ceff = self.cumulative_ceff()
        return {
            "model_id": self.model_id,
            "e_t": self.e_t,
            "total_sessions": self.session_count,
            "consented_sessions": self.consented_session_count,
            "executed_sessions": self.executed_session_count,
            "cumulative_ceff": cum_ceff,
            "cumulative_ratio": round(cum_ceff / max(self.e_t, 1e-9), 4),
            "monotonically_improving": self.monotonically_improving(),
            "created_at": self.created_at,
        }

    # ── Serialization ─────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "model_id": self.model_id,
            "e_t": self.e_t,
            "created_at": self.created_at,
            "sessions": [asdict(s) for s in self.sessions],
        }
        return json.dumps(data, indent=2, allow_nan=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "GroundingTracker":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        tracker = cls(model_id=data["model_id"], e_t=data["e_t"])
        tracker.created_at = data.get("created_at", tracker.created_at)
        for sd in data.get("sessions", []):
            tracker.sessions.append(GroundingSession(**sd))
        return tracker
