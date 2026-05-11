"""
distributed_viability.py — Federated Viability Condition for DiLoCo deployments.

Extends viability_condition.py to multi-learner deployments where each learner
contributes a fragment to a sync round. The Viability Condition becomes:

    Ceff_global(r) > E_global(r)

where Ceff_global is summed across verified fragments only (compromised or
unverified fragments are excluded), and E_global incorporates per-learner
error plus the Radial-Directional Averaging merge error scaling as 1/√K.

See docs/diloco_integration_2026-05-11.md for the full framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

from .viability_condition import ViabilityAssessment, assess as assess_single


@dataclass
class LearnerContribution:
    """A single learner's contribution to a sync round.

    Attributes
    ----------
    learner_id : str
        Stable identifier for the learner (e.g. 'clinic-bolivia-03').
    sessions_per_round : float
        Number of Maestro sessions this learner processed in the round.
    avg_turns_per_session : float
        Average turns per session (default 6 matches single-node convention).
    consent_grant_rate : float
        Fraction of sessions with training_signal granted (0..1).
    quantization_hostility : float
        Per-learner E(t) component from Prism geometry.
    deployment_scale_factor : float
        Per-learner deployment scale factor for E(t).
    is_verified : bool
        Did the syncer's verify_fragment() pass for this learner? If False,
        the contribution is excluded from Ceff_global but its E(t) is still
        considered for E_global (we cap on the worst E we saw, not the
        worst accepted E).
    rejection_reason : Optional[str]
        Human-readable rejection reason if is_verified is False.
    """

    learner_id: str
    sessions_per_round: float
    avg_turns_per_session: float = 6.0
    consent_grant_rate: float = 0.85
    quantization_hostility: float = 0.1
    deployment_scale_factor: float = 1.0
    is_verified: bool = True
    rejection_reason: Optional[str] = None

    def ceff_contribution(self) -> float:
        """Local Ceff in turns/round. Zero if not verified."""
        if not self.is_verified:
            return 0.0
        return (
            self.sessions_per_round
            * self.avg_turns_per_session
            * self.consent_grant_rate
        )

    def e_contribution(self) -> float:
        """Local E in dimensionless hostility-units. Counts even if unverified
        because an unverified learner is evidence the system has hostile state
        somewhere — we don't get to ignore it for the error budget."""
        return self.quantization_hostility * self.deployment_scale_factor


@dataclass
class MergeQuorumPolicy:
    """DiLoCo syncer merge-quorum policy.

    minimum_quorum : int
        Minimum K of M learners required to commit a round. Below K, the round
        is rolled back.
    grace_window_seconds : float
        Adaptive grace window: wait this long after quorum K is reached for
        additional learners to report before sealing the round.
    merge_error_floor : float
        Lower bound on the Radial-Directional Averaging merge error contribution
        to E_global. Prevents the 1/√K term from going to zero at very large K.
    """

    minimum_quorum: int = 3
    grace_window_seconds: float = 30.0
    merge_error_floor: float = 0.005

    def merge_error(self, num_accepted: int) -> float:
        """Estimated merge error introduced by averaging num_accepted fragments.

        Returns max(merge_error_floor, 1/√K). At K=5: ~0.45; K=20: ~0.22; K=100: ~0.10.
        Calibration is rough — the exact constant depends on fragment-norm
        variance, which depends on the data distribution across learners.
        """
        if num_accepted <= 0:
            return float("inf")
        return max(self.merge_error_floor, 1.0 / sqrt(num_accepted))


@dataclass
class FederatedViabilityAssessment:
    """Result of a federated Viability Condition evaluation across a sync round."""

    viable_global: bool
    ceff_global: float                 # in turns/round (sum over verified learners)
    e_global: float                    # max E plus merge_error
    num_learners_total: int
    num_learners_verified: int
    num_learners_rejected: int
    quorum_met: bool
    quorum_minimum: int
    merge_error_estimate: float
    rejected_learners: list[str]
    per_learner_assessment: dict[str, ViabilityAssessment]
    round_recommendation: str          # "commit" | "rollback" | "alert_operator"
    single_node_summary: ViabilityAssessment  # the "as if it were one big node" view

    def __repr__(self) -> str:
        status = "VIABLE_GLOBAL" if self.viable_global else "VIOLATED_GLOBAL"
        return (
            f"FederatedViability({status}, "
            f"verified={self.num_learners_verified}/{self.num_learners_total}, "
            f"quorum={'met' if self.quorum_met else 'failed'}, "
            f"action={self.round_recommendation!r})"
        )


def assess_federated(
    contributions: list[LearnerContribution],
    policy: Optional[MergeQuorumPolicy] = None,
    synthetic_data_ratio: float = 0.0,
) -> FederatedViabilityAssessment:
    """Evaluate the federated Viability Condition for one DiLoCo sync round.

    Parameters
    ----------
    contributions : list[LearnerContribution]
        All learners that attempted to report this round. Verified and rejected
        alike — the function classifies them.
    policy : MergeQuorumPolicy, optional
        Syncer merge policy. Defaults to a permissive K=3 policy.
    synthetic_data_ratio : float
        Global synthetic-data ratio (0..1). Reduces Ceff_global by this factor.

    Returns
    -------
    FederatedViabilityAssessment
        Full diagnosis including per-learner breakdown and a recommended action.

    Notes
    -----
    The recommendation logic is intentionally conservative:
      - quorum_failed → "rollback" regardless of viability
      - quorum_met AND viable → "commit"
      - quorum_met AND not viable → "alert_operator" (don't silently commit)
      - quorum_met AND any rejections → also "alert_operator" (review the rejections
        before next round, even if numerically viable)
    """
    if policy is None:
        policy = MergeQuorumPolicy()

    verified = [c for c in contributions if c.is_verified]
    rejected = [c for c in contributions if not c.is_verified]
    num_total = len(contributions)
    num_verified = len(verified)
    num_rejected = len(rejected)
    quorum_met = num_verified >= policy.minimum_quorum

    # Ceff_global: sum over verified learners, scaled by (1 - synthetic_data_ratio).
    raw_ceff = sum(c.ceff_contribution() for c in verified)
    ceff_global = raw_ceff * (1.0 - synthetic_data_ratio)

    # E_global: max over ALL learners (including rejected, since their hostility
    # is evidence of system state), plus merge_error from K-fragment averaging.
    if num_total > 0:
        worst_e = max(c.e_contribution() for c in contributions)
    else:
        worst_e = 0.0
    merge_err = policy.merge_error(num_verified)
    e_global = worst_e + merge_err

    # Viability check
    viable_global = ceff_global > e_global if e_global > 0 else False

    # Per-learner single-node assessments for the report
    per_learner = {}
    for c in contributions:
        per_learner[c.learner_id] = assess_single(
            error_rate_estimate=c.e_contribution(),
            verification_bandwidth_estimate=c.sessions_per_round * c.avg_turns_per_session,
            synthetic_data_ratio=synthetic_data_ratio,
            model_id=c.learner_id,
            prism_hostility=c.quantization_hostility,
        )

    # Recommendation logic
    if not quorum_met:
        rec = "rollback"
    elif viable_global and num_rejected == 0:
        rec = "commit"
    else:
        # Either viability is violated, or we had rejections to investigate.
        rec = "alert_operator"

    # "Single-node equivalent" view: pretend the federation were one big node.
    single_node = assess_single(
        error_rate_estimate=e_global,
        verification_bandwidth_estimate=raw_ceff,
        synthetic_data_ratio=synthetic_data_ratio,
        model_id=f"federated-K{num_verified}-of-{num_total}",
    )

    return FederatedViabilityAssessment(
        viable_global=viable_global,
        ceff_global=round(ceff_global, 4),
        e_global=round(e_global, 6),
        num_learners_total=num_total,
        num_learners_verified=num_verified,
        num_learners_rejected=num_rejected,
        quorum_met=quorum_met,
        quorum_minimum=policy.minimum_quorum,
        merge_error_estimate=round(merge_err, 6),
        rejected_learners=[c.learner_id for c in rejected],
        per_learner_assessment=per_learner,
        round_recommendation=rec,
        single_node_summary=single_node,
    )
