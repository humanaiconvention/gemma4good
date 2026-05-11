"""
enforcement_evidence_contract.py — Structured evidence contract for governance
decisions whose external signal is environmental rather than social.

Ports the SimSat `ObservationEvidence` + 4-action contract (accept · refine
· defer · skip) to the Gemma4Good deforestation scenario. Provides a
model-agnostic, operator-meaningful decision vocabulary for AI systems making
enforcement-consequential judgments about physical-world events (deforestation
flags, structural-damage assessments, illicit-construction detection, etc.).

The contract is intentionally *generic* over the underlying environmental
signal — Sentinel-2 land cover, SAR backscatter, LiDAR, ground-station
photography — so the same governance loop applies regardless of sensor.

This complements the four-tool governance pipeline:
  - assess_wellbeing_domain      → who is affected
  - verify_consent_and_provenance → is the evidence allowed to inform action
  - run_prism_analysis           → is the model's reasoning interpretable
  - generate_alignment_receipt   → cryptographic anchor
The enforcement_evidence_contract adds:
  - structured evidence per observation (8 keys, defined below)
  - 4-action decision vocabulary
  - per-decision audit trail for human review
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from utils.merkle import sha3_256_hex


class EnforcementAction(str, Enum):
    """The four-action contract.

    Identical semantics to SimSat's encounter-triage actions, generalised:

    ACCEPT  — Evidence is sufficient and unambiguous; trigger the enforcement
              action (dispatch reviewer, file report, alert authority). High
              confidence; the cost of false-positive is acceptable.

    REFINE  — Evidence indicates a possible event but the observation is
              insufficient. Schedule additional sensing (next satellite
              pass, drone overflight, ground confirmation) and re-assess.
              Most common action in well-tuned systems.

    DEFER   — Evidence is ambiguous or the case is sensitive in a way that
              requires human judgment. Route to a human reviewer with the
              full evidence trace. Lower volume than refine; higher stakes.

    SKIP    — No usable observation. Cloud cover, sensor failure, occlusion,
              or scene mismatch. No action; not a failure of the system,
              just absence of signal. Should be rare under normal conditions
              but can spike during seasonal weather (cloud blackouts).
    """
    ACCEPT = "accept"
    REFINE = "refine"
    DEFER = "defer"
    SKIP = "skip"


# Threshold guidance for mapping confidence × evidence to action. These are
# defaults; specific deployments tune them per sensor + per geographical context.
ACCEPT_CONFIDENCE_MIN = 0.80
REFINE_CONFIDENCE_MIN = 0.40
DEFER_AMBIGUITY_BAND = (0.40, 0.65)   # confidence in this band routes to human
MAX_OCCLUSION_FOR_ACTION = 0.50       # > 0.50 occlusion = SKIP


@dataclass
class EnforcementEvidence:
    """Eight-key evidence contract for a single observation.

    Modelled directly on SimSat's `ObservationEvidence`. Each field is a
    bounded float or boolean produced by the model from the input
    observation; the contract is the same regardless of what the underlying
    sensor or model is.

    usable_observation:
        Hard boolean. False = cloud cover, occlusion, sensor failure, scene
        mismatch. When False, action MUST be SKIP regardless of other fields.

    scene_match_score [0..1]:
        How well the observation matches the expected scene for the target
        (e.g. correct geographic registration, expected vegetation type for
        the season). Low scores indicate location/registration error.

    salience_score [0..1]:
        How prominent is the candidate event in the observation? Higher =
        bigger contiguous area of detected change, more pixels above
        threshold, etc.

    change_or_event_score [0..1]:
        Strength of the change signal vs the baseline (prior NDVI, prior
        SAR backscatter, prior tile). Higher = more confident there IS an
        event vs sensor noise.

    occlusion_or_cloud_risk [0..1]:
        Fraction of the observation likely obstructed. SimSat: 0.50 was
        the action threshold; higher → SKIP.

    confidence [0..1]:
        Calibrated overall confidence in the model's recommended_action.
        This is the model's own self-assessment, post-calibration.

    rationale_tags : list[str]
        Free-form short tags describing why the model recommended this
        action. Used for human review and for clustering similar decisions.

    raw_observation_id : Optional[str]
        Pointer to the underlying sensor data (file path, S3 key, tile ID).
        Never the data itself — keeps payloads small for syncing.
    """

    usable_observation: bool = False
    scene_match_score: float = 0.0
    salience_score: float = 0.0
    change_or_event_score: float = 0.0
    occlusion_or_cloud_risk: float = 0.0
    confidence: float = 0.0
    rationale_tags: list[str] = field(default_factory=list)
    raw_observation_id: Optional[str] = None

    def as_leaf_hash(self) -> str:
        """SHA3-256 leaf hash for inclusion in a session/round Merkle tree."""
        import json
        return sha3_256_hex(json.dumps(asdict(self), sort_keys=True))


@dataclass
class EnforcementAssessment:
    """One model assessment: evidence + recommended action + provenance."""

    assessment_id: str
    target_id: str                       # what's being assessed (geo cell, target object)
    scenario_pack: str                   # e.g. "amazon_deforestation"
    model_id: str
    created_at: str                      # ISO 8601
    evidence: EnforcementEvidence
    recommended_action: EnforcementAction
    confidence_adjustment: float = 0.0   # post-hoc calibration delta, if any
    raw_response_text: Optional[str] = None

    def as_leaf_hash(self) -> str:
        """Stable Merkle leaf hash for the receipt chain."""
        import json
        payload = {
            "assessment_id": self.assessment_id,
            "target_id": self.target_id,
            "scenario_pack": self.scenario_pack,
            "model_id": self.model_id,
            "evidence": asdict(self.evidence),
            "recommended_action": self.recommended_action.value,
            "confidence_adjustment": self.confidence_adjustment,
        }
        return sha3_256_hex(json.dumps(payload, sort_keys=True))


def derive_action(evidence: EnforcementEvidence) -> tuple[EnforcementAction, list[str]]:
    """Map evidence → action using the SimSat threshold convention.

    Returns (action, reason_tags). The reason tags explain why this action
    was chosen — useful for the audit log and for human review.

    Ordering of checks (first match wins):
      1. Not usable → SKIP
      2. Occlusion too high → SKIP
      3. Confidence ambiguity band → DEFER
      4. Confidence high + change signal high → ACCEPT
      5. Otherwise → REFINE
    """
    tags: list[str] = []

    if not evidence.usable_observation:
        tags.append("not_usable_observation")
        return EnforcementAction.SKIP, tags

    if evidence.occlusion_or_cloud_risk > MAX_OCCLUSION_FOR_ACTION:
        tags.append(f"occlusion_{evidence.occlusion_or_cloud_risk:.2f}_above_max_{MAX_OCCLUSION_FOR_ACTION}")
        return EnforcementAction.SKIP, tags

    if DEFER_AMBIGUITY_BAND[0] < evidence.confidence < DEFER_AMBIGUITY_BAND[1]:
        tags.append(f"confidence_in_ambiguity_band_{DEFER_AMBIGUITY_BAND}")
        return EnforcementAction.DEFER, tags

    if (evidence.confidence >= ACCEPT_CONFIDENCE_MIN
            and evidence.change_or_event_score >= ACCEPT_CONFIDENCE_MIN):
        tags.append("high_confidence_and_change")
        return EnforcementAction.ACCEPT, tags

    if evidence.confidence < REFINE_CONFIDENCE_MIN:
        tags.append("low_confidence")
        return EnforcementAction.REFINE, tags

    # Middle case: confidence above ambiguity but not high enough to accept;
    # or change signal too weak to accept. Refine.
    tags.append("intermediate_confidence_or_change")
    return EnforcementAction.REFINE, tags


def build_assessment(
    *,
    assessment_id: str,
    target_id: str,
    scenario_pack: str,
    model_id: str,
    created_at: str,
    evidence: EnforcementEvidence,
    override_action: Optional[EnforcementAction] = None,
    confidence_adjustment: float = 0.0,
    raw_response_text: Optional[str] = None,
) -> EnforcementAssessment:
    """Construct an EnforcementAssessment with action derived (or overridden).

    `override_action` lets the model directly specify an action when its
    reasoning differs from the threshold-based derivation (e.g. domain rules
    that aren't expressed in the 8-key contract). When used, the assessment
    records the override in `evidence.rationale_tags`.
    """
    if override_action is not None:
        evidence.rationale_tags = list(evidence.rationale_tags) + [
            f"action_overridden_to_{override_action.value}"
        ]
        action = override_action
    else:
        action, derived_tags = derive_action(evidence)
        evidence.rationale_tags = list(evidence.rationale_tags) + derived_tags

    return EnforcementAssessment(
        assessment_id=assessment_id,
        target_id=target_id,
        scenario_pack=scenario_pack,
        model_id=model_id,
        created_at=created_at,
        evidence=evidence,
        recommended_action=action,
        confidence_adjustment=confidence_adjustment,
        raw_response_text=raw_response_text,
    )
