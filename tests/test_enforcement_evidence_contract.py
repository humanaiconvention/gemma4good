"""Tests for tools/enforcement_evidence_contract.py — VLA-style 8-key evidence + 4 actions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.enforcement_evidence_contract import (
    ACCEPT_CONFIDENCE_MIN,
    DEFER_AMBIGUITY_BAND,
    MAX_OCCLUSION_FOR_ACTION,
    REFINE_CONFIDENCE_MIN,
    EnforcementAction,
    EnforcementEvidence,
    build_assessment,
    derive_action,
)


# ── derive_action: priority order matters ───────────────────────────────────

def test_skip_when_not_usable_regardless_of_confidence():
    """SKIP takes precedence over everything else."""
    evidence = EnforcementEvidence(
        usable_observation=False,
        scene_match_score=1.0,
        salience_score=1.0,
        change_or_event_score=1.0,
        occlusion_or_cloud_risk=0.0,
        confidence=0.99,
    )
    action, tags = derive_action(evidence)
    assert action == EnforcementAction.SKIP
    assert "not_usable_observation" in tags


def test_skip_when_occlusion_too_high():
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.99,
        change_or_event_score=0.99,
        occlusion_or_cloud_risk=MAX_OCCLUSION_FOR_ACTION + 0.01,
    )
    action, tags = derive_action(evidence)
    assert action == EnforcementAction.SKIP
    assert any("occlusion" in t for t in tags)


def test_defer_when_confidence_in_ambiguity_band():
    """SimSat ambiguity band: 0.40 < conf < 0.65 → DEFER."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.55,
        change_or_event_score=0.7,
    )
    action, tags = derive_action(evidence)
    assert action == EnforcementAction.DEFER


def test_accept_requires_both_high_confidence_and_high_change():
    """ACCEPT only when confidence ≥ 0.80 AND change_or_event_score ≥ 0.80."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.9,
        change_or_event_score=0.9,
    )
    action, _ = derive_action(evidence)
    assert action == EnforcementAction.ACCEPT


def test_high_confidence_low_change_does_not_accept():
    """High confidence alone (without high change signal) does not trigger ACCEPT —
    refines because the change evidence is too weak to act on."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.9,
        change_or_event_score=0.3,
    )
    action, _ = derive_action(evidence)
    assert action == EnforcementAction.REFINE


def test_low_confidence_refines():
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.2,
        change_or_event_score=0.5,
    )
    action, tags = derive_action(evidence)
    assert action == EnforcementAction.REFINE
    assert "low_confidence" in tags


def test_intermediate_confidence_above_ambiguity_band_refines():
    """0.65 ≤ confidence < 0.80, change not high → REFINE."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.7,
        change_or_event_score=0.6,
    )
    action, _ = derive_action(evidence)
    assert action == EnforcementAction.REFINE


# ── Boundary cases ──────────────────────────────────────────────────────────

def test_confidence_at_ambiguity_lower_bound_does_not_defer():
    """confidence == 0.40 is NOT inside the band (strict inequality)."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=DEFER_AMBIGUITY_BAND[0],
        change_or_event_score=0.5,
    )
    action, _ = derive_action(evidence)
    assert action != EnforcementAction.DEFER


def test_confidence_at_ambiguity_upper_bound_does_not_defer():
    """confidence == 0.65 is NOT inside the band (strict inequality)."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=DEFER_AMBIGUITY_BAND[1],
        change_or_event_score=0.5,
    )
    action, _ = derive_action(evidence)
    assert action != EnforcementAction.DEFER


def test_occlusion_at_max_passes():
    """occlusion == 0.50 is NOT > 0.50 → does not trigger SKIP on occlusion alone."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.9,
        change_or_event_score=0.9,
        occlusion_or_cloud_risk=MAX_OCCLUSION_FOR_ACTION,
    )
    action, _ = derive_action(evidence)
    assert action == EnforcementAction.ACCEPT


# ── build_assessment ────────────────────────────────────────────────────────

def test_build_assessment_derives_action_from_evidence():
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.9,
        change_or_event_score=0.9,
        rationale_tags=["amazon_cell_4521", "june"],
    )
    a = build_assessment(
        assessment_id="asm-1",
        target_id="amazon-cell-4521",
        scenario_pack="amazon_deforestation",
        model_id="haic-gemma4-v42",
        created_at="2026-05-11T00:00:00Z",
        evidence=evidence,
    )
    assert a.recommended_action == EnforcementAction.ACCEPT
    # Original tags preserved + derived tags appended
    assert "amazon_cell_4521" in a.evidence.rationale_tags
    assert any("high_confidence" in t for t in a.evidence.rationale_tags)


def test_build_assessment_respects_override():
    """When the model overrides the derived action, the override wins and is logged."""
    evidence = EnforcementEvidence(
        usable_observation=True,
        confidence=0.9,
        change_or_event_score=0.9,
    )
    a = build_assessment(
        assessment_id="asm-1",
        target_id="amazon-cell-4521",
        scenario_pack="amazon_deforestation",
        model_id="haic-gemma4-v42",
        created_at="2026-05-11T00:00:00Z",
        evidence=evidence,
        override_action=EnforcementAction.DEFER,
    )
    assert a.recommended_action == EnforcementAction.DEFER
    assert any("action_overridden_to_defer" in t for t in a.evidence.rationale_tags)


# ── Merkle leaf compatibility ───────────────────────────────────────────────

def test_evidence_leaf_hash_is_stable():
    """Two evidence objects with identical fields produce identical leaf hashes."""
    e1 = EnforcementEvidence(
        usable_observation=True,
        scene_match_score=0.8, salience_score=0.7, change_or_event_score=0.9,
        occlusion_or_cloud_risk=0.1, confidence=0.85,
        rationale_tags=["a", "b"],
    )
    e2 = EnforcementEvidence(
        usable_observation=True,
        scene_match_score=0.8, salience_score=0.7, change_or_event_score=0.9,
        occlusion_or_cloud_risk=0.1, confidence=0.85,
        rationale_tags=["a", "b"],
    )
    assert e1.as_leaf_hash() == e2.as_leaf_hash()


def test_assessment_leaf_hash_excludes_volatile_fields():
    """The leaf hash should not depend on raw_response_text (which is volatile
    and not stable across deserialisations)."""
    evidence = EnforcementEvidence(
        usable_observation=True, confidence=0.9, change_or_event_score=0.9,
    )
    a1 = build_assessment(
        assessment_id="asm-1", target_id="t", scenario_pack="p",
        model_id="m", created_at="ts", evidence=evidence,
        raw_response_text="version one",
    )
    a2 = build_assessment(
        assessment_id="asm-1", target_id="t", scenario_pack="p",
        model_id="m", created_at="ts", evidence=evidence,
        raw_response_text="version two — different but stable hash",
    )
    # NOTE: build_assessment mutates evidence.rationale_tags, so the second
    # call gets DOUBLE-appended derived tags. We use the same evidence object,
    # so the hashes match after the second call's appendments.
    # For a stricter test, build evidence fresh per assessment.
    fresh1 = EnforcementEvidence(usable_observation=True, confidence=0.9, change_or_event_score=0.9)
    fresh2 = EnforcementEvidence(usable_observation=True, confidence=0.9, change_or_event_score=0.9)
    b1 = build_assessment(
        assessment_id="asm-1", target_id="t", scenario_pack="p",
        model_id="m", created_at="ts", evidence=fresh1,
        raw_response_text="one",
    )
    b2 = build_assessment(
        assessment_id="asm-1", target_id="t", scenario_pack="p",
        model_id="m", created_at="ts", evidence=fresh2,
        raw_response_text="two — different",
    )
    assert b1.as_leaf_hash() == b2.as_leaf_hash()


# ── Enum value stability ────────────────────────────────────────────────────

def test_action_enum_values_match_simsat_contract():
    """The 4 action values must match SimSat exactly for cross-project parity."""
    assert EnforcementAction.ACCEPT.value == "accept"
    assert EnforcementAction.REFINE.value == "refine"
    assert EnforcementAction.DEFER.value == "defer"
    assert EnforcementAction.SKIP.value == "skip"


def test_thresholds_documented():
    assert ACCEPT_CONFIDENCE_MIN == 0.80
    assert REFINE_CONFIDENCE_MIN == 0.40
    assert DEFER_AMBIGUITY_BAND == (0.40, 0.65)
    assert MAX_OCCLUSION_FOR_ACTION == 0.50
