"""
session_gates.py — Six non-compensatory viability gates for grounding sessions.

These are the per-session gates that complement the four governance tools and
sit between the tool layer and the federation layer. Ported from SimSat's
`src/sim/haic/viability.py::evaluate_viability` and adapted to operate on the
HAIC Maestro session shape rather than the SimSat ConventionSession.

The six gates are non-compensatory: all six must pass for the session's
training_signal to be admitted to the federated Ceff(t). A perfect score on
gate 5 cannot override failure on gate 1.

  1. entropy_reduction       — Δentropy < −ε from Prism delta
  2. extraction_risk         — bulk-extraction risk score ≤ 0.15
  3. prism_consistency       — claimed reduction matches geometric measurement
  4. participation_covenant  — valid stimulus + min turns/words + provenance
  5. federated_exchange      — raw data stays at edge (no image blobs in turns)
  6. epistemic_alignment     — exchange reduces uncertainty, not bias

The four-tool governance pipeline produces the inputs these gates consume:
- run_prism_analysis → Prism delta + snapshots (gates 1, 3)
- assess_wellbeing_domain → vulnerability flags (informs gate 2)
- verify_consent_and_provenance → provenance score (gate 4)
- generate_alignment_receipt → final session anchor

This module is the bridge between per-tool checks and the federation-level
Viability Condition. A session that fails any gate has its training_signal
withheld from the next DiLoCo round even if all four tools returned valid
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Thresholds (mirror SimSat reference)
ENTROPY_EPSILON = 0.01         # minimum entropy reduction required
EXTRACTION_RISK_LIMIT = 0.15   # max bulk-extraction risk
MIN_PROVENANCE_SCORE = 0.90    # min PoG provenance score
MIN_USER_TURNS = 2             # min meaningful user turns
MIN_WORD_COUNT = 10            # min total user words
MAX_TURN_CHAR_LENGTH = 50000   # max chars in a single turn (catches blob smuggling)
MIN_VOCAB_DIVERSITY = 0.30     # min unique-word ratio across user turns


@dataclass
class GroundingSessionView:
    """Lightweight session shape that the gates consume.

    Constructed by the caller (e.g. the gateway) from a Maestro session object
    plus any PRISM telemetry the session produced. We don't depend on a
    specific session class — just on this view contract.
    """

    session_id: str
    interview_turns: list[dict]                       # role/content dicts
    has_stimulus: bool = True                         # did the session have a valid grounding stimulus
    pog_provenance_score: Optional[float] = None     # PoG telemetry result (None if unmeasured)
    entropy_delta: Optional[dict] = None             # Prism delta: {delta_spectral_entropy, reduction_verified, snapshot_before, snapshot_after}
    image_count: int = 0                             # number of images attached to stimulus
    extra_risk_factors: dict[str, float] = field(default_factory=dict)  # caller-supplied extra risk additions


@dataclass
class SessionGateResult:
    """Result of evaluating the six gates on one session."""

    session_id: str
    entropy_reduction: bool
    extraction_risk: bool
    prism_consistency: bool
    participation_covenant: bool
    federated_exchange: bool
    epistemic_alignment: bool
    extraction_risk_score: float
    vocab_diversity: float
    failure_reasons: list[str]

    @property
    def all_passed(self) -> bool:
        return (
            self.entropy_reduction
            and self.extraction_risk
            and self.prism_consistency
            and self.participation_covenant
            and self.federated_exchange
            and self.epistemic_alignment
        )

    @property
    def num_passed(self) -> int:
        return sum([
            self.entropy_reduction,
            self.extraction_risk,
            self.prism_consistency,
            self.participation_covenant,
            self.federated_exchange,
            self.epistemic_alignment,
        ])

    def __repr__(self) -> str:
        status = "ALL_PASS" if self.all_passed else f"{self.num_passed}/6"
        return f"SessionGates(session={self.session_id!r}, {status})"


def evaluate_session(session: GroundingSessionView) -> SessionGateResult:
    """Evaluate all six gates on a grounding-session view."""
    reasons: list[str] = []

    # 1. entropy_reduction
    gate1, why = _gate_entropy_reduction(session)
    if not gate1:
        reasons.append(f"entropy_reduction: {why}")

    # 2. extraction_risk
    risk = _compute_extraction_risk(session)
    gate2 = risk <= EXTRACTION_RISK_LIMIT
    if not gate2:
        reasons.append(f"extraction_risk: {risk:.4f} > {EXTRACTION_RISK_LIMIT}")

    # 3. prism_consistency
    gate3, why = _gate_prism_consistency(session)
    if not gate3:
        reasons.append(f"prism_consistency: {why}")

    # 4. participation_covenant
    gate4, why = _gate_participation_covenant(session)
    if not gate4:
        reasons.append(f"participation_covenant: {why}")

    # 5. federated_exchange
    gate5, why = _gate_federated_exchange(session)
    if not gate5:
        reasons.append(f"federated_exchange: {why}")

    # 6. epistemic_alignment
    gate6, vocab_diversity, why = _gate_epistemic_alignment(session)
    if not gate6:
        reasons.append(f"epistemic_alignment: {why}")

    return SessionGateResult(
        session_id=session.session_id,
        entropy_reduction=gate1,
        extraction_risk=gate2,
        prism_consistency=gate3,
        participation_covenant=gate4,
        federated_exchange=gate5,
        epistemic_alignment=gate6,
        extraction_risk_score=risk,
        vocab_diversity=vocab_diversity,
        failure_reasons=reasons,
    )


# ── Individual gates ────────────────────────────────────────────────────────

def _gate_entropy_reduction(session: GroundingSessionView) -> tuple[bool, str]:
    """Gate 1: ΔS < −ε from the Prism delta."""
    if session.entropy_delta is None:
        # No PRISM — auto-pass (system may be running without geometry telemetry)
        return True, "no_prism_delta_supplied"

    delta_se = session.entropy_delta.get("delta_spectral_entropy", 0.0)
    reduction_verified = session.entropy_delta.get("reduction_verified", False)

    if reduction_verified:
        return True, "prism_verified"
    if delta_se < -ENTROPY_EPSILON:
        return True, f"delta_se={delta_se:.4f}_below_{-ENTROPY_EPSILON}"
    return False, f"delta_se={delta_se:.4f}_above_threshold_{-ENTROPY_EPSILON}"


def _gate_prism_consistency(session: GroundingSessionView) -> tuple[bool, str]:
    """Gate 3: Claimed entropy reduction matches geometric measurement."""
    if session.entropy_delta is None:
        return True, "no_prism_delta_supplied"

    delta = session.entropy_delta
    snap_before = delta.get("snapshot_before", {})
    snap_after = delta.get("snapshot_after", {})

    if not snap_before or not snap_after:
        return False, "missing_snapshots"

    se_before = snap_before.get("mean_spectral_entropy", 0.0)
    se_after = snap_after.get("mean_spectral_entropy", 0.0)
    claimed_delta = delta.get("delta_spectral_entropy", 0.0)
    actual_delta = se_after - se_before

    if abs(claimed_delta - actual_delta) < 0.001:
        return True, "consistent"
    return False, f"claimed={claimed_delta:.6f}_vs_actual={actual_delta:.6f}"


def _gate_participation_covenant(session: GroundingSessionView) -> tuple[bool, str]:
    """Gate 4: Valid stimulus + min participation + provenance."""
    if not session.has_stimulus:
        return False, "no_stimulus"

    if session.pog_provenance_score is not None:
        if session.pog_provenance_score < MIN_PROVENANCE_SCORE:
            return False, f"provenance={session.pog_provenance_score:.3f}_below_{MIN_PROVENANCE_SCORE}"

    user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
    if len(user_turns) < MIN_USER_TURNS:
        return False, f"only_{len(user_turns)}_user_turns_min_{MIN_USER_TURNS}"

    total_words = sum(len(t.get("content", "").split()) for t in user_turns)
    if total_words < MIN_WORD_COUNT:
        return False, f"only_{total_words}_words_min_{MIN_WORD_COUNT}"

    return True, "ok"


def _gate_federated_exchange(session: GroundingSessionView) -> tuple[bool, str]:
    """Gate 5: Raw data stays at edge — no image blobs / oversized turns."""
    for i, turn in enumerate(session.interview_turns):
        content = turn.get("content", "")
        if "data:image" in content:
            return False, f"turn_{i}_contains_data_image_uri"
        if len(content) > MAX_TURN_CHAR_LENGTH:
            return False, f"turn_{i}_exceeds_{MAX_TURN_CHAR_LENGTH}_chars"
    return True, "ok"


def _gate_epistemic_alignment(session: GroundingSessionView) -> tuple[bool, float, str]:
    """Gate 6: Reduces uncertainty, not bias.

    Heuristic checks:
    - Assistant doesn't repeat itself (unique-text ratio over assistant turns)
    - User vocabulary diversity ≥ MIN_VOCAB_DIVERSITY

    Returns (passed, vocab_diversity, reason). Vocab diversity is returned
    separately for the audit log.
    """
    user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
    assistant_turns = [t for t in session.interview_turns if t.get("role") == "assistant"]

    if len(user_turns) < 2:
        return True, 1.0, "insufficient_data_to_evaluate"

    # Assistant repetition check
    if assistant_turns:
        assistant_texts = [t.get("content", "") for t in assistant_turns]
        unique_assistant = len(set(assistant_texts))
        # Fail if MORE THAN half the assistant turns are duplicates of each other
        if unique_assistant < max(1, len(assistant_texts) // 2):
            return False, 0.0, f"assistant_repeated_{unique_assistant}_unique_of_{len(assistant_texts)}"

    # Vocabulary diversity check
    all_words: list[str] = []
    for t in user_turns:
        all_words.extend(t.get("content", "").lower().split())

    if not all_words:
        return True, 0.0, "no_user_words"

    diversity = len(set(all_words)) / len(all_words)
    if diversity < MIN_VOCAB_DIVERSITY:
        return False, diversity, f"vocab_diversity={diversity:.3f}_below_{MIN_VOCAB_DIVERSITY}"

    return True, diversity, "ok"


# ── Extraction risk computation ─────────────────────────────────────────────

def _compute_extraction_risk(session: GroundingSessionView) -> float:
    """Compute the extraction risk score for gate 2.

    Risk factors:
    - Long transcripts (bulk-extraction risk)
    - High image count + low turn count (image-scraping pattern)
    - No PoG telemetry (provenance unknown)
    - Caller-supplied extra factors (whitelisted into `extra_risk_factors`)
    """
    risk = 0.0

    total_chars = sum(len(t.get("content", "")) for t in session.interview_turns)
    if total_chars > 10000:
        risk += 0.05
    if total_chars > 50000:
        risk += 0.05

    user_turn_count = sum(1 for t in session.interview_turns if t.get("role") == "user")
    if session.image_count > 2 and user_turn_count < 2:
        risk += 0.08

    if session.pog_provenance_score is None:
        risk += 0.02

    for _, contribution in session.extra_risk_factors.items():
        risk += float(contribution)

    return min(1.0, round(risk, 4))
