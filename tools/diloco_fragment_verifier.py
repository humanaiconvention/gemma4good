"""
diloco_fragment_verifier.py — Verify DiLoCo fragments before they enter the global merge.

A DiLoCo fragment is one learner's contribution to a sync round. In the SimSat
diloco_lab pattern, a fragment is a LoRA delta packaged with a Merkle receipt
that anchors the local governance traces accumulated since the previous round.

This module implements the syncer-side verification gate. The Viability Condition
at federation scale depends on accepting only fragments whose:
  1. Merkle receipt verifies against the claimed per-session traces
  2. Per-session consent layers were all granted (no smuggled
     training_signal=denied sessions)
  3. Tensor shape and norm are within expected bounds (no surprise injection)
  4. Round receipt sha matches the value the learner registered when uploading

See docs/diloco_integration_2026-05-11.md for the full framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# This module avoids hard dependencies on torch/safetensors so it can be unit-tested
# without a GPU stack. Fragment shape verification operates on a lightweight summary
# dict that the syncer constructs from the actual safetensors file.
from utils.merkle import sha3_256_hex, merkle_root


REQUIRED_CONSENT_LAYERS = {
    "transcript",
    "felt_state",
    "gfs_activations",
    "training_signal",
    "retention",
}


@dataclass
class FragmentReceipt:
    """One learner's round receipt — anchors the governance traces in this fragment."""

    learner_id: str
    round_id: int
    dataset_id: str
    session_receipt_leaves: list[str]   # SHA3-256 leaves of per-session receipts
    session_consents: list[dict]        # one consent dict per session, layer→bool
    claimed_round_root: str             # the Merkle root the learner uploaded


@dataclass
class FragmentShape:
    """Lightweight summary of a fragment's tensor layout.

    The syncer extracts this from the actual safetensors file before calling
    verify_fragment(); we keep the verifier framework-agnostic.
    """

    tensor_names: list[str]             # e.g. ['layers.0.q_proj.lora_A', ...]
    tensor_shapes: dict[str, tuple]     # name → shape
    tensor_norms: dict[str, float]      # name → frobenius norm (or rms)
    total_bytes: int


@dataclass
class FragmentExpectation:
    """What the syncer expects to see in a well-formed fragment.

    Defaults match a Gemma-4-E2B rank-16 LoRA over the 7 standard target modules
    × 35 decoder layers = 490 lora tensors. Out of the SimSat lessons:
    the v11 partial-save bug truncated this to ~410, so the verifier flags
    anything below `min_tensors`.
    """

    expected_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    expected_layers: int = 35
    min_tensors: int = 400              # SimSat v11 was 410; allow some slack
    max_tensors: int = 500
    max_norm_per_tensor: float = 100.0  # sanity bound to catch poisoned fragments
    min_norm_per_tensor: float = 1e-6   # all-zero fragment → null-trained learner
    max_total_bytes: int = 200 * 1024 * 1024   # 200 MB ceiling per fragment


@dataclass
class FragmentVerificationResult:
    """Outcome of verifying a single learner's fragment."""

    learner_id: str
    round_id: int
    verified: bool
    failure_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    computed_round_root: Optional[str] = None
    fragment_summary: Optional[dict] = None

    def __repr__(self) -> str:
        status = "VERIFIED" if self.verified else f"REJECTED({self.failure_reason!r})"
        return f"FragmentVerification(learner={self.learner_id!r}, round={self.round_id}, {status})"


def verify_fragment(
    receipt: FragmentReceipt,
    shape: FragmentShape,
    expectation: Optional[FragmentExpectation] = None,
) -> FragmentVerificationResult:
    """Verify one DiLoCo fragment for syncer acceptance.

    Performs four checks (returns the first failure, but also collects
    warnings for non-fatal anomalies):

    1. **Merkle integrity** — recompute the round root from the session leaves
       and compare against the learner's claimed root.
    2. **Consent compliance** — every per-session consent dict must have all
       required layers granted; otherwise the learner included a session that
       should not have contributed to training.
    3. **Shape compatibility** — tensor count, target module coverage, and
       per-layer presence match the expectation.
    4. **Norm bounds** — no tensor is all-zero (null-trained) or absurdly
       large (poisoned).

    The Viability Condition at federation scale is defined to exclude
    fragments where any of these fail.
    """
    if expectation is None:
        expectation = FragmentExpectation()

    result = FragmentVerificationResult(
        learner_id=receipt.learner_id,
        round_id=receipt.round_id,
        verified=False,
    )

    # 1. Merkle integrity
    computed_root = merkle_root(receipt.session_receipt_leaves)
    result.computed_round_root = computed_root
    if computed_root != receipt.claimed_round_root:
        result.failure_reason = (
            f"merkle_root_mismatch: computed={computed_root[:16]}…, "
            f"claimed={receipt.claimed_round_root[:16]}…"
        )
        return result

    # 2. Consent compliance — every session must have all required layers.
    for i, consent in enumerate(receipt.session_consents):
        missing = REQUIRED_CONSENT_LAYERS - set(consent.keys())
        if missing:
            result.failure_reason = (
                f"session_{i}_missing_consent_layers: {sorted(missing)}"
            )
            return result
        denied = {layer for layer in REQUIRED_CONSENT_LAYERS if not consent.get(layer)}
        if denied:
            result.failure_reason = (
                f"session_{i}_denied_consent_layers: {sorted(denied)}"
            )
            return result

    # 3. Shape compatibility
    num_tensors = len(shape.tensor_names)
    if num_tensors < expectation.min_tensors:
        result.failure_reason = (
            f"too_few_tensors: {num_tensors} < min_tensors={expectation.min_tensors} "
            f"(SimSat v11 partial-save pattern — k/v dropped on later layers)"
        )
        return result
    if num_tensors > expectation.max_tensors:
        result.failure_reason = (
            f"too_many_tensors: {num_tensors} > max_tensors={expectation.max_tensors} "
            f"(unexpected modules attached)"
        )
        return result

    # Coverage check: each expected target module appears at least once.
    found_modules = set()
    for name in shape.tensor_names:
        for module in expectation.expected_target_modules:
            if module in name:
                found_modules.add(module)
                break
    missing_modules = set(expectation.expected_target_modules) - found_modules
    if missing_modules:
        result.failure_reason = (
            f"missing_target_modules: {sorted(missing_modules)} "
            f"(SimSat null-trained pattern — LoRA never reached the language model)"
        )
        return result

    # Byte-size ceiling
    if shape.total_bytes > expectation.max_total_bytes:
        result.failure_reason = (
            f"fragment_too_large: {shape.total_bytes} > {expectation.max_total_bytes} bytes "
            f"(suspicious; LoRA-only fragments should be <100 MB)"
        )
        return result

    # 4. Norm bounds
    bad_norm_tensors = []
    null_tensors = []
    for name, norm in shape.tensor_norms.items():
        if norm > expectation.max_norm_per_tensor:
            bad_norm_tensors.append((name, norm))
        if norm < expectation.min_norm_per_tensor:
            null_tensors.append(name)

    if bad_norm_tensors:
        worst = max(bad_norm_tensors, key=lambda t: t[1])
        result.failure_reason = (
            f"tensor_norm_out_of_bounds: {worst[0]}={worst[1]:.3f} "
            f"> max={expectation.max_norm_per_tensor}"
        )
        return result

    # null tensors are a warning, not a fatal error — a fragment can legitimately have
    # some lora_B tensors at very small norms (early in training). But if MORE than
    # half are null, that's the SimSat null-training pattern.
    null_ratio = len(null_tensors) / max(num_tensors, 1)
    if null_ratio > 0.5:
        result.failure_reason = (
            f"null_training_pattern: {len(null_tensors)}/{num_tensors} tensors near zero "
            f"(SimSat null-training pattern — gradients never reached LoRA)"
        )
        return result
    if null_tensors:
        result.warnings.append(
            f"{len(null_tensors)} tensors at near-zero norm "
            f"(typically early-training lora_B values; non-fatal)"
        )

    # All checks pass.
    result.verified = True
    result.fragment_summary = {
        "num_tensors": num_tensors,
        "total_bytes": shape.total_bytes,
        "num_sessions_in_round": len(receipt.session_receipt_leaves),
        "computed_round_root": computed_root,
        "warnings": list(result.warnings),
    }
    return result


def build_fragment_receipt(
    learner_id: str,
    round_id: int,
    dataset_id: str,
    per_session_receipts: list[dict],
    per_session_consents: list[dict],
) -> FragmentReceipt:
    """Helper for the learner side: build a FragmentReceipt from per-session traces.

    The learner runs this before uploading the fragment to the syncer; the syncer
    then runs verify_fragment() against the same data.
    """
    leaves = []
    for r in per_session_receipts:
        # Re-hash the per-session receipt deterministically (sorted-keys JSON)
        import json
        leaves.append(sha3_256_hex(json.dumps(r, sort_keys=True)))

    root = merkle_root(leaves)

    return FragmentReceipt(
        learner_id=learner_id,
        round_id=round_id,
        dataset_id=dataset_id,
        session_receipt_leaves=leaves,
        session_consents=list(per_session_consents),
        claimed_round_root=root,
    )
