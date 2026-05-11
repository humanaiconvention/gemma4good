"""Tests for tools/diloco_fragment_verifier.py — fragment verification gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.diloco_fragment_verifier import (
    FragmentExpectation,
    FragmentReceipt,
    FragmentShape,
    build_fragment_receipt,
    verify_fragment,
)
from utils.merkle import sha3_256_hex, merkle_root


# ── Fixture helpers ─────────────────────────────────────────────────────────

def _full_consent() -> dict:
    return {
        "transcript": True,
        "felt_state": True,
        "gfs_activations": True,
        "training_signal": True,
        "retention": True,
    }


def _make_session_receipt(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "turns": 6,
        "consent": _full_consent(),
        "ts": "2026-05-11T00:00:00Z",
    }


def _good_shape(num_layers: int = 35) -> FragmentShape:
    """Build a well-formed shape covering all 7 target modules x num_layers."""
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
    names = []
    shapes = {}
    norms = {}
    total = 0
    for layer in range(num_layers):
        for module in target_modules:
            for ab in ("lora_A", "lora_B"):
                n = f"layers.{layer}.{module}.{ab}.default.weight"
                names.append(n)
                # Rank 16 over 1536 hidden = 16*1536 = 24576 fp16 elements
                shapes[n] = (16, 1536) if ab == "lora_A" else (1536, 16)
                norms[n] = 0.5  # healthy non-zero norm
                total += 24576 * 2  # fp16
    return FragmentShape(
        tensor_names=names,
        tensor_shapes=shapes,
        tensor_norms=norms,
        total_bytes=total,
    )


def _good_receipt(num_sessions: int = 12) -> FragmentReceipt:
    receipts = [_make_session_receipt(f"s-{i}") for i in range(num_sessions)]
    consents = [_full_consent() for _ in range(num_sessions)]
    return build_fragment_receipt(
        learner_id="clinic-bolivia-03",
        round_id=7,
        dataset_id="haic-clinic-week-7",
        per_session_receipts=receipts,
        per_session_consents=consents,
    )


# ── Happy path ──────────────────────────────────────────────────────────────

def test_well_formed_fragment_verifies():
    receipt = _good_receipt()
    shape = _good_shape()
    result = verify_fragment(receipt, shape)
    assert result.verified
    assert result.failure_reason is None
    assert result.computed_round_root == receipt.claimed_round_root
    assert result.fragment_summary["num_tensors"] == 35 * 7 * 2
    assert result.fragment_summary["num_sessions_in_round"] == 12


def test_build_fragment_receipt_round_root_matches_recompute():
    receipt = _good_receipt(num_sessions=5)
    recomputed = merkle_root(receipt.session_receipt_leaves)
    assert recomputed == receipt.claimed_round_root


# ── Merkle integrity failures ───────────────────────────────────────────────

def test_tampered_root_rejected():
    receipt = _good_receipt()
    # Tamper the claimed root
    receipt.claimed_round_root = "deadbeef" * 8
    shape = _good_shape()
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "merkle_root_mismatch" in result.failure_reason


def test_extra_leaf_inserted_rejected():
    """A learner that tries to anchor a Merkle root over MORE leaves than they
    list in session_receipt_leaves will be caught."""
    receipt = _good_receipt(num_sessions=3)
    # Compute root over 4 leaves but only present 3
    extra_leaves = receipt.session_receipt_leaves + [sha3_256_hex("smuggled")]
    receipt.claimed_round_root = merkle_root(extra_leaves)
    shape = _good_shape()
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "merkle_root_mismatch" in result.failure_reason


# ── Consent failures ────────────────────────────────────────────────────────

def test_missing_consent_layer_rejected():
    receipt = _good_receipt(num_sessions=3)
    # Remove gfs_activations from session 1
    receipt.session_consents[1] = {
        k: True for k in ["transcript", "felt_state", "training_signal", "retention"]
    }
    shape = _good_shape()
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "session_1_missing_consent_layers" in result.failure_reason
    assert "gfs_activations" in result.failure_reason


def test_denied_consent_layer_rejected():
    receipt = _good_receipt(num_sessions=3)
    # Mark training_signal as denied in session 2 — should never have entered the fragment
    receipt.session_consents[2]["training_signal"] = False
    shape = _good_shape()
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "session_2_denied_consent_layers" in result.failure_reason
    assert "training_signal" in result.failure_reason


# ── Shape failures ──────────────────────────────────────────────────────────

def test_too_few_tensors_rejected_simsat_v11_pattern():
    """Reproduces the SimSat v11 partial-save: k/v dropped on later layers."""
    receipt = _good_receipt()
    bad_shape = _good_shape(num_layers=20)  # only 20*7*2 = 280 < min_tensors=400
    result = verify_fragment(receipt, bad_shape)
    assert not result.verified
    assert "too_few_tensors" in result.failure_reason


def test_missing_target_module_rejected_simsat_null_pattern():
    """Reproduces the SimSat null-training pattern: LoRA on towers only, no q/k/v."""
    receipt = _good_receipt()
    shape = _good_shape()
    # Drop all q_proj tensors — model has no q_proj LoRA at all
    shape.tensor_names = [n for n in shape.tensor_names if "q_proj" not in n]
    shape.tensor_shapes = {k: v for k, v in shape.tensor_shapes.items() if "q_proj" not in k}
    shape.tensor_norms = {k: v for k, v in shape.tensor_norms.items() if "q_proj" not in k}
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "missing_target_modules" in result.failure_reason
    assert "q_proj" in result.failure_reason


def test_fragment_too_large_rejected():
    receipt = _good_receipt()
    shape = _good_shape()
    shape.total_bytes = 500 * 1024 * 1024  # 500 MB — suspicious for LoRA-only
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "fragment_too_large" in result.failure_reason


# ── Norm failures ───────────────────────────────────────────────────────────

def test_poisoned_huge_norm_tensor_rejected():
    receipt = _good_receipt()
    shape = _good_shape()
    # Inject a tensor with absurdly large norm
    first_name = shape.tensor_names[0]
    shape.tensor_norms[first_name] = 1e6
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "tensor_norm_out_of_bounds" in result.failure_reason


def test_null_training_pattern_rejected():
    """All-zero tensors → SimSat null-training pattern, reject."""
    receipt = _good_receipt()
    shape = _good_shape()
    for name in shape.tensor_norms:
        shape.tensor_norms[name] = 0.0
    result = verify_fragment(receipt, shape)
    assert not result.verified
    assert "null_training_pattern" in result.failure_reason


def test_small_minority_of_null_tensors_warns_but_passes():
    """A few near-zero lora_B values early in training is normal — warn, don't reject."""
    receipt = _good_receipt()
    shape = _good_shape()
    # Zero out 10% of tensors
    names = list(shape.tensor_norms.keys())
    for name in names[: len(names) // 10]:
        shape.tensor_norms[name] = 0.0
    result = verify_fragment(receipt, shape)
    assert result.verified
    assert any("near-zero norm" in w for w in result.warnings)


# ── Custom expectation ──────────────────────────────────────────────────────

def test_custom_expectation_can_relax_constraints():
    """Smaller models (fewer layers) should be verifiable with a relaxed expectation."""
    receipt = _good_receipt()
    small_shape = _good_shape(num_layers=12)  # 12*7*2 = 168 tensors
    relaxed = FragmentExpectation(
        expected_layers=12,
        min_tensors=150,
        max_tensors=200,
    )
    result = verify_fragment(receipt, small_shape, expectation=relaxed)
    assert result.verified, result.failure_reason
