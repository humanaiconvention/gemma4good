"""
Tests for tools/incremental_grounding.py

Validates the session-driven continual learning pipeline:
  - SFT pair extraction from HAIC sessions
  - Consent gating (training_signal denied → no output)
  - Two-level Merkle training receipt construction
  - Dry-run mode returns real data or None (never fake)
  - Adapter state hashing
"""

import json
import pytest

from tools.incremental_grounding import (
    format_session_as_sft,
    run_grounding_update,
    generate_training_receipt,
    hash_adapter_state,
    GroundingConfig,
    GroundingUpdateResult,
    _sha256,
    _merkle_root,
)


# ── Test fixtures ─────────────────────────────────────────────────────────────

SAMPLE_7_TURN_SESSION = [
    {"role": "system", "content": "You are the HAIC semantic grounding interviewer."},
    {"role": "user", "content": "I grew up in a small farming community in rural Queensland."},
    {"role": "assistant", "content": "[PIVOT: SENSORY]\nWhat did the soil smell like after the first rain of the season?"},
    {"role": "user", "content": "It smelled like iron and eucalyptus. We called it petrichor."},
    {"role": "assistant", "content": "That's a very specific sensory memory. Was that scent connected to a feeling of relief or anxiety about the crops?"},
    {"role": "user", "content": "Relief, always. If the rain came, we'd eat. If it didn't, Dad would go quiet for days."},
    {"role": "assistant", "content": "[COMPRESSION:]\nSo the rain wasn't just weather — it was the difference between your father's presence and his withdrawal?"},
]

CONSENT_GRANTED = {
    "transcript": "granted",
    "felt_state": "granted",
    "training_signal": "granted",
    "retention": "granted",
}

CONSENT_DENIED = {
    "transcript": "granted",
    "felt_state": "granted",
    "training_signal": "denied",
    "retention": "granted",
}

SAMPLE_SESSION_RECEIPT_ROOT = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


# ── format_session_as_sft ─────────────────────────────────────────────────────

class TestFormatSessionAsSft:

    def test_extracts_three_pairs_from_7_turn(self):
        """7-turn HAIC session should produce 3 SFT pairs (T2, T4, T6)."""
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, CONSENT_GRANTED)
        assert len(pairs) == 3

    def test_pairs_have_correct_structure(self):
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, CONSENT_GRANTED)
        for p in pairs:
            assert "instruction" in p
            assert "response" in p
            assert "turn_index" in p
            assert "context_messages" in p
            assert isinstance(p["instruction"], str)
            assert isinstance(p["response"], str)
            assert len(p["instruction"]) > 0
            assert len(p["response"]) > 0

    def test_first_pair_is_pivot_response(self):
        """First SFT pair should be the T2 pivot (assistant response to T1)."""
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, CONSENT_GRANTED)
        assert "[PIVOT:" in pairs[0]["response"]

    def test_consent_denied_returns_empty(self):
        """If training_signal is denied, no SFT pairs should be extracted."""
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, CONSENT_DENIED)
        assert pairs == []

    def test_consent_missing_key_returns_empty(self):
        """If training_signal key is absent, default to denied."""
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, {"transcript": "granted"})
        assert pairs == []

    def test_empty_messages_returns_empty(self):
        pairs = format_session_as_sft([], CONSENT_GRANTED)
        assert pairs == []

    def test_single_message_returns_empty(self):
        pairs = format_session_as_sft(
            [{"role": "user", "content": "hello"}],
            CONSENT_GRANTED,
        )
        assert pairs == []

    def test_context_messages_exclude_system(self):
        """Context should only include user and assistant turns."""
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, CONSENT_GRANTED)
        for p in pairs:
            for m in p["context_messages"]:
                assert m["role"] in ("user", "assistant")

    def test_invalid_messages_type_returns_empty(self):
        """If messages is not a list (e.g. LLM hallucinated string), return empty."""
        pairs = format_session_as_sft("This isn't a list", CONSENT_GRANTED)
        assert pairs == []

    def test_invalid_consent_type_returns_empty(self):
        """If consent is not a dict (e.g. LLM hallucinated string), return empty."""
        pairs = format_session_as_sft(SAMPLE_7_TURN_SESSION, "{\"training_signal\": \"granted\"}")
        assert pairs == []



# ── generate_training_receipt ─────────────────────────────────────────────────

class TestGenerateTrainingReceipt:

    def test_receipt_has_required_fields(self):
        config = GroundingConfig()
        receipt = generate_training_receipt(
            session_receipt_root=SAMPLE_SESSION_RECEIPT_ROOT,
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        assert "training_receipt_root" in receipt
        assert "training_executed" in receipt
        assert "session_receipt_root" in receipt
        assert "leaves" in receipt
        assert "verifiable" in receipt
        assert receipt["verifiable"] is True

    def test_dry_run_receipt_declares_not_executed(self):
        config = GroundingConfig()
        receipt = generate_training_receipt(
            session_receipt_root=SAMPLE_SESSION_RECEIPT_ROOT,
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        assert receipt["training_executed"] is False
        assert receipt["adapter_hash_before"] is None
        assert receipt["adapter_hash_after"] is None
        assert receipt["loss_trajectory"] is None

    def test_receipt_is_deterministic(self):
        """Same inputs must produce same Merkle root."""
        config = GroundingConfig()
        r1 = generate_training_receipt(
            session_receipt_root=SAMPLE_SESSION_RECEIPT_ROOT,
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        r2 = generate_training_receipt(
            session_receipt_root=SAMPLE_SESSION_RECEIPT_ROOT,
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        assert r1["training_receipt_root"] == r2["training_receipt_root"]

    def test_different_inputs_produce_different_roots(self):
        config = GroundingConfig()
        r1 = generate_training_receipt(
            session_receipt_root="root_a",
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        r2 = generate_training_receipt(
            session_receipt_root="root_b",
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        assert r1["training_receipt_root"] != r2["training_receipt_root"]

    def test_leaves_has_five_entries(self):
        config = GroundingConfig()
        receipt = generate_training_receipt(
            session_receipt_root=SAMPLE_SESSION_RECEIPT_ROOT,
            adapter_hash_before=None,
            adapter_hash_after=None,
            loss_trajectory=None,
            config=config,
            training_executed=False,
            sft_pair_count=3,
        )
        assert len(receipt["leaves"]) == 5
        assert set(receipt["leaves"].keys()) == {
            "session", "adapter_before", "config", "loss", "adapter_after"
        }


# ── Merkle utilities ──────────────────────────────────────────────────────────

class TestMerkleUtilities:

    def test_sha256_deterministic(self):
        assert _sha256("hello") == _sha256("hello")
        assert _sha256("hello") != _sha256("world")

    def test_merkle_root_single_leaf(self):
        leaf = _sha256("test_data")  # proper 64-char hash
        root = _merkle_root([leaf])
        assert isinstance(root, str)
        assert len(root) == 64
        assert root == leaf  # single leaf IS the root

    def test_merkle_root_empty(self):
        root = _merkle_root([])
        assert root == _sha256("empty")

    def test_merkle_root_deterministic(self):
        leaves = ["a", "b", "c", "d"]
        r1 = _merkle_root(leaves)
        r2 = _merkle_root(leaves)
        assert r1 == r2

    def test_merkle_root_order_matters(self):
        r1 = _merkle_root(["a", "b"])
        r2 = _merkle_root(["b", "a"])
        assert r1 != r2


# ── run_grounding_update (dry_run) ────────────────────────────────────────────

class TestDryRunGroundingUpdate:

    def test_dry_run_returns_result(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert isinstance(result, GroundingUpdateResult)

    def test_dry_run_training_not_executed(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert result.training_executed is False
        assert result.mode == "dry_run"
        assert result.steps_executed == 0

    def test_dry_run_has_real_sft_pairs(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert result.sft_pair_count == 3
        assert len(result.sft_pairs) == 3

    def test_dry_run_loss_is_none_not_fake(self):
        """Anti-hallucination: dry_run must not generate fake loss values."""
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert result.loss_before is None
        assert result.loss_after is None
        assert result.loss_trajectory is None

    def test_dry_run_adapter_hash_is_none_not_fake(self):
        """Anti-hallucination: dry_run must not generate fake adapter hashes."""
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert result.adapter_hash_before is None
        assert result.adapter_hash_after is None

    def test_dry_run_has_training_receipt(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert result.training_receipt is not None
        assert result.training_receipt["training_executed"] is False

    def test_dry_run_has_real_token_count(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert result.token_count is not None
        assert result.token_count > 0

    def test_dry_run_with_session_receipt_root(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
            session_receipt_root=SAMPLE_SESSION_RECEIPT_ROOT,
        )
        assert result.session_receipt_root == SAMPLE_SESSION_RECEIPT_ROOT
        assert result.training_receipt["session_receipt_root"] == SAMPLE_SESSION_RECEIPT_ROOT


# ── Consent gating ────────────────────────────────────────────────────────────

class TestConsentGate:

    def test_denied_consent_blocks_update(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_DENIED,
        )
        assert result.consent_blocked is True
        assert result.consent_valid is False
        assert result.training_executed is False
        assert result.sft_pair_count == 0
        assert result.training_receipt is None    # no receipt for blocked updates
        assert result.error is not None

    def test_missing_consent_key_blocks_update(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent={"transcript": "granted"},
        )
        assert result.consent_blocked is True
        assert result.sft_pair_count == 0

    def test_empty_consent_blocks_update(self):
        result = run_grounding_update(
            session_messages=SAMPLE_7_TURN_SESSION,
            consent={},
        )
        assert result.consent_blocked is True


# ── hash_adapter_state ────────────────────────────────────────────────────────

class TestHashAdapterState:

    def test_none_returns_none(self):
        assert hash_adapter_state(None) is None

    def test_dict_produces_hex_string(self):
        import numpy as np
        state = {"layer.weight": np.array([1.0, 2.0, 3.0])}
        h = hash_adapter_state(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_dict_is_deterministic(self):
        import numpy as np
        state = {"layer.weight": np.array([1.0, 2.0, 3.0])}
        h1 = hash_adapter_state(state)
        h2 = hash_adapter_state(state)
        assert h1 == h2

    def test_different_weights_different_hash(self):
        import numpy as np
        h1 = hash_adapter_state({"w": np.array([1.0])})
        h2 = hash_adapter_state({"w": np.array([2.0])})
        assert h1 != h2


# ── Mode validation ───────────────────────────────────────────────────────────

class TestModeValidation:

    def test_valid_dry_run_mode(self):
        config = GroundingConfig(mode="dry_run")
        assert config.mode == "dry_run"

    def test_valid_live_mode(self):
        config = GroundingConfig(mode="live")
        assert config.mode == "live"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            GroundingConfig(mode="simulate")

    def test_bogus_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            GroundingConfig(mode="bogus")

    def test_empty_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            GroundingConfig(mode="")


# ── Handler integration ───────────────────────────────────────────────────────

class TestHandler:

    def test_handler_returns_dict(self):
        from tools.incremental_grounding import run_grounding_update_handler
        result = run_grounding_update_handler(
            session_id="test-session-001",
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert isinstance(result, dict)
        assert result["session_id"] == "test-session-001"
        assert result["training_executed"] is False
        assert result["sft_pair_count"] == 3

    def test_handler_includes_sft_summaries(self):
        from tools.incremental_grounding import run_grounding_update_handler
        result = run_grounding_update_handler(
            session_id="test-session-002",
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_GRANTED,
        )
        assert "sft_pairs_summary" in result
        assert len(result["sft_pairs_summary"]) == 3
        # Summaries have length info, not full content (privacy)
        for summary in result["sft_pairs_summary"]:
            assert "turn_index" in summary
            assert "instruction_length" in summary
            assert "response_length" in summary

    def test_handler_consent_blocked_has_error(self):
        from tools.incremental_grounding import run_grounding_update_handler
        result = run_grounding_update_handler(
            session_id="test-session-003",
            session_messages=SAMPLE_7_TURN_SESSION,
            consent=CONSENT_DENIED,
        )
        assert result["consent_blocked"] is True
        assert "error" in result


# ── Config hashing ────────────────────────────────────────────────────────────

class TestConfigHashing:

    def test_adapter_path_excluded_from_hash(self):
        """adapter_path is environment-specific and must not affect hash."""
        c1 = GroundingConfig(adapter_path="/path/a")
        c2 = GroundingConfig(adapter_path="/path/b")
        assert c1.to_hashable_dict() == c2.to_hashable_dict()

    def test_config_hash_is_deterministic(self):
        c1 = GroundingConfig()
        c2 = GroundingConfig()
        d1 = c1.to_hashable_dict()
        d2 = c2.to_hashable_dict()
        assert d1 == d2

    def test_different_lr_different_hash(self):
        c1 = GroundingConfig(lr=5e-5)
        c2 = GroundingConfig(lr=1e-4)
        assert c1.to_hashable_dict() != c2.to_hashable_dict()

