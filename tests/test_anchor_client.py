"""Tests for onchain.anchor_client — pure helpers, cross-language, CLI.

Live web3 path is exercised by Foundry tests in onchain/test/ and the manual
smoke test in onchain/README.md. Mocking web3 here added little value over the
on-chain test suite.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onchain.anchor_client import (
    session_id_to_bytes32,
    hex_to_bytes32,
    _KIND_MAP,
)
from utils.merkle import sha3_256_hex, merkle_root, hash_items_to_leaves


class TestPureHelpers:
    def test_session_id_deterministic(self):
        a = session_id_to_bytes32("sess_42")
        assert a == session_id_to_bytes32("sess_42")
        assert len(a) == 32
        assert a.hex() == sha3_256_hex("sess_42")

    def test_session_id_distinct(self):
        assert session_id_to_bytes32("a") != session_id_to_bytes32("b")

    def test_hex_to_bytes32_strips_prefix(self):
        h = sha3_256_hex("hello")
        assert hex_to_bytes32(h) == hex_to_bytes32("0x" + h)
        assert len(hex_to_bytes32(h)) == 32

    def test_hex_to_bytes32_rejects_short(self):
        with pytest.raises(ValueError):
            hex_to_bytes32("deadbeef")

    def test_hex_to_bytes32_rejects_long(self):
        with pytest.raises(ValueError):
            hex_to_bytes32("ab" * 33)

    def test_kind_map_complete(self):
        assert _KIND_MAP == {"session": 0, "training": 1, "consent": 2}


class TestCrossLanguageConsistency:
    """Roots produced here MUST match what the Solidity contract stores."""

    def test_merkle_root_is_bytes32_compatible(self):
        leaves = hash_items_to_leaves([{"role": "user", "content": "hi"}])
        root = merkle_root(leaves)
        b = hex_to_bytes32(root)
        assert len(b) == 32
        assert b.hex() == root

    def test_session_id_known_value(self):
        expected = sha3_256_hex("sess_42")
        assert session_id_to_bytes32("sess_42").hex() == expected
        assert len(expected) == 64

    def test_full_receipt_roundtrip(self):
        from maestro_integration.maestro_client import MaestroClient
        msgs = [{"role": "user", "content": "test"},
                {"role": "assistant", "content": "ok"}]
        consent = {"transcript": "granted"}
        receipt = MaestroClient._local_receipt("sess_xyz", msgs, consent)

        sid_b = session_id_to_bytes32(receipt["session_id"])
        root_b = hex_to_bytes32(receipt["merkle_root"])
        assert len(sid_b) == 32
        assert len(root_b) == 32


class TestCLI:
    def test_dry_run_reads_receipt(self, tmp_path, capsys):
        receipt = {
            "session_id": "sess_42",
            "merkle_root": sha3_256_hex("payload"),
            "node_count": 5,
        }
        p = tmp_path / "r.json"
        p.write_text(json.dumps(receipt))

        with patch("sys.argv", ["anchor_client", "--receipt", str(p), "--dry-run"]):
            from onchain.anchor_client import cli
            cli()
        out = capsys.readouterr().out
        assert "sess_42" in out
        assert "dry run" in out
        assert sha3_256_hex("payload") in out

    def test_dry_run_rejects_bad_root(self, tmp_path):
        receipt = {"session_id": "x", "merkle_root": "not_hex", "node_count": 0}
        p = tmp_path / "r.json"
        p.write_text(json.dumps(receipt))
        with patch("sys.argv", ["anchor_client", "--receipt", str(p), "--dry-run"]):
            from onchain.anchor_client import cli
            with pytest.raises(ValueError):
                cli()


class TestFixturesJsonAgreement:
    """Confirm fixtures.json (consumed by Foundry test) matches live Python output."""

    def test_fixtures_match_live_python(self):
        import json
        from pathlib import Path
        fixtures_path = Path(__file__).resolve().parent.parent / "onchain" / "test" / "fixtures.json"
        if not fixtures_path.exists():
            pytest.skip("fixtures.json not present — run python onchain/gen_fixtures.py")
        fixtures = json.loads(fixtures_path.read_text())

        # Sessions
        for label, hex_val in fixtures["sessions"].items():
            assert hex_val == "0x" + sha3_256_hex(label)

        # Single-payload root
        assert fixtures["roots"]["payload_single"] == "0x" + sha3_256_hex("payload_single")

        # Empty tree convention
        assert fixtures["roots"]["empty_tree"] == "0x" + merkle_root([])

        # Three-message tree (must match the MESSAGES_3 constant in gen_fixtures.py)
        msgs_3 = [
            {"role": "user",      "content": "Tell me about your AI use."},
            {"role": "assistant", "content": "[PIVOT: SHADOW] When did that start?"},
            {"role": "user",      "content": "Last month when my boss installed it."},
        ]
        assert fixtures["roots"]["three_message_tree"] == "0x" + merkle_root(hash_items_to_leaves(msgs_3))
