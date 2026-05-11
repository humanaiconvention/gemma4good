"""
tests/test_merkle.py — Unit tests for utils/merkle.py.

Validates the canonical SHA3-256 Merkle tree implementation that all
three receipt producers (haic_tools, incremental_grounding, maestro_client)
now share.
"""

import hashlib
import json
import os
import sys
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.merkle import sha3_256_hex, merkle_root, hash_items_to_leaves


# ── sha3_256_hex ──────────────────────────────────────────────────────────────

class TestSha3256Hex:
    """Low-level SHA3-256 wrapper."""

    def test_deterministic(self):
        assert sha3_256_hex("hello") == sha3_256_hex("hello")

    def test_different_inputs_different_outputs(self):
        assert sha3_256_hex("a") != sha3_256_hex("b")

    def test_matches_hashlib_directly(self):
        expected = hashlib.sha3_256("test".encode("utf-8")).hexdigest()
        assert sha3_256_hex("test") == expected

    def test_returns_64_char_hex(self):
        h = sha3_256_hex("anything")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_accepts_bytes(self):
        h = sha3_256_hex(b"raw bytes")
        assert len(h) == 64
        assert sha3_256_hex(b"raw bytes") == sha3_256_hex(b"raw bytes")

    def test_string_and_bytes_equivalent(self):
        """sha3_256_hex("x") should equal sha3_256_hex(b"x") since it encodes utf-8."""
        assert sha3_256_hex("x") == sha3_256_hex(b"x")

    def test_empty_string(self):
        h = sha3_256_hex("")
        assert len(h) == 64


# ── merkle_root ───────────────────────────────────────────────────────────────

class TestMerkleRoot:
    """Pairwise SHA3-256 reduction."""

    def test_empty_list_returns_hash_of_empty(self):
        expected = sha3_256_hex("empty")
        assert merkle_root([]) == expected

    def test_single_leaf(self):
        leaf = sha3_256_hex("only_leaf")
        # Single leaf: [leaf] → done (no pairing needed since len == 1)
        assert merkle_root([leaf]) == leaf

    def test_two_leaves(self):
        a = sha3_256_hex("a")
        b = sha3_256_hex("b")
        expected = sha3_256_hex(a + b)
        assert merkle_root([a, b]) == expected

    def test_three_leaves_duplicates_last(self):
        a = sha3_256_hex("a")
        b = sha3_256_hex("b")
        c = sha3_256_hex("c")
        # 3 leaves → duplicate last → [a, b, c, c]
        # Level 1: hash(a+b), hash(c+c)
        # Level 2: hash(hash(a+b) + hash(c+c))
        ab = sha3_256_hex(a + b)
        cc = sha3_256_hex(c + c)
        expected = sha3_256_hex(ab + cc)
        assert merkle_root([a, b, c]) == expected

    def test_four_leaves(self):
        leaves = [sha3_256_hex(str(i)) for i in range(4)]
        l01 = sha3_256_hex(leaves[0] + leaves[1])
        l23 = sha3_256_hex(leaves[2] + leaves[3])
        expected = sha3_256_hex(l01 + l23)
        assert merkle_root(leaves) == expected

    def test_deterministic(self):
        leaves = [sha3_256_hex(f"item_{i}") for i in range(5)]
        r1 = merkle_root(leaves)
        r2 = merkle_root(leaves)
        assert r1 == r2

    def test_order_matters(self):
        a = sha3_256_hex("x")
        b = sha3_256_hex("y")
        assert merkle_root([a, b]) != merkle_root([b, a])

    def test_returns_64_char_hex(self):
        leaves = [sha3_256_hex(str(i)) for i in range(10)]
        root = merkle_root(leaves)
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)


# ── hash_items_to_leaves ─────────────────────────────────────────────────────

class TestHashItemsToLeaves:
    """JSON-serialize and hash a list of dicts."""

    def test_empty_list(self):
        assert hash_items_to_leaves([]) == []

    def test_single_item(self):
        items = [{"key": "value"}]
        leaves = hash_items_to_leaves(items)
        assert len(leaves) == 1
        expected = sha3_256_hex(json.dumps({"key": "value"}, sort_keys=True))
        assert leaves[0] == expected

    def test_multiple_items(self):
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        leaves = hash_items_to_leaves(items)
        assert len(leaves) == 3

    def test_sorted_keys(self):
        """Key order shouldn't matter — sort_keys=True is applied."""
        items_a = [{"z": 1, "a": 2}]
        items_b = [{"a": 2, "z": 1}]
        assert hash_items_to_leaves(items_a) == hash_items_to_leaves(items_b)

    def test_each_leaf_is_64_hex(self):
        items = [{"role": "user", "content": "hello"}]
        for leaf in hash_items_to_leaves(items):
            assert len(leaf) == 64
            assert all(c in "0123456789abcdef" for c in leaf)


# ── Cross-module consistency ──────────────────────────────────────────────────

class TestCrossModuleConsistency:
    """Verify the refactored modules produce identical Merkle roots."""

    MESSAGES = [
        {"role": "user", "content": "Test message 1"},
        {"role": "assistant", "content": "Test response 1"},
    ]
    CONSENT = {"transcript": "granted", "training_signal": "granted"}

    def _compute_root_manually(self, messages, consent):
        """Replicate the original inline algorithm for reference."""
        nodes = [
            hashlib.sha3_256(json.dumps(m, sort_keys=True).encode()).hexdigest()
            for m in messages
        ]
        nodes.append(
            hashlib.sha3_256(json.dumps(consent, sort_keys=True).encode()).hexdigest()
        )
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            nodes = [
                hashlib.sha3_256((nodes[i] + nodes[i + 1]).encode()).hexdigest()
                for i in range(0, len(nodes), 2)
            ]
        return nodes[0] if nodes else hashlib.sha3_256(b"empty").hexdigest()

    def test_utils_matches_original_algorithm(self):
        """utils.merkle must produce the same root as the original inline code."""
        expected = self._compute_root_manually(self.MESSAGES, self.CONSENT)

        leaves = hash_items_to_leaves(self.MESSAGES)
        leaves.append(sha3_256_hex(json.dumps(self.CONSENT, sort_keys=True)))
        actual = merkle_root(leaves)

        assert actual == expected

    def test_haic_tools_receipt_matches(self):
        """generate_receipt local fallback uses utils.merkle — verify consistency."""
        from unittest.mock import patch
        from tools.haic_tools import generate_receipt

        with patch("requests.post", side_effect=ConnectionError("no server")):
            result = generate_receipt("ses-test", self.MESSAGES, self.CONSENT)

        expected = self._compute_root_manually(self.MESSAGES, self.CONSENT)
        assert result["merkle_root"] == expected

    def test_maestro_local_receipt_matches(self):
        """MaestroClient._local_receipt uses utils.merkle — verify consistency."""
        from maestro_integration.maestro_client import MaestroClient
        result = MaestroClient._local_receipt("ses-test", self.MESSAGES, self.CONSENT)
        expected = self._compute_root_manually(self.MESSAGES, self.CONSENT)
        assert result["merkle_root"] == expected
