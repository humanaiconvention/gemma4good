"""
Tests for tools/eval_receipt.py — Merkle eval-receipt construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_receipt import (  # noqa: E402
    sha256_text, sha256_canonical, merkle_root,
    scenario_leaves, aggregate_summary_hash, build_receipt,
)


# ── Hash + Merkle primitives ────────────────────────────────────────────────


class TestSha256:

    def test_text_deterministic(self):
        assert sha256_text("x") == sha256_text("x")

    def test_canonical_sort_keys(self):
        a = sha256_canonical({"a": 1, "b": 2})
        b = sha256_canonical({"b": 2, "a": 1})
        assert a == b


class TestMerkleRoot:

    def test_empty_returns_null_hash(self):
        # Receipt of an empty list gets a deterministic non-empty hash
        r = merkle_root([])
        assert r == sha256_text("empty")

    def test_single_leaf_returns_leaf(self):
        # Single-leaf root collapses to the leaf hash
        r = merkle_root(["abc"])
        assert r == "abc"

    def test_two_leaves_pair(self):
        r = merkle_root(["a", "b"])
        assert r == sha256_text("ab")

    def test_three_leaves_carry_odd(self):
        # ["a","b","c"] → [hash(ab), hash(cc)] → hash(hash(ab)+hash(cc))
        ab = sha256_text("ab")
        cc = sha256_text("cc")
        expected = sha256_text(ab + cc)
        assert merkle_root(["a", "b", "c"]) == expected

    def test_deterministic(self):
        leaves = [str(i) for i in range(7)]
        assert merkle_root(leaves) == merkle_root(leaves)

    def test_input_order_matters(self):
        # Reordered leaves should produce different roots
        assert merkle_root(["a", "b"]) != merkle_root(["b", "a"])


# ── Scenario leaves ─────────────────────────────────────────────────────────


class TestScenarioLeaves:

    def test_each_record_produces_one_leaf(self):
        records = [
            {"scenario_id": "x", "kind": "grounding", "result": "PASS",
             "has_pivot": True, "complied": None, "stayed_in_protocol": None,
             "seed": 42, "response_preview": "..."},
            {"scenario_id": "y", "kind": "security", "result": "FAIL",
             "has_pivot": False, "complied": True, "stayed_in_protocol": False,
             "seed": 43, "response_preview": "..."},
        ]
        leaves = scenario_leaves(records)
        assert len(leaves) == 2
        assert all("leaf_hash" in l for l in leaves)
        assert leaves[0]["leaf_hash"] != leaves[1]["leaf_hash"]

    def test_response_text_changes_leaf(self):
        records_a = [{"scenario_id": "x", "result": "PASS",
                      "response_preview": "alpha"}]
        records_b = [{"scenario_id": "x", "result": "PASS",
                      "response_preview": "beta"}]
        a = scenario_leaves(records_a)
        b = scenario_leaves(records_b)
        assert a[0]["leaf_hash"] != b[0]["leaf_hash"]

    def test_seed_changes_leaf(self):
        records_a = [{"scenario_id": "x", "result": "PASS",
                      "response_preview": "x", "seed": 1}]
        records_b = [{"scenario_id": "x", "result": "PASS",
                      "response_preview": "x", "seed": 2}]
        a = scenario_leaves(records_a)
        b = scenario_leaves(records_b)
        assert a[0]["leaf_hash"] != b[0]["leaf_hash"]


# ── Aggregate hash ──────────────────────────────────────────────────────────


class TestAggregateSummaryHash:

    def _record(self):
        return {
            "pass_type": "sampling",
            "n_per_scenario": 10,
            "grounding_passes": 11,
            "grounding_trials": 30,
            "grounding_pass_rate": 0.367,
            "grounding_ci95": [0.22, 0.54],
            "security_pass_rate": 0.0,
            "seed": 42,
            "model_id": "test",
            "decoding": {"temperature": 0.7},
        }

    def test_deterministic(self):
        r = self._record()
        assert aggregate_summary_hash(r) == aggregate_summary_hash(r)

    def test_metric_change_changes_hash(self):
        a = self._record()
        b = dict(a); b["grounding_passes"] = 12
        assert aggregate_summary_hash(a) != aggregate_summary_hash(b)

    def test_seed_change_changes_hash(self):
        a = self._record()
        b = dict(a); b["seed"] = 100
        assert aggregate_summary_hash(a) != aggregate_summary_hash(b)


# ── Full receipt ────────────────────────────────────────────────────────────


def _make_min_report():
    """Minimal SGT report skeleton."""
    pass_record = {
        "pass_type": "sampling", "n_per_scenario": 5,
        "grounding_passes": 3, "grounding_trials": 10,
        "security_passes": 5, "security_trials": 5, "security_fails": 0,
        "grounding_pass_rate": 0.3, "grounding_ci95": [0.1, 0.6],
        "security_pass_rate": 1.0, "security_ci95": [0.5, 1.0],
        "sgt_score_out_of_10": 3.0,
        "per_scenario": [
            {"scenario_id": "x", "kind": "grounding", "result": "PASS",
             "has_pivot": True, "seed": 1, "response_preview": "[PIVOT: DEEPEN] ..."},
        ],
        "seed": 42, "model_id": "test", "decoding": {"temperature": 0.7},
    }
    det = dict(pass_record); det["pass_type"] = "deterministic"; det["n_per_scenario"] = 1
    return {"finetune": {"deterministic": det, "sampling": pass_record}}


class TestBuildReceipt:

    def test_root_is_64_hex(self):
        rep = _make_min_report()
        r = build_receipt(rep)
        assert len(r["eval_receipt_root"]) == 64
        assert all(c in "0123456789abcdef" for c in r["eval_receipt_root"])

    def test_root_changes_when_response_changes(self):
        a = _make_min_report()
        b = _make_min_report()
        b["finetune"]["sampling"]["per_scenario"][0]["response_preview"] = "different"
        ra = build_receipt(a)
        rb = build_receipt(b)
        assert ra["eval_receipt_root"] != rb["eval_receipt_root"]

    def test_root_changes_when_decoding_changes(self):
        a = _make_min_report()
        b = _make_min_report()
        b["finetune"]["sampling"]["decoding"] = {"temperature": 0.9}
        ra = build_receipt(a)
        rb = build_receipt(b)
        assert ra["eval_receipt_root"] != rb["eval_receipt_root"]

    def test_leakage_present_changes_root(self):
        a_rep = _make_min_report()
        ra_no_leak = build_receipt(a_rep, leakage_receipt=None)
        ra_with_leak = build_receipt(a_rep, leakage_receipt={"verdict": "PASS"})
        assert ra_no_leak["eval_receipt_root"] != ra_with_leak["eval_receipt_root"]

    def test_decision_present_changes_root(self):
        a_rep = _make_min_report()
        ra_no = build_receipt(a_rep, decision_receipt=None)
        ra_yes = build_receipt(a_rep, decision_receipt={"decision": "BLOCKED"})
        assert ra_no["eval_receipt_root"] != ra_yes["eval_receipt_root"]

    def test_baseline_present_changes_root(self):
        a = _make_min_report()
        b = _make_min_report()
        b["baseline"] = b["finetune"]
        ra = build_receipt(a)
        rb = build_receipt(b)
        assert ra["eval_receipt_root"] != rb["eval_receipt_root"]

    def test_leaf_counts_correct(self):
        rep = _make_min_report()
        r = build_receipt(rep)
        assert r["leaf_counts"]["finetune_sampling"] == 1
        assert r["leaf_counts"]["finetune_deterministic"] == 1
        assert r["leaf_counts"]["baseline_sampling"] == 0

    def test_idempotent(self):
        rep = _make_min_report()
        a = build_receipt(rep)
        b = build_receipt(rep)
        assert a["eval_receipt_root"] == b["eval_receipt_root"]
