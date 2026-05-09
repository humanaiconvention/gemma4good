"""Tests for experiments/scenarios_loader.py.

Pure-Python — exercises JSONL → SgtScenario parsing, canonical hashing,
and the equivalence check against Garrett's DEFAULT_SCENARIOS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scenarios_loader import (  # noqa: E402
    load_scenarios_jsonl,
    scenarios_hash,
    hash_scenarios_file,
    equivalent_to_harness_default,
    _DEFAULT_PATH,
)
from experiments.sgt_harness import SgtScenario


# ── load_scenarios_jsonl ────────────────────────────────────────────────────


class TestLoadScenariosJSONL:

    def test_default_file_loads(self):
        scs = load_scenarios_jsonl()
        assert len(scs) == 5
        assert all(isinstance(s, SgtScenario) for s in scs)

    def test_loaded_ids_match_harness_defaults(self):
        from experiments.sgt_harness import DEFAULT_SCENARIOS
        a = sorted([s.id for s in load_scenarios_jsonl()])
        b = sorted([s.id for s in DEFAULT_SCENARIOS])
        assert a == b

    def test_kind_field_preserved(self):
        scs = load_scenarios_jsonl()
        kinds = {s.id: s.kind for s in scs}
        assert kinds["sgt_basic_grounding"] == "grounding"
        assert kinds["sgt_adversarial_inject"] == "security"

    def test_missing_file_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_scenarios_jsonl(tmp_path / "nope.jsonl")

    def test_handcrafted_jsonl_loads(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(
            json.dumps({
                "id": "test_a",
                "user_msg": "hi there",
                "kind": "grounding",
                "expected_pivot": True,
                "version": "1.0",
            }) + "\n"
            + json.dumps({
                "id": "test_b",
                "user_msg": "another",
                "kind": "security",
                "expected_pivot": False,
                "version": "1.0",
            }) + "\n"
        )
        scs = load_scenarios_jsonl(f)
        assert len(scs) == 2
        assert {s.id for s in scs} == {"test_a", "test_b"}

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(
            "\n\n"
            + json.dumps({"id": "a", "user_msg": "x", "kind": "grounding"}) + "\n"
            + "\n"
        )
        scs = load_scenarios_jsonl(f)
        assert len(scs) == 1


# ── scenarios_hash ──────────────────────────────────────────────────────────


class TestScenariosHash:

    def test_deterministic_on_dataclass_input(self):
        scs = load_scenarios_jsonl()
        h1 = scenarios_hash(scs)
        h2 = scenarios_hash(scs)
        assert h1 == h2

    def test_deterministic_on_dict_input(self):
        scs_dict = [
            {"id": "a", "user_msg": "x", "kind": "grounding"},
            {"id": "b", "user_msg": "y", "kind": "security"},
        ]
        h1 = scenarios_hash(scs_dict)
        h2 = scenarios_hash(scs_dict)
        assert h1 == h2

    def test_dict_and_dataclass_produce_same_hash(self):
        scs = load_scenarios_jsonl()
        scs_dict = [{"id": s.id, "user_msg": s.user_msg, "kind": s.kind} for s in scs]
        assert scenarios_hash(scs) == scenarios_hash(scs_dict)

    def test_order_invariant(self):
        scs1 = [
            {"id": "a", "user_msg": "x", "kind": "grounding"},
            {"id": "b", "user_msg": "y", "kind": "security"},
        ]
        scs2 = list(reversed(scs1))
        assert scenarios_hash(scs1) == scenarios_hash(scs2)

    def test_change_in_user_msg_changes_hash(self):
        a = [{"id": "x", "user_msg": "hello", "kind": "grounding"}]
        b = [{"id": "x", "user_msg": "hello world", "kind": "grounding"}]
        assert scenarios_hash(a) != scenarios_hash(b)

    def test_change_in_kind_changes_hash(self):
        a = [{"id": "x", "user_msg": "hi", "kind": "grounding"}]
        b = [{"id": "x", "user_msg": "hi", "kind": "security"}]
        assert scenarios_hash(a) != scenarios_hash(b)

    def test_default_jsonl_hash_is_stable(self):
        # Pin the canonical hash so future drift in the JSONL file is loud.
        # If you change a scenario, this test should fail and you update it.
        h = hash_scenarios_file()
        assert h == "3276e6c7841bd01711098f6a899d571dfbbf6d806d31152c439130bdd4ef5ec8", (
            f"sgt_scenarios.jsonl hash drifted to {h!r}. "
            "If this is intentional, update this test with the new hash."
        )

    def test_hash_is_sha3_256_hex(self):
        h = hash_scenarios_file()
        assert len(h) == 64
        int(h, 16)  # raises if not valid hex


# ── equivalent_to_harness_default ───────────────────────────────────────────


class TestEquivalence:

    def test_jsonl_matches_harness(self):
        scs = load_scenarios_jsonl()
        assert equivalent_to_harness_default(scs)

    def test_modified_jsonl_does_not_match(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps({
            "id": "different_id",
            "user_msg": "different msg",
            "kind": "grounding",
        }) + "\n")
        scs = load_scenarios_jsonl(f)
        assert not equivalent_to_harness_default(scs)
