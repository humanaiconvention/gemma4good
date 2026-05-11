"""
Tests for tools/eval_leakage_check.py.

Pure-Python — exercises scenario parsing, hashing, n-gram jaccard, and
the verdict aggregation. Independent of any model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_leakage_check import (  # noqa: E402
    sha256_text,
    extract_scenarios_from_harness,
    check_exact_match,
    check_near_match,
    _ngrams,
    _jaccard,
    _normalize,
    load_shard,
)


# ── sha256_text ─────────────────────────────────────────────────────────────


class TestSha256:

    def test_deterministic(self):
        assert sha256_text("hello") == sha256_text("hello")

    def test_distinct(self):
        assert sha256_text("a") != sha256_text("b")


# ── n-gram + jaccard ────────────────────────────────────────────────────────


class TestNgrams:

    def test_basic_5gram(self):
        s = "the quick brown fox jumps over the lazy dog"
        grams = _ngrams(s, n=5)
        # 9 tokens → 5 5-grams
        assert len(grams) == 5

    def test_short_text_returns_full(self):
        grams = _ngrams("hi there", n=5)
        # Falls back to single concatenation
        assert grams == {"hi there"}

    def test_empty_returns_empty(self):
        assert _ngrams("", n=5) == set()

    def test_normalize_strips_punctuation(self):
        assert _normalize("Hello, world!") == "hello world"


class TestJaccard:

    def test_identical_sets_one(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets_zero(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        # |∩|=1, |∪|=3 → 1/3
        score = _jaccard({"a", "b"}, {"b", "c"})
        assert abs(score - 1/3) < 1e-9


# ── exact-match ─────────────────────────────────────────────────────────────


class TestExactMatch:

    def test_substring_match_passes(self):
        assert check_exact_match("foo bar", "this has foo bar in it")

    def test_no_match(self):
        assert not check_exact_match("foo bar", "nothing here")

    def test_whitespace_stripped(self):
        assert check_exact_match("  foo  ", "this has foo in it")


# ── near-match ──────────────────────────────────────────────────────────────


class TestNearMatch:

    def test_high_similarity_above_threshold(self):
        scenario = "I use AI to manage my team's schedules every day"
        line = "I use AI to manage my team's schedules every single day"
        passes, score, _ = check_near_match(scenario, [line], threshold=0.4)
        assert passes
        assert score > 0.4

    def test_low_similarity_below_threshold(self):
        scenario = "I use AI to manage my team's schedules every day"
        line = "Pizza is a great food for sharing with friends"
        passes, score, _ = check_near_match(scenario, [line], threshold=0.4)
        assert not passes

    def test_short_lines_skipped(self):
        scenario = "I use AI to manage my team's schedules every day"
        # All lines below 20 chars
        passes, score, _ = check_near_match(scenario, ["short", "tiny", "x"])
        assert score == 0.0

    def test_empty_scenario_returns_zero(self):
        passes, score, _ = check_near_match("", ["any line at all here ok"])
        assert not passes
        assert score == 0.0


# ── shard loading ───────────────────────────────────────────────────────────


class TestLoadShard:

    def test_jsonl_extracts_message_content(self, tmp_path):
        shard = tmp_path / "x.jsonl"
        shard.write_text(json.dumps({
            "messages": [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "user text"},
                {"role": "assistant", "content": "assistant text"},
            ]
        }) + "\n")
        full, lines, h = load_shard(shard)
        assert "system text" in lines
        assert "user text" in lines
        assert "assistant text" in lines
        assert len(h) == 64  # SHA3-256 hex

    def test_jsonl_handles_multimodal_content(self, tmp_path):
        shard = tmp_path / "x.jsonl"
        shard.write_text(json.dumps({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image", "url": "file.png"},
                ]}
            ]
        }) + "\n")
        full, lines, h = load_shard(shard)
        assert "look at this" in lines

    def test_plain_text(self, tmp_path):
        shard = tmp_path / "x.txt"
        shard.write_text("first line\nsecond line\n")
        full, lines, h = load_shard(shard)
        assert lines == ["first line", "second line"]


# ── scenario extraction from harness ────────────────────────────────────────


class TestScenarioExtraction:

    def test_real_harness_parses(self):
        # Smoke test against the actual sgt_harness.py — it had better parse.
        scenarios = extract_scenarios_from_harness(
            ROOT / "experiments" / "sgt_harness.py"
        )
        assert len(scenarios) >= 5
        ids = {s["id"] for s in scenarios}
        # Default scenarios from Garrett's commit
        assert "sgt_basic_grounding" in ids
        assert "sgt_emotional" in ids
        assert "sgt_compression_trigger" in ids
        assert "sgt_adversarial_inject" in ids
        assert "sgt_indirect_inject" in ids

    def test_each_scenario_has_kind(self):
        scenarios = extract_scenarios_from_harness(
            ROOT / "experiments" / "sgt_harness.py"
        )
        for s in scenarios:
            assert s["kind"] in {"grounding", "security"}

    def test_handcrafted_input(self, tmp_path):
        f = tmp_path / "fake_harness.py"
        f.write_text(
            'SgtScenario(\n'
            '    id="x_test",\n'
            '    user_msg="hello world",\n'
            '    kind="grounding",\n'
            '    expected_pivot=True,\n'
            '),\n'
            'SgtScenario(\n'
            '    id="y_test",\n'
            '    user_msg="another scenario",\n'
            '    kind="security",\n'
            '    expected_pivot=False,\n'
            ')\n'
        )
        scenarios = extract_scenarios_from_harness(f)
        assert len(scenarios) == 2
        assert {s["id"] for s in scenarios} == {"x_test", "y_test"}
