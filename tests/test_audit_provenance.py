"""Tests for tools/audit_provenance.py — Cisco MPK governance tool.

All tests mock the MPK CLI via the `cli_runner` injection point. NO 908 MB
download happens in CI. The mocked responses match the schema Cisco's MPK
documents (composite_score, signals, optional pipeline_score / mfi_tier).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audit_provenance import (
    THRESHOLD_HIGH_CONFIDENCE,
    THRESHOLD_WEAK_MATCH,
    TOOL_SCHEMA,
    ProvenanceCheckResult,
    _verdict_from_score,
    execute_audit_provenance,
)


# ── Helper: build a stub CLI runner ────────────────────────────────────────


def _stub_runner(payload: dict, returncode: int = 0,
                 stderr: str = "") -> "subprocess.CompletedProcess":
    """Return a callable that simulates `provenancekit compare ... --json`."""
    def runner(cmd: list[str]) -> subprocess.CompletedProcess:
        proc = subprocess.CompletedProcess(
            args=cmd, returncode=returncode,
            stdout=json.dumps(payload), stderr=stderr,
        )
        return proc
    return runner


def _stub_runner_error(stderr: str,
                        returncode: int = 1) -> "subprocess.CompletedProcess":
    """Stub a non-zero MPK exit (CLI error path)."""
    def runner(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout="", stderr=stderr,
        )
    return runner


# ── _verdict_from_score: Cisco's documented tiers ──────────────────────────


def test_verdict_high_confidence_at_score_above_075():
    assert _verdict_from_score(0.80) == "high_confidence_match"
    assert _verdict_from_score(0.95) == "high_confidence_match"
    # Boundary: > 0.75 means strictly above, not >=
    assert _verdict_from_score(0.751) == "high_confidence_match"
    # 0.75 exactly is NOT high confidence (per Cisco's docs: "> 0.75")
    assert _verdict_from_score(0.75) == "weak_match"


def test_verdict_weak_match_in_065_to_075_band():
    assert _verdict_from_score(0.70) == "weak_match"
    assert _verdict_from_score(0.651) == "weak_match"
    # 0.65 is NOT a weak match (per Cisco's docs: "> 0.65")
    assert _verdict_from_score(0.65) == "not_matched"


def test_verdict_not_matched_below_065():
    assert _verdict_from_score(0.50) == "not_matched"
    assert _verdict_from_score(0.0) == "not_matched"


def test_verdict_confirmed_match_via_pipeline_score():
    """pipeline_score == 1.0 overrides composite score."""
    assert _verdict_from_score(0.30, pipeline_score=1.0) == "confirmed_match"


def test_verdict_confirmed_match_via_mfi_tier():
    """mfi_tier <= 2 overrides composite score."""
    assert _verdict_from_score(0.30, mfi_tier=1) == "confirmed_match"
    assert _verdict_from_score(0.30, mfi_tier=2) == "confirmed_match"
    # tier > 2 doesn't trigger confirmed
    assert _verdict_from_score(0.50, mfi_tier=3) == "not_matched"


def test_verdict_none_score_returns_error():
    assert _verdict_from_score(None) == "error"


# ── execute_audit_provenance happy paths ────────────────────────────────────


def test_high_confidence_match_path():
    runner = _stub_runner({
        "composite_score": 0.88,
        "signals": {
            "EAS": 0.92, "END": 0.85, "NLF": 0.90, "LEP": 0.86, "WVC": 0.87,
        },
        "mpk_version": "1.0.0",
    })
    result = execute_audit_provenance({
        "candidate_model": "haic-gemma4-v42",
        "reference_model": "google/gemma-4-e2b-it",
        "cli_runner": runner,
    })
    assert result["verdict"] == "high_confidence_match"
    assert result["composite_score"] == 0.88
    assert set(result["five_signals"].keys()) == {"EAS", "END", "NLF", "LEP", "WVC"}
    assert result["five_signals"]["EAS"] == 0.92
    assert result["audit_hash"] != ""


def test_weak_match_path():
    runner = _stub_runner({
        "composite_score": 0.70,
        "signals": {"EAS": 0.7, "END": 0.7, "NLF": 0.7, "LEP": 0.7, "WVC": 0.7},
    })
    result = execute_audit_provenance({
        "candidate_model": "haic-gemma4-v42",
        "reference_model": "google/gemma-4-e2b-it",
        "cli_runner": runner,
    })
    assert result["verdict"] == "weak_match"


def test_not_matched_path():
    runner = _stub_runner({
        "composite_score": 0.30,
        "signals": {"EAS": 0.3, "END": 0.3, "NLF": 0.3, "LEP": 0.3, "WVC": 0.3},
    })
    result = execute_audit_provenance({
        "candidate_model": "haic-gemma4-v42",
        "reference_model": "meta-llama/Llama-3.1-8B",  # unrelated reference
        "cli_runner": runner,
    })
    assert result["verdict"] == "not_matched"


def test_confirmed_match_via_pipeline_score():
    runner = _stub_runner({
        "composite_score": 0.50,
        "pipeline_score": 1.0,    # forces confirmed regardless of composite
        "signals": {"EAS": 0.5, "END": 0.5, "NLF": 0.5, "LEP": 0.5, "WVC": 0.5},
    })
    result = execute_audit_provenance({
        "candidate_model": "haic-gemma4-v42",
        "reference_model": "google/gemma-4-e2b-it",
        "cli_runner": runner,
    })
    assert result["verdict"] == "confirmed_match"


# ── Fallback / unavailable / error paths ───────────────────────────────────


def test_enabled_false_returns_mpk_unavailable_stub():
    """The notebook gates MPK behind a feature flag. enabled=False must NOT
    invoke any CLI and must return the 'mpk_unavailable' verdict."""
    def boom_runner(cmd):
        raise AssertionError("cli_runner must not be called when enabled=False")
    result = execute_audit_provenance({
        "candidate_model": "x",
        "reference_model": "y",
        "enabled": False,
        "cli_runner": boom_runner,
    })
    assert result["verdict"] == "mpk_unavailable"
    assert "MPK_ENABLED=False" in " ".join(result["notes"])


def test_model_not_in_database_path():
    """When MPK can't find a model in its reference dataset, the verdict
    is 'model_not_in_database' (a graceful degradation, not an error).
    This is the expected case for Gemma-4 until Cisco adds it."""
    runner = _stub_runner_error(
        stderr="Error: google/gemma-4-e2b-it not found in deep-signals dataset",
    )
    result = execute_audit_provenance({
        "candidate_model": "haic-gemma4-v42",
        "reference_model": "google/gemma-4-e2b-it",
        "cli_runner": runner,
    })
    assert result["verdict"] == "model_not_in_database"
    # The fallback note must mention PRISM as a backup signal
    joined_notes = " ".join(result["notes"])
    assert "PRISM" in joined_notes or "prism" in joined_notes


def test_runtime_error_other_than_not_found():
    """An unrelated runtime error gets the 'error' verdict."""
    runner = _stub_runner_error(stderr="Error: cache directory is read-only")
    result = execute_audit_provenance({
        "candidate_model": "x", "reference_model": "y",
        "cli_runner": runner,
    })
    assert result["verdict"] == "error"


def test_non_json_output_raises_through_to_error_verdict():
    """If MPK returns garbage (non-JSON), the wrapper surfaces an error."""
    def garbage_runner(cmd):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="not json at all", stderr="",
        )
    result = execute_audit_provenance({
        "candidate_model": "x", "reference_model": "y",
        "cli_runner": garbage_runner,
    })
    assert result["verdict"] == "error"


# ── Hash + receipt-compatibility checks ────────────────────────────────────


def test_audit_hash_changes_with_score():
    """Different composite scores produce different audit_hash values
    (so tampering with a result is detectable)."""
    runner_a = _stub_runner({
        "composite_score": 0.88,
        "signals": {"EAS": 0.9, "END": 0.9, "NLF": 0.9, "LEP": 0.9, "WVC": 0.9},
    })
    runner_b = _stub_runner({
        "composite_score": 0.30,
        "signals": {"EAS": 0.9, "END": 0.9, "NLF": 0.9, "LEP": 0.9, "WVC": 0.9},
    })
    args = {"candidate_model": "c", "reference_model": "r"}
    r_a = execute_audit_provenance({**args, "cli_runner": runner_a})
    r_b = execute_audit_provenance({**args, "cli_runner": runner_b})
    assert r_a["audit_hash"] != r_b["audit_hash"]


def test_result_is_json_serializable():
    """Merkle-leaf compatibility: the result must round-trip JSON."""
    runner = _stub_runner({
        "composite_score": 0.80,
        "signals": {"EAS": 0.8, "END": 0.8, "NLF": 0.8, "LEP": 0.8, "WVC": 0.8},
    })
    result = execute_audit_provenance({
        "candidate_model": "x", "reference_model": "y",
        "cli_runner": runner,
    })
    blob = json.dumps(result, sort_keys=True)
    round_tripped = json.loads(blob)
    assert round_tripped == result


def test_disclaimer_is_present_and_says_not_cryptographic():
    """The non-cryptographic-proof disclaimer must always appear,
    regardless of score."""
    runner = _stub_runner({
        "composite_score": 1.0,
        "signals": {"EAS": 1.0, "END": 1.0, "NLF": 1.0, "LEP": 1.0, "WVC": 1.0},
    })
    result = execute_audit_provenance({
        "candidate_model": "x", "reference_model": "y",
        "cli_runner": runner,
    })
    assert "NOT cryptographic" in result["disclaimer"]
    assert "strong statistical evidence" in result["disclaimer"].lower()


def test_citation_is_present_and_links_to_cisco_repo():
    """Apache-2.0 + CC BY 4.0 require attribution. Verify the citation."""
    runner = _stub_runner({
        "composite_score": 0.80,
        "signals": {"EAS": 0.8, "END": 0.8, "NLF": 0.8, "LEP": 0.8, "WVC": 0.8},
    })
    result = execute_audit_provenance({
        "candidate_model": "x", "reference_model": "y",
        "cli_runner": runner,
    })
    assert "Cisco" in result["citation"]
    assert "github.com/cisco-ai-defense/model-provenance-kit" in result["citation"]
    assert "CC BY 4.0" in result["citation"] or "Apache" in result["citation"]


# ── Tool schema ────────────────────────────────────────────────────────────


def test_tool_schema_correct_shape():
    assert TOOL_SCHEMA["name"] == "audit_provenance"
    assert "description" in TOOL_SCHEMA
    params = TOOL_SCHEMA["parameters"]
    assert "candidate_model" in params["required"]
    assert "reference_model" in params["required"]


def test_tool_schema_includes_disclaimer():
    """The tool's description must surface that this is NOT cryptographic
    so the agent in the function-calling loop doesn't over-claim."""
    desc = TOOL_SCHEMA["description"]
    assert "NOT cryptographic" in desc
    assert "statistical evidence" in desc.lower()


# ── Threshold sanity ────────────────────────────────────────────────────────


def test_documented_thresholds_match_cisco_readme():
    """If these change, MPK's docs probably changed. Check upstream."""
    assert THRESHOLD_HIGH_CONFIDENCE == 0.75
    assert THRESHOLD_WEAK_MATCH == 0.65
