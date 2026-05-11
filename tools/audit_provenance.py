"""
audit_provenance.py — 6th HAIC governance tool: model-derivation provenance.

Wraps Cisco AI Defense's Model Provenance Kit (MPK), released 2026-05-04
under Apache-2.0 with a CC BY 4.0 reference dataset on Hugging Face:

  Blog:  https://blogs.cisco.com/ai/model-provenance-kit
  Code:  https://github.com/cisco-ai-defense/model-provenance-kit
  Data:  https://huggingface.co/datasets/cisco-ai/model-provenance-kit

MPK answers a falsifiable question: "Is model A statistically derived from
model B?" It does this via five weight-level signals (EAS, END, NLF, LEP,
WVC) combined into a composite identity score. The default Cisco-documented
tiers are:

  Pipeline Score = 1.0 or MFI Tier ≤ 2       — "Confirmed Match"
  Score > 0.75                                — "High-Confidence Match"
  0.65 < Score ≤ 0.75                         — "Weak Match"
  Score ≤ 0.65                                — "Not Matched"

Critical disclaimer (from MPK's own README):
  > MPK provides strong statistical evidence of model derivation but is NOT
  > cryptographic proof. It cannot distinguish "trained from the same template"
  > from "copied weights"; both produce high similarity scores.

## Why this is the 6th HAIC governance tool

The prior five tools (assess_wellbeing_domain, verify_consent_and_provenance,
run_prism_analysis, generate_alignment_receipt, audit_activation_explanation)
operate on inputs, metadata, model geometry, internal semantics, and the
audit trail. MPK answers a different question — "is this model what it
claims to be derived from?" That's a structural-identity check, useful
when the audit trail is incomplete or you want third-party-tool corroboration.

## Honest limits (per MPK's own README; do not soften these in summaries)

1. NOT cryptographic. Strong evidence, not absolute proof.
2. Cannot disambiguate sibling fine-tunes from copied weights.
3. Reference dataset is 908 MB; first scan downloads it.
4. Coverage is the union of what's in `cisco-ai/model-provenance-kit` —
   model families not in the dataset cannot be scanned. As of release,
   the dataset doesn't list its catalog on the HF page; coverage of
   recently-released models (including Gemma-4) is unconfirmed and the
   tool falls back gracefully.

## Integration mode

MPK runs as a CLI:
  provenancekit compare MODEL_A MODEL_B --json
  provenancekit scan MODEL_ID --json

This module wraps that CLI. We don't bind to MPK as a Python library —
its versioning is fast-moving and CLI-stable. The CLI path also lets
operators invoke MPK independently and compare results.

In tests, the subprocess is mocked. No 908 MB download happens in CI.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from utils.merkle import sha3_256_hex

logger = logging.getLogger(__name__)

# Cisco-documented score interpretation tiers (from MPK README).
# Do NOT change these without checking the upstream README; downstream
# audit consumers depend on the documented thresholds.
THRESHOLD_HIGH_CONFIDENCE = 0.75
THRESHOLD_WEAK_MATCH      = 0.65
THRESHOLD_SCAN_DEFAULT    = 0.50  # documented MPK scan-inclusion floor

VALID_VERDICTS = (
    "confirmed_match",      # pipeline_score == 1.0 or MFI tier <= 2
    "high_confidence_match", # score > 0.75
    "weak_match",            # 0.65 < score <= 0.75
    "not_matched",           # score <= 0.65
    "model_not_in_database", # MPK doesn't know one of the models
    "mpk_unavailable",       # the CLI isn't installed or failed
    "error",                 # something else went wrong
)


@dataclass
class ProvenanceCheckResult:
    """The shape consumers receive from this tool. JSON-serializable so
    it folds cleanly into a governance receipt as a Merkle leaf."""

    candidate_model: str
    reference_model: str
    verdict: str
    composite_score: Optional[float]
    five_signals: dict     # {EAS, END, NLF, LEP, WVC} → float, or {} on failure
    mpk_version: Optional[str]
    mpk_cli_path: Optional[str]
    raw_stdout_excerpt: str  # first ~500 chars of MPK output for the audit
    audit_hash: str            # SHA3-256 of the canonical payload
    disclaimer: str = (
        "MPK provides strong statistical evidence of model derivation but is "
        "NOT cryptographic proof. It cannot distinguish 'trained from the "
        "same template' from 'copied weights'."
    )
    citation: str = field(default=(
        "Cisco Systems, Inc. (2026). Model Provenance Kit. "
        "Apache-2.0 code, CC BY 4.0 reference dataset. "
        "https://github.com/cisco-ai-defense/model-provenance-kit"
    ))
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _verdict_from_score(score: Optional[float],
                         pipeline_score: Optional[float] = None,
                         mfi_tier: Optional[int] = None) -> str:
    """Apply Cisco's documented tier rules to compute a verdict string."""
    # Confirmed match takes precedence
    if pipeline_score is not None and pipeline_score >= 1.0:
        return "confirmed_match"
    if mfi_tier is not None and mfi_tier <= 2:
        return "confirmed_match"
    if score is None:
        return "error"
    if score > THRESHOLD_HIGH_CONFIDENCE:
        return "high_confidence_match"
    if score > THRESHOLD_WEAK_MATCH:
        return "weak_match"
    return "not_matched"


def _find_mpk_cli() -> Optional[str]:
    """Locate the `provenancekit` CLI on PATH. Returns None if absent."""
    return shutil.which("provenancekit")


def run_mpk_compare(
    candidate: str,
    reference: str,
    *,
    cli_runner: Optional[Callable[[list[str]], subprocess.CompletedProcess]] = None,
    cache_dir: Optional[Path] = None,
    timeout_sec: int = 600,
) -> dict:
    """Invoke `provenancekit compare CANDIDATE REFERENCE --json`.

    Args:
      candidate:  HF model id or local path of the candidate model
      reference:  HF model id or local path of the reference model
      cli_runner: optional callable that takes a command list and returns a
                  subprocess.CompletedProcess. Tests inject a mock here.
      cache_dir:  optional override for MPK's cache directory (set via env).
      timeout_sec: subprocess timeout.

    Returns: parsed JSON dict from MPK's stdout, with a `_meta` key added
             that records `mpk_version`, `cli_path`, `stdout_excerpt`.
    """
    if cli_runner is None:
        cli_path = _find_mpk_cli()
        if cli_path is None:
            raise FileNotFoundError(
                "provenancekit CLI not found on PATH. "
                "Install with: pip install provenancekit  (Python 3.12+)"
            )
        def _runner(cmd: list[str]) -> subprocess.CompletedProcess:
            env = os.environ.copy()
            if cache_dir is not None:
                env["MPK_CACHE_DIR"] = str(cache_dir)
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_sec, env=env)
        cli_runner = _runner
        cli_path_for_meta: Optional[str] = cli_path
    else:
        cli_path_for_meta = "<injected runner>"

    cmd = ["provenancekit", "compare", candidate, reference, "--json"]
    proc = cli_runner(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"provenancekit compare exited with code {proc.returncode}. "
            f"stderr: {proc.stderr[:500]!r}"
        )

    # MPK is documented to emit JSON on stdout when --json is passed.
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"MPK returned non-JSON output despite --json: {e}. "
            f"stdout: {proc.stdout[:300]!r}"
        ) from e

    # Try to glean the MPK version (some CLIs print it to stderr; some
    # include it in the JSON payload).
    mpk_version = (
        payload.get("mpk_version")
        or payload.get("version")
        or proc.stderr.split("provenancekit ")[-1].split("\n")[0]
        if "provenancekit " in proc.stderr
        else None
    )
    payload["_meta"] = {
        "mpk_version": mpk_version,
        "cli_path": cli_path_for_meta,
        "stdout_excerpt": proc.stdout[:500],
        "stderr_excerpt": proc.stderr[:200],
    }
    return payload


def execute_audit_provenance(args: dict) -> dict:
    """Tool entry point used by the Gemma 4 function-calling pipeline.

    Args:
      candidate_model:   str — the model being checked (e.g. "haic-gemma4-v42")
      reference_model:   str — the alleged parent (e.g. "google/gemma-4-e2b-it")
      cli_runner:        optional injected subprocess wrapper (tests)
      cache_dir:         optional override for MPK cache directory
      enabled:           if False, returns an "mpk_unavailable" stub without
                         attempting subprocess. Lets the notebook gate the
                         tool behind a feature flag.

    Returns: ProvenanceCheckResult.to_dict()
    """
    candidate = args.get("candidate_model", "")
    reference = args.get("reference_model", "")
    enabled = args.get("enabled", True)
    cli_runner = args.get("cli_runner")
    cache_dir = args.get("cache_dir")

    notes: list[str] = []

    if not enabled:
        result = ProvenanceCheckResult(
            candidate_model=candidate,
            reference_model=reference,
            verdict="mpk_unavailable",
            composite_score=None,
            five_signals={},
            mpk_version=None,
            mpk_cli_path=None,
            raw_stdout_excerpt="",
            audit_hash="",
            notes=["MPK disabled by feature flag (MPK_ENABLED=False)"],
        )
        result.audit_hash = _hash_result(result)
        return result.to_dict()

    # Try to invoke MPK
    try:
        payload = run_mpk_compare(
            candidate, reference,
            cli_runner=cli_runner, cache_dir=cache_dir,
        )
    except FileNotFoundError as e:
        result = ProvenanceCheckResult(
            candidate_model=candidate,
            reference_model=reference,
            verdict="mpk_unavailable",
            composite_score=None,
            five_signals={},
            mpk_version=None,
            mpk_cli_path=None,
            raw_stdout_excerpt="",
            audit_hash="",
            notes=[
                f"MPK CLI not found: {e}",
                "Install MPK with: pip install provenancekit  (Python 3.12+)",
                "Falling back to PRISM geometry signature for derivation evidence.",
            ],
        )
        result.audit_hash = _hash_result(result)
        return result.to_dict()
    except RuntimeError as e:
        # Distinguish "model not in MPK's reference database" from other
        # runtime errors. MPK's stderr typically says "not found in
        # deep-signals dataset" or similar in this case.
        err_str = str(e).lower()
        if "not found" in err_str or "not in" in err_str or "unknown" in err_str:
            verdict = "model_not_in_database"
            notes.append(
                "One or both models are not in MPK's reference dataset "
                "(version cisco-ai/model-provenance-kit). If the model is "
                "recently released, it may not yet be fingerprinted. "
                "Falling back to PRISM geometry signature for derivation "
                "evidence."
            )
        else:
            verdict = "error"
            notes.append(f"MPK runtime error: {e}")

        result = ProvenanceCheckResult(
            candidate_model=candidate,
            reference_model=reference,
            verdict=verdict,
            composite_score=None,
            five_signals={},
            mpk_version=None,
            mpk_cli_path=_find_mpk_cli(),
            raw_stdout_excerpt="",
            audit_hash="",
            notes=notes,
        )
        result.audit_hash = _hash_result(result)
        return result.to_dict()

    # Parse the successful payload
    score = payload.get("composite_score") or payload.get("score")
    pipeline_score = payload.get("pipeline_score")
    mfi_tier = payload.get("mfi_tier")
    if score is not None:
        score = float(score)

    five_signals_raw = payload.get("signals") or payload.get("five_signals") or {}
    # Normalize signal keys to upper-case acronym form
    five_signals = {}
    for k, v in five_signals_raw.items():
        key_upper = k.upper()
        if key_upper in ("EAS", "END", "NLF", "LEP", "WVC"):
            try:
                five_signals[key_upper] = float(v)
            except (TypeError, ValueError):
                five_signals[key_upper] = None

    verdict = _verdict_from_score(score, pipeline_score=pipeline_score, mfi_tier=mfi_tier)
    meta = payload.get("_meta", {})

    result = ProvenanceCheckResult(
        candidate_model=candidate,
        reference_model=reference,
        verdict=verdict,
        composite_score=score,
        five_signals=five_signals,
        mpk_version=meta.get("mpk_version"),
        mpk_cli_path=meta.get("cli_path"),
        raw_stdout_excerpt=meta.get("stdout_excerpt", ""),
        audit_hash="",
        notes=notes,
    )
    result.audit_hash = _hash_result(result)
    return result.to_dict()


def _hash_result(result: ProvenanceCheckResult) -> str:
    """SHA3-256 over the canonical payload (excluding audit_hash itself)."""
    payload = {
        "candidate_model":   result.candidate_model,
        "reference_model":   result.reference_model,
        "verdict":           result.verdict,
        "composite_score":   result.composite_score,
        "five_signals":      result.five_signals,
        "mpk_version":       result.mpk_version,
    }
    return sha3_256_hex(json.dumps(payload, sort_keys=True))


# ── Gemma 4 native function-calling schema ─────────────────────────────────


TOOL_SCHEMA = {
    "name": "audit_provenance",
    "description": (
        "Use Cisco's Model Provenance Kit (MPK) to verify whether a "
        "candidate model is statistically derived from a reference model. "
        "Returns the composite identity score (0-1), the five weight-level "
        "signals (EAS/END/NLF/LEP/WVC), and a verdict using Cisco's "
        "documented tiers (>0.75 = high-confidence match, 0.65-0.75 = weak "
        "match, ≤0.65 = not matched). NOT cryptographic proof — strong "
        "statistical evidence only. Gracefully falls back if MPK isn't "
        "installed or the models aren't in MPK's reference dataset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_model": {
                "type": "string",
                "description": "HF model id or local path of the model "
                               "being checked (e.g. 'haic-gemma4-v42')",
            },
            "reference_model": {
                "type": "string",
                "description": "HF model id of the alleged parent model "
                               "(e.g. 'google/gemma-4-e2b-it')",
            },
            "enabled": {
                "type": "boolean",
                "description": "If false, tool returns 'mpk_unavailable' "
                               "without attempting the 908 MB dataset "
                               "download. Default true.",
                "default": True,
            },
        },
        "required": ["candidate_model", "reference_model"],
    },
}
