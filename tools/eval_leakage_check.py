"""
eval_leakage_check.py — mechanize evaluation Gate 2 (eval-set leakage risk).

The HAIC evaluation doctrine ([docs/evaluation_doctrine.md]) requires that
every promotion run answer the question: are any of the eval scenarios
present, verbatim or as near-paraphrase, in the training data?

This tool produces a leakage receipt:
  - SHA-256 over each scenario id+body (sorted, canonicalized)
  - SHA-256 over each training shard
  - exact-match check (any scenario user_msg appears verbatim in any shard)
  - near-match heuristic (jaccard on normalized 5-grams ≥ 0.6)

Usage:
    python -m tools.eval_leakage_check \\
        --scenarios experiments/sgt_harness.py \\
        --training data/v35_gov_final.jsonl data/v38_synthetic.jsonl \\
        --out experiments/v38_leakage_receipt.json

Exit code:
  0  no leakage detected
  1  exact-match leakage detected (BLOCKING)
  2  near-match leakage suspected (review required)

The output JSON is the receipt; commit it alongside the eval report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


# ── Hashing utilities ────────────────────────────────────────────────────────


def sha256_text(s: str) -> str:
    """Hash a string. Named ``sha256_text`` for callsite compatibility, but
    actually computes SHA3-256 — the canonical HAIC hash for receipt
    interoperability with utils/merkle.py."""
    return hashlib.sha3_256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """File hash. Same naming/algorithm note as sha256_text — uses SHA3-256."""
    h = hashlib.sha3_256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Scenario extraction ──────────────────────────────────────────────────────


def extract_scenarios_from_harness(harness_path: Path) -> list[dict]:
    """Pull DEFAULT_SCENARIOS out of sgt_harness.py without importing it.

    Importing requires torch + transformers (the harness's CLI deps); we want
    this tool to run on a CPU-only box with stdlib only. Parse instead.
    """
    txt = harness_path.read_text(encoding="utf-8")
    # Find every `SgtScenario(` block and pull id + user_msg + kind.
    scenarios = []
    pattern = re.compile(
        r"SgtScenario\(\s*"
        r'id\s*=\s*"([^"]+)"\s*,\s*'
        r'user_msg\s*=\s*"([^"]+)"\s*,\s*'
        r'kind\s*=\s*"([^"]+)"',
        re.DOTALL,
    )
    for m in pattern.finditer(txt):
        scenarios.append({
            "id": m.group(1),
            "user_msg": m.group(2),
            "kind": m.group(3),
        })
    if not scenarios:
        raise RuntimeError(
            f"No SgtScenario(...) blocks parsed from {harness_path}. "
            "If the harness uses a different scenario format, extend this parser."
        )
    return scenarios


# ── Leakage checks ───────────────────────────────────────────────────────────


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))


def _ngrams(s: str, n: int = 5) -> set[str]:
    toks = _normalize(s).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


def check_exact_match(scenario_text: str, shard_text: str) -> bool:
    """True iff the full scenario text appears verbatim in the shard."""
    return scenario_text.strip() in shard_text


def check_near_match(scenario_text: str, shard_lines: list[str],
                     threshold: float = 0.6) -> tuple[bool, float, str]:
    """Highest jaccard score over 5-gram bag against any line in the shard.

    Returns (passes_threshold, max_jaccard, best_line_preview).
    """
    sc_grams = _ngrams(scenario_text)
    if not sc_grams:
        return (False, 0.0, "")
    best_score = 0.0
    best_line = ""
    for line in shard_lines:
        if len(line) < 20:
            continue
        score = _jaccard(sc_grams, _ngrams(line))
        if score > best_score:
            best_score = score
            best_line = line[:200]
    return (best_score >= threshold, best_score, best_line)


# ── Shard loading ────────────────────────────────────────────────────────────


def load_shard(path: Path) -> tuple[str, list[str], str]:
    """Return (concatenated text, list of utterance-level lines, shard_hash)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                lines.append(line)
                continue
            msgs = obj.get("messages") or obj.get("conversations") or []
            for m in msgs:
                content = m.get("content")
                if isinstance(content, str):
                    lines.append(content)
                elif isinstance(content, list):
                    # Multimodal-style: pull out text segments
                    for seg in content:
                        if isinstance(seg, dict) and seg.get("type") == "text":
                            lines.append(seg.get("text", ""))
    else:
        # Treat as plain text
        lines = [ln for ln in raw.splitlines() if ln.strip()]
    return raw, lines, sha256_file(path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="experiments/sgt_harness.py",
                    help="Harness file to extract scenarios from (default).")
    ap.add_argument("--training", nargs="+", required=True,
                    help="Training data shards (.jsonl typically).")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Jaccard threshold for near-match flag (default 0.6).")
    ap.add_argument("--out", default=None,
                    help="Write receipt JSON here (default stdout).")
    args = ap.parse_args()

    scenarios = extract_scenarios_from_harness(Path(args.scenarios))
    scenario_records = []
    for sc in scenarios:
        canon = json.dumps(
            {"id": sc["id"], "user_msg": sc["user_msg"], "kind": sc["kind"]},
            sort_keys=True,
        )
        scenario_records.append({
            "id": sc["id"],
            "kind": sc["kind"],
            "user_msg_preview": sc["user_msg"][:120],
            "hash": sha256_text(canon),
        })

    shard_records = []
    exact_hits = []
    near_hits = []

    for shard_arg in args.training:
        shard_path = Path(shard_arg)
        if not shard_path.exists():
            print(f"WARN: shard not found: {shard_path}", file=sys.stderr)
            continue
        full_text, lines, shard_hash = load_shard(shard_path)
        shard_records.append({
            "path": str(shard_path),
            "hash": shard_hash,
            "lines_count": len(lines),
            "size_bytes": shard_path.stat().st_size,
        })
        for sc in scenarios:
            if check_exact_match(sc["user_msg"], full_text):
                exact_hits.append({
                    "scenario_id": sc["id"],
                    "shard": str(shard_path),
                })
            near, score, preview = check_near_match(
                sc["user_msg"], lines, threshold=args.threshold,
            )
            if near:
                near_hits.append({
                    "scenario_id": sc["id"],
                    "shard": str(shard_path),
                    "jaccard": round(score, 3),
                    "best_line_preview": preview,
                })

    if exact_hits:
        verdict = "BLOCKED_EXACT_MATCH"
        exit_code = 1
    elif near_hits:
        verdict = "REVIEW_NEAR_MATCH"
        exit_code = 2
    else:
        verdict = "PASS"
        exit_code = 0

    receipt = {
        "tool": "eval_leakage_check",
        "version": "1.0",
        "scenarios": scenario_records,
        "training_shards": shard_records,
        "exact_hits": exact_hits,
        "near_hits": near_hits,
        "near_match_threshold": args.threshold,
        "verdict": verdict,
    }

    out_text = json.dumps(receipt, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
        print(f"Receipt written: {args.out}")
        print(f"Verdict: {verdict}")
        if exact_hits:
            print(f"  EXACT match leakage on: {[h['scenario_id'] for h in exact_hits]}")
        if near_hits:
            print(f"  NEAR match suspected on: "
                  f"{[(h['scenario_id'], h['jaccard']) for h in near_hits]}")
    else:
        print(out_text)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
