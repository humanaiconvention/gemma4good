"""
evaluate_promotion.py — single entry point for the full HAIC evaluation pipeline.

Combines:
  1. Rigorous SGT harness (experiments/sgt_harness.py)
  2. Eval-set leakage check (tools/eval_leakage_check.py)
  3. Six-gate promotion decision (tools/check_promotion.py)

Designed to be the one call a kaggle notebook cell or a CI promotion job
makes. Returns a single decision JSON that contains all three receipts
plus the final go/no-go.

Usage as CLI:

    python -m tools.evaluate_promotion \\
        --model haic-gemma4-v39 \\
        --base D:/models/.../gemma-4-E2B-it/snapshots/... \\
        --adapter D:/.../haic-gemma4-v39-adapter \\
        --training data/v35_gov_final.jsonl data/v39_synthetic.jsonl \\
        --n-samples 20 \\
        --profile default \\
        --out experiments/v39_promotion.json

Usage from a notebook (after model is loaded):

    from tools.evaluate_promotion import evaluate_promotion
    decision = evaluate_promotion(
        model=model, tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        adapter_id="haic-gemma4-v39",
        training_shards=["data/v35_gov_final.jsonl", "data/v39_synthetic.jsonl"],
        scenario_set="default",  # or "extended"
        n_samples=20,
        baseline_backend=None,   # optional: another make_hf_backend on base model
        profile="default",
    )
    print(decision["overall"]["decision"])

In notebook usage, the leakage check is best-effort (skipped if shards
not provided); the rigorous SGT pass is always run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Module-level imports of pure-Python utilities (no torch yet)
from experiments.sgt_harness import (
    DEFAULT_SCENARIOS, DEFAULT_SECURITY_RUBRICS, run_sgt,
)
from tools.eval_leakage_check import (
    extract_scenarios_from_harness,
    sha256_text, sha256_file,
    check_exact_match, check_near_match,
    load_shard,
)
from tools.check_promotion import (
    PROFILES,
    gate_capability_gain, gate_leakage, gate_consistency,
    gate_covenant, gate_isolation, gate_epistemic,
    aggregate_decision,
)


# ── Scenario-set selection ───────────────────────────────────────────────────


def _resolve_scenarios(scenario_set: str, rubric_kind: str = "refined"):
    """Pick scenarios + security rubrics.

    rubric_kind:
      - "refined" (default): RefinedSecurityRubric — adds refusal_markers and
        negation-aware compliance matching. Doctrine-aligned per
        docs/security_rubric_finding.md. Recommended for all promotion runs.
      - "strict": Garrett's original SecurityRubric — used for upstream-PR
        comparability and as the conservative reference.

    The rubric_kind only affects security scenarios; grounding scenarios are
    unaffected.
    """
    if scenario_set == "default":
        scenarios = DEFAULT_SCENARIOS
        if rubric_kind == "strict":
            rubrics = DEFAULT_SECURITY_RUBRICS
        else:
            try:
                from experiments.sgt_extended_scenarios import (
                    REFINED_DEFAULT_SECURITY_RUBRICS,
                )
            except ImportError as e:
                raise RuntimeError(
                    f"Refined rubric not available: {e}. "
                    "Use rubric_kind='strict' or install the extended module."
                )
            rubrics = REFINED_DEFAULT_SECURITY_RUBRICS
        return scenarios, rubrics
    if scenario_set == "extended":
        try:
            from experiments.sgt_extended_scenarios import (
                ALL_SCENARIOS, ALL_RUBRICS,
            )
            if rubric_kind == "refined":
                from experiments.sgt_extended_scenarios import refined_rubric_from
                rubrics = {sid: refined_rubric_from(rub) for sid, rub in ALL_RUBRICS.items()}
            else:
                rubrics = ALL_RUBRICS
        except ImportError as e:
            raise RuntimeError(
                f"Extended scenarios not available: {e}. "
                "Use scenario_set='default' or install the extended module."
            )
        return ALL_SCENARIOS, rubrics
    raise ValueError(f"Unknown scenario_set={scenario_set!r}; "
                     "use 'default' or 'extended'.")


# ── In-process leakage check ─────────────────────────────────────────────────


def _run_leakage(scenarios, training_shards: list[Path],
                 threshold: float = 0.6) -> dict:
    """Same logic as tools/eval_leakage_check.py main() but in-process."""
    scenario_records = []
    for sc in scenarios:
        canon = json.dumps(
            {"id": sc.id, "user_msg": sc.user_msg, "kind": sc.kind},
            sort_keys=True,
        )
        scenario_records.append({
            "id": sc.id,
            "kind": sc.kind,
            "user_msg_preview": sc.user_msg[:120],
            "hash": sha256_text(canon),
        })

    shard_records, exact_hits, near_hits = [], [], []
    for shard_path in training_shards:
        if not shard_path.exists():
            continue
        full_text, lines, shard_hash = load_shard(shard_path)
        shard_records.append({
            "path": str(shard_path),
            "hash": shard_hash,
            "lines_count": len(lines),
            "size_bytes": shard_path.stat().st_size,
        })
        for sc in scenarios:
            if check_exact_match(sc.user_msg, full_text):
                exact_hits.append({"scenario_id": sc.id, "shard": str(shard_path)})
            near, score, preview = check_near_match(
                sc.user_msg, lines, threshold=threshold,
            )
            if near:
                near_hits.append({
                    "scenario_id": sc.id, "shard": str(shard_path),
                    "jaccard": round(score, 3),
                    "best_line_preview": preview,
                })

    if exact_hits:
        verdict = "BLOCKED_EXACT_MATCH"
    elif near_hits:
        verdict = "REVIEW_NEAR_MATCH"
    else:
        verdict = "PASS"

    return {
        "tool": "eval_leakage_check",
        "version": "1.0",
        "scenarios": scenario_records,
        "training_shards": shard_records,
        "exact_hits": exact_hits,
        "near_hits": near_hits,
        "near_match_threshold": threshold,
        "verdict": verdict,
    }


# ── Promotion decision (in-process, no path required) ────────────────────────


def _run_promotion_decision(report: dict, leakage_receipt: Optional[dict],
                             profile_name: str) -> dict:
    profile = PROFILES[profile_name]
    # Build a synthetic Path so gate_covenant has something to hash. We
    # write a temp-file-ish byte string and just hash the JSON.
    report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
    report_hash = __import__("hashlib").sha3_256(report_bytes).hexdigest()

    # Wrap report path: gate_covenant calls _file_sha256 on the path. We
    # provide an in-memory equivalent by patching with a closure. Easier:
    # write to a temp file. Use stdlib tempfile.
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".json", encoding="utf-8",
    ) as f:
        f.write(json.dumps(report))
        tmp_path = Path(f.name)
    try:
        verdicts = [
            gate_capability_gain(report, profile),
            gate_leakage(leakage_receipt),
            gate_consistency(report, profile),
            gate_covenant(report, tmp_path),
            gate_isolation(report),
            gate_epistemic(report, profile),
        ]
    finally:
        os.unlink(tmp_path)

    decision, exit_code, decision_rationale = aggregate_decision(verdicts)
    return {
        "tool": "check_promotion",
        "version": "1.0",
        "profile": profile_name,
        "profile_thresholds": profile,
        "report_sha256": report_hash,
        "gate_verdicts": [asdict(v) for v in verdicts],
        "decision": decision,
        "exit_code": exit_code,
        "decision_rationale": decision_rationale,
    }


# ── Public entry ─────────────────────────────────────────────────────────────


def evaluate_promotion(
    *,
    backend,                                  # generate_fn from make_hf_backend
    adapter_id: str,                          # human-readable model id for the report
    training_shards: list[Path] | None = None,
    scenario_set: str = "default",
    rubric_kind: str = "refined",             # "refined" (default) or "strict"
    n_samples: int = 10,
    seed: int = 42,
    decoding_record: dict | None = None,
    baseline_backend=None,                    # optional generate_fn for base model
    baseline_id: str = "google/gemma-4-E2B-it",
    profile: str = "default",
    leakage_threshold: float = 0.6,
) -> dict:
    """
    Run the full promotion-gate pipeline against an HF generate_fn backend.

    Returns a single dict with sub-receipts for sgt, leakage, and decision,
    plus a top-level "overall" with the final go/no-go.

    rubric_kind:
      - "refined" (default): RefinedSecurityRubric — credits clean refusals
        as security-correct, with negation-aware compliance matching.
        Doctrine-aligned per docs/security_rubric_finding.md. The number this
        produces is the load-bearing one for promotion decisions.
      - "strict": Garrett's original rubric (security PASS requires both
        stayed_in_protocol AND not complied). Useful for upstream-PR
        comparability and as the conservative reference.
    """
    scenarios, rubrics = _resolve_scenarios(scenario_set, rubric_kind)

    # --- 1. Rigorous SGT (finetune) ------------------------------------------
    t0 = time.time()
    finetune_result = run_sgt(
        backend, scenarios=scenarios, rubrics=rubrics,
        n_samples=n_samples, seed=seed,
        model_id=adapter_id, decoding=decoding_record,
    )
    sgt_finetune_seconds = time.time() - t0

    sgt_report = {"finetune": finetune_result}

    # --- 1b. Rigorous SGT (baseline, optional) ------------------------------
    if baseline_backend is not None:
        t0 = time.time()
        baseline_result = run_sgt(
            baseline_backend, scenarios=scenarios, rubrics=rubrics,
            n_samples=n_samples, seed=seed,
            model_id=baseline_id, decoding=decoding_record,
        )
        sgt_report["baseline"] = baseline_result
        sgt_report["sgt_baseline_seconds"] = time.time() - t0

    sgt_report["sgt_finetune_seconds"] = sgt_finetune_seconds

    # --- 2. Leakage receipt --------------------------------------------------
    leakage_receipt = None
    if training_shards:
        leakage_receipt = _run_leakage(
            scenarios, [Path(s) for s in training_shards],
            threshold=leakage_threshold,
        )

    # --- 3. Promotion decision ----------------------------------------------
    decision_receipt = _run_promotion_decision(sgt_report, leakage_receipt, profile)

    # --- Overall envelope ---------------------------------------------------
    return {
        "evaluate_promotion_version": "1.0",
        "scenario_set": scenario_set,
        "n_samples": n_samples,
        "seed": seed,
        "sgt": sgt_report,
        "leakage": leakage_receipt,
        "decision": decision_receipt,
        "overall": {
            "decision": decision_receipt["decision"],
            "rationale": decision_receipt["decision_rationale"],
            "profile": profile,
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Adapter / model id for the report.")
    ap.add_argument("--base", required=True, help="Path or HF id for the base model.")
    ap.add_argument("--adapter", default=None,
                    help="Optional adapter path. If omitted, runs base only.")
    ap.add_argument("--training", nargs="*", default=[],
                    help="Training data shards for leakage check. Optional.")
    ap.add_argument("--scenario-set", default="default", choices=("default", "extended"))
    ap.add_argument("--rubric", default="refined", choices=("refined", "strict"),
                    help="Security rubric. 'refined' (default) credits clean refusals "
                         "and uses negation-aware compliance matching. 'strict' is "
                         "Garrett's original (concealed-compliance only).")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--four-bit", action="store_true", default=True)
    ap.add_argument("--no-four-bit", dest="four_bit", action="store_false")
    ap.add_argument("--baseline", action="store_true",
                    help="Also evaluate the base model and report Δ.")
    ap.add_argument("--profile", default="default", choices=list(PROFILES))
    ap.add_argument("--system-prompt-file", default=None,
                    help="Path to system prompt. Defaults to V38_SYSTEM_PROMPT.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Late imports — torch costs a few seconds.
    from experiments.run_v38_sgt import load_model, V38_SYSTEM_PROMPT
    from experiments.sgt_harness import make_hf_backend
    import torch

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text()
    else:
        system_prompt = V38_SYSTEM_PROMPT

    decoding_record = dict(
        temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        precision="4bit_nf4" if args.four_bit else "fp16",
    )

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.base, args.adapter, args.four_bit)
    backend = make_hf_backend(
        model, tokenizer, system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    baseline_backend = None
    if args.baseline:
        # Run finetune first, then swap to base. Avoids holding both in VRAM.
        # We'll run twice via two evaluate_promotion calls — first WITH adapter,
        # then a second pass with base only — and merge the SGT receipts.
        # Simpler: pass a deferred loader closure. But for now do two passes.
        pass  # handled below

    decision = evaluate_promotion(
        backend=backend,
        adapter_id=args.model,
        training_shards=args.training,
        scenario_set=args.scenario_set,
        rubric_kind=args.rubric,
        n_samples=args.n_samples,
        seed=args.seed,
        decoding_record=decoding_record,
        baseline_backend=None,
        profile=args.profile,
    )

    # If baseline requested, run a second pass on base alone, merge.
    if args.baseline:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Loading base for baseline pass ...")
        base_model, base_tok = load_model(args.base, None, args.four_bit)
        base_backend = make_hf_backend(
            base_model, base_tok, system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )
        from experiments.sgt_harness import run_sgt
        scenarios, rubrics = _resolve_scenarios(args.scenario_set, args.rubric)
        base_sgt = run_sgt(
            base_backend, scenarios=scenarios, rubrics=rubrics,
            n_samples=args.n_samples, seed=args.seed,
            model_id="google/gemma-4-E2B-it", decoding=decoding_record,
        )
        decision["sgt"]["baseline"] = base_sgt
        # Re-run the decision now that we have baseline data
        decision["decision"] = _run_promotion_decision(
            decision["sgt"], decision.get("leakage"), args.profile,
        )
        decision["overall"] = {
            "decision": decision["decision"]["decision"],
            "rationale": decision["decision"]["decision_rationale"],
            "profile": args.profile,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(decision, indent=2))
    print()
    print("=" * 60)
    print(f"OVERALL: {decision['overall']['decision']}")
    print(f"  {decision['overall']['rationale']}")
    print("=" * 60)
    print(f"Full report written to {args.out}")
    sys.exit(decision["decision"]["exit_code"])


if __name__ == "__main__":
    main()
