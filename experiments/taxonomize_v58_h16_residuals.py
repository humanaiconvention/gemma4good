#!/usr/bin/env python3
"""Extract the residual H16 failure taxonomy for v58.

This is intentionally narrow: it does not re-grade the run or change the
precommitted H16 verdict. It explains the two failed gates from the frozen
canonical artifact:

- direct adversarial-injection misses;
- prompt/protocol disclosure-preview marker hits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.check_h16_go_live import DISCLOSURE_MARKERS


VERSIONS = ("v42", "v55", "v56", "v57", "v58")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_records(report: dict):
    for seed in report.get("per_seed", []):
        for scenario_id, scenario in seed.get("per_scenario", {}).items():
            for index, record in enumerate(scenario.get("records", [])):
                yield seed["seed"], scenario_id, index, record


def _bucket_adversarial_failure(record: dict) -> str:
    rubric = record["rubric_v1"]
    strict_class = record["rubric_strict"]["class"]
    if rubric.get("complied") or strict_class == "SEMANTIC_LEAK":
        return "compliance_marker_or_semantic_leak"
    if rubric.get("refused") or strict_class == "EXPLICIT_REFUSAL":
        return "refusal_not_credited_by_v1"
    return "abstract_deflection_or_soft_pivot"


def _disclosure_hits(preview: str) -> list[str]:
    lower = preview.lower()
    return [marker for marker in DISCLOSURE_MARKERS if marker in lower]


def _version_summary(report: dict) -> dict:
    total_records = 0
    disclosure_flags = 0
    disclosure_by_scenario: dict[str, int] = {}
    adv_failures = 0
    adv_passes = 0
    adv_total = 0
    adv_buckets: dict[str, int] = {}

    for _seed, scenario_id, _index, record in _all_records(report):
        total_records += 1
        hits = _disclosure_hits(record.get("response_preview", ""))
        if hits:
            disclosure_flags += 1
            disclosure_by_scenario[scenario_id] = disclosure_by_scenario.get(scenario_id, 0) + 1

        if scenario_id == "sgt_adversarial_inject":
            adv_total += 1
            if record["rubric_v1"].get("result") == "PASS":
                adv_passes += 1
            else:
                adv_failures += 1
                bucket = _bucket_adversarial_failure(record)
                adv_buckets[bucket] = adv_buckets.get(bucket, 0) + 1

    return {
        "aggregate_security": report["aggregate"]["rubric_v1"]["aggregate_security"]["pooled_rate"],
        "strict_concealed_refusal": report["aggregate"]["rubric_strict"]["pooled"]["explicit_refusal_rate_nonempty"],
        "strict_concealed_leak": report["aggregate"]["rubric_strict"]["pooled"]["semantic_leak_rate"],
        "adversarial_inject_rate": adv_passes / adv_total if adv_total else 0.0,
        "adversarial_inject_passes": adv_passes,
        "adversarial_inject_total": adv_total,
        "adversarial_inject_failures": adv_failures,
        "adversarial_failure_buckets": adv_buckets,
        "disclosure_preview_flags": disclosure_flags,
        "disclosure_preview_total": total_records,
        "disclosure_preview_rate": disclosure_flags / total_records if total_records else 0.0,
        "disclosure_flags_by_scenario": disclosure_by_scenario,
    }


def build_taxonomy(report_path: Path, baseline_paths: list[Path]) -> dict:
    report = _load(report_path)

    direct_failures = []
    disclosure_flags = []
    disclosure_marker_counts: dict[str, int] = {}

    for seed, scenario_id, index, record in _all_records(report):
        preview = record.get("response_preview", "")
        if scenario_id == "sgt_adversarial_inject" and record["rubric_v1"].get("result") != "PASS":
            direct_failures.append({
                "seed": seed,
                "index": index,
                "sub_seed": record.get("sub_seed"),
                "bucket": _bucket_adversarial_failure(record),
                "rubric_v1": record["rubric_v1"],
                "rubric_strict": record["rubric_strict"],
                "response_preview": preview,
            })

        hits = _disclosure_hits(preview)
        if hits:
            for hit in hits:
                disclosure_marker_counts[hit] = disclosure_marker_counts.get(hit, 0) + 1
            disclosure_flags.append({
                "seed": seed,
                "scenario_id": scenario_id,
                "index": index,
                "sub_seed": record.get("sub_seed"),
                "markers": hits,
                "rubric_v1": record["rubric_v1"],
                "rubric_strict": record["rubric_strict"],
                "response_preview": preview,
            })

    baseline_comparison = {}
    for path in baseline_paths:
        if path.exists():
            baseline_comparison[path.stem.replace("_canonical_old_prompt", "")] = _version_summary(_load(path))

    return {
        "kind": "v58_h16_residual_taxonomy",
        "source_report": str(report_path),
        "model_id": report.get("model_id"),
        "self_anchor": report.get("self_anchor"),
        "h16_verdict": "FAIL",
        "h16_failed_gates": [
            "sgt_adversarial_inject",
            "prompt_protocol_disclosure_preview_rate",
        ],
        "direct_adversarial_failures": direct_failures,
        "disclosure_preview_flags": disclosure_flags,
        "disclosure_marker_counts": dict(sorted(disclosure_marker_counts.items())),
        "baseline_comparison": baseline_comparison,
        "notes": [
            "This taxonomy explains failures; it does not alter the precommitted H16 verdict.",
            "response_preview is capped by canonical_eval.py; do not infer full text beyond the stored preview.",
            "rubric_strict was designed for concealed compliance, so strict labels on other scenarios are diagnostic only.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("experiments/v58_canonical_old_prompt.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/v58_h16_residual_taxonomy.json"))
    args = parser.parse_args()

    baseline_paths = [Path(f"experiments/{version}_canonical_old_prompt.json") for version in VERSIONS]
    taxonomy = build_taxonomy(args.report, baseline_paths)
    args.out.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"anchor={taxonomy['self_anchor']}")
    print(f"direct adversarial failures={len(taxonomy['direct_adversarial_failures'])}")
    print(f"disclosure preview flags={len(taxonomy['disclosure_preview_flags'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
