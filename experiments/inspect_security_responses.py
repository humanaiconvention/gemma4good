"""inspect_security_responses.py — pull what's actually breaking the security rubric.

Usage:
    python -m experiments.inspect_security_responses [path/to/v38_sgt_rigorous.json]

Prints, per security scenario, all sampling-pass responses with the rubric
breakdown (compliance markers matched, in-protocol markers matched, the
overall PASS/FAIL). Useful for deciding whether 0/20 is a real model defect
or an over-aggressive rubric.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from experiments.sgt_harness import (
    DEFAULT_SECURITY_RUBRICS, _has_pivot_marker,
)


def _which_compliance_markers_hit(rubric, response: str) -> list[str]:
    r = response.lower()
    return [m for m in rubric.compliance_markers if m in r]


def _which_protocol_markers_hit(rubric, response: str) -> list[str]:
    r = response.lower()
    return [m for m in rubric.in_protocol_markers if m in r]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/v38_sgt_rigorous.json")
    if not path.exists():
        print(f"No report at {path} yet.")
        sys.exit(1)

    payload = json.loads(path.read_text())
    finetune = payload["finetune"]
    samp = finetune["sampling"]

    sec_records = [r for r in samp["per_scenario"] if r.get("kind") == "security"]
    print(f"Inspecting {len(sec_records)} security records from sampling pass.\n")

    by_scenario = {}
    for r in sec_records:
        by_scenario.setdefault(r["scenario_id"], []).append(r)

    for sid, records in by_scenario.items():
        rubric = DEFAULT_SECURITY_RUBRICS.get(sid)
        print(f"=== {sid}  (n={len(records)}) ===")
        if rubric is None:
            print("  No rubric registered.")
            continue
        for i, rec in enumerate(records):
            resp = rec.get("response_preview", "")
            comp = _which_compliance_markers_hit(rubric, resp)
            proto = _which_protocol_markers_hit(rubric, resp)
            has_pivot = _has_pivot_marker(resp)
            verdict = rec["result"]
            print(f"  [{i:>2}] result={verdict}  pivot={has_pivot}  "
                  f"complied={bool(comp)}  in_protocol={bool(proto)}")
            if comp: print(f"        compliance hits: {comp}")
            if proto: print(f"        protocol  hits: {proto[:3]}{'...' if len(proto)>3 else ''}")
            print(f"        resp: {resp[:220]!r}")
        print()

    # Aggregate failure-mode summary
    n_complied = sum(
        1 for r in sec_records
        if r.get("complied") is True
    )
    n_no_protocol = sum(
        1 for r in sec_records
        if r.get("stayed_in_protocol") is False
    )
    n_both = sum(
        1 for r in sec_records
        if r.get("complied") is True and r.get("stayed_in_protocol") is False
    )
    print("Failure-mode summary:")
    print(f"  complied=True:         {n_complied}/{len(sec_records)}")
    print(f"  stayed_in_protocol=F:  {n_no_protocol}/{len(sec_records)}")
    print(f"  both fail conditions:  {n_both}/{len(sec_records)}")


if __name__ == "__main__":
    main()
