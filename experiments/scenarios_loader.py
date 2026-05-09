"""
scenarios_loader.py — load SGT scenarios from versioned JSONL.

The canonical source of truth for the eval scenarios is
``experiments/sgt_scenarios.jsonl``. The harness module
(``experiments/sgt_harness.py``, which is Garrett's commit) has its own
embedded ``DEFAULT_SCENARIOS`` for backwards compatibility; this loader
prefers the JSONL when present.

Why decoupled:
  - The eval set is a HAIC-style versioned artifact. It should have a
    stable hash that's independent of harness code edits.
  - v40 (and later) can extend scenarios without touching Garrett's
    upstream-PR-able harness file.
  - The leakage receipt cites a scenario-set hash; that hash should be
    over the canonical JSONL, not over a regex-parse of a Python file.

Usage:

    from experiments.scenarios_loader import (
        load_scenarios_jsonl, scenarios_hash,
    )

    scenarios = load_scenarios_jsonl()  # default path
    h = scenarios_hash(scenarios)       # SHA3-256 over canonical form
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from experiments.sgt_harness import SgtScenario


_DEFAULT_PATH = Path(__file__).resolve().parent / "sgt_scenarios.jsonl"


def load_scenarios_jsonl(path: Path | str = _DEFAULT_PATH) -> list[SgtScenario]:
    """Load scenarios from JSONL into SgtScenario instances.

    The JSONL format mirrors the dataclass fields plus an optional
    ``description`` and a required ``version`` for evolution tracking.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenarios file not found: {p}")

    out: list[SgtScenario] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(SgtScenario(
            id=d["id"],
            user_msg=d["user_msg"],
            kind=d["kind"],
            expected_pivot=d.get("expected_pivot", d["kind"] == "grounding"),
            description=d.get("description", ""),
        ))
    return out


def scenarios_hash(scenarios: list[SgtScenario] | list[dict]) -> str:
    """SHA3-256 over the canonical sorted JSON form of the scenario set.

    Matches the HAIC receipt format (utils/merkle uses SHA3-256 throughout).
    Stable across the order scenarios appear in the JSONL.
    """
    canon = []
    for sc in scenarios:
        if isinstance(sc, dict):
            canon.append({
                "id": sc["id"],
                "user_msg": sc["user_msg"],
                "kind": sc["kind"],
            })
        else:
            canon.append({
                "id": sc.id,
                "user_msg": sc.user_msg,
                "kind": sc.kind,
            })
    canon.sort(key=lambda x: x["id"])
    body = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha3_256(body.encode("utf-8")).hexdigest()


def hash_scenarios_file(path: Path | str = _DEFAULT_PATH) -> str:
    """Convenience: load + hash. The receipt-friendly one-liner."""
    return scenarios_hash(load_scenarios_jsonl(path))


def equivalent_to_harness_default(scenarios: list[SgtScenario]) -> bool:
    """Sanity check: confirm the JSONL load matches Garrett's
    DEFAULT_SCENARIOS by id+user_msg+kind. Used by tests to ensure the
    JSONL is a faithful copy and didn't drift accidentally.
    """
    from experiments.sgt_harness import DEFAULT_SCENARIOS
    a = sorted([(s.id, s.user_msg, s.kind) for s in scenarios])
    b = sorted([(s.id, s.user_msg, s.kind) for s in DEFAULT_SCENARIOS])
    return a == b


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_PATH
    scs = load_scenarios_jsonl(p)
    print(f"Loaded {len(scs)} scenarios from {p}")
    print(f"  hash (SHA3-256, canonical form): {scenarios_hash(scs)}")
    print(f"  matches harness DEFAULT_SCENARIOS: {equivalent_to_harness_default(scs)}")
    for s in scs:
        print(f"    - {s.id:<28} {s.kind:9}  {s.user_msg[:60]}...")
