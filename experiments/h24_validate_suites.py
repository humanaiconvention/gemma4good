"""H24 suite pre-freeze validator.

Verifies the two H24 frozen suites satisfy their predeclared properties
before they're used to score H24 gates.

Attack suite (`h24_leet_attack_suite.jsonl`):
  - For each prompt, the leet-folded form MUST match a v3 (H18r4) rule.
  - For each prompt, the original normalized form MUST NOT match any rule.
    (The substitution must actually obscure the attack — otherwise the
    suite is testing rule coverage, not leet-fold mitigation.)

Benign-digits suite (`h24_benign_digits_suite.jsonl`):
  - For each prompt, the original normalized form MUST NOT match a rule.
  - For each prompt, the leet-folded form MUST NOT match a rule.
    (Otherwise the benign prompt is accidentally an attack-pattern.)

Exits non-zero on any failure; prints a clean per-prompt report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.v42_boundary_guard import RULES
from tools.v42_boundary_guard_v2 import _normalize_for_match
from tools.v42_boundary_guard_v6 import _leet_fold


def first_rule_match(text: str) -> str | None:
    for rule in RULES:
        if rule.pattern.search(text):
            return rule.rule_id
    return None


def validate_attack_suite(path: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        item = json.loads(line)
        prompt = item["prompt"]
        norm = _normalize_for_match(prompt)
        folded = _leet_fold(norm)
        orig_match = first_rule_match(norm)
        fold_match = first_rule_match(folded)
        if orig_match is not None:
            failures.append(
                f"[{item['id']}] FAIL: original normalized form matches rule "
                f"{orig_match} — leet-substitution did not obscure the attack."
            )
        if fold_match is None:
            failures.append(
                f"[{item['id']}] FAIL: leet-folded form does NOT match any rule. "
                f"Decoded='{folded}'"
            )
    return (len(failures) == 0, failures), n


def validate_benign_suite(path: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        item = json.loads(line)
        prompt = item["prompt"]
        norm = _normalize_for_match(prompt)
        folded = _leet_fold(norm)
        orig_match = first_rule_match(norm)
        fold_match = first_rule_match(folded)
        if orig_match is not None:
            failures.append(
                f"[{item['id']}] FAIL: benign prompt matches rule {orig_match} "
                f"on the original normalized form."
            )
        if fold_match is not None:
            failures.append(
                f"[{item['id']}] FAIL: benign prompt matches rule {fold_match} "
                f"AFTER leet-fold. Decoded='{folded}' — would cause E1 FP."
            )
    return (len(failures) == 0, failures), n


def main() -> int:
    parser = argparse.ArgumentParser(description="H24 suite pre-freeze validator")
    parser.add_argument(
        "--attack",
        default="experiments/h24_leet_attack_suite.jsonl",
        help="Path to the leet-attack suite",
    )
    parser.add_argument(
        "--benign",
        default="experiments/h24_benign_digits_suite.jsonl",
        help="Path to the benign-digits suite",
    )
    args = parser.parse_args()

    attack_path = Path(args.attack)
    benign_path = Path(args.benign)

    print(f"Validating attack suite: {attack_path}")
    (attack_ok, attack_fails), n_attack = validate_attack_suite(attack_path)
    print(f"  {n_attack} prompts checked; {len(attack_fails)} failures")
    for f in attack_fails:
        print(f"  {f}")

    print(f"Validating benign suite: {benign_path}")
    (benign_ok, benign_fails), n_benign = validate_benign_suite(benign_path)
    print(f"  {n_benign} prompts checked; {len(benign_fails)} failures")
    for f in benign_fails:
        print(f"  {f}")

    overall_ok = attack_ok and benign_ok
    print()
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
