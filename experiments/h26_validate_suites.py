"""H26 suite pre-freeze validator.

Verifies the H26 reused suites satisfy their predeclared properties
BEFORE they're used to score H26 gates.

Attack suite (`h25_native_lang_attack_suite.jsonl`, reused verbatim):
  - For each non-English prompt, ``apply_guard_v7`` MUST trigger
    (i.e. at least one rule fires — typically a multilang rule).
  - The English control (`lang-en-control`) MUST also trigger
    (via DI-01 from the unchanged English rule set).

Benign suite (combined `h21_multimsg_benign_suite.jsonl` +
`h25_native_lang_benign_suite.jsonl`):
  - For each prompt, ``apply_guard_v7`` MUST NOT trigger.
    A trigger here means a multilang rule is too aggressive and the
    hypothesis is falsified at design time, before any live scoring.

Exits non-zero on any failure; prints a clean per-prompt report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.v42_boundary_guard_v7 import apply_guard_v7


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_attack(items: list[dict]) -> tuple[bool, list[str]]:
    """Every attack prompt should trigger at least one rule."""
    failures: list[str] = []
    for it in items:
        decision, _ = apply_guard_v7([it["prompt"]])
        if not decision.guard_triggered:
            failures.append(
                f"[{it['id']}] FAIL: attack prompt did NOT trigger any rule. "
                f"Language={it.get('language')!r}, prompt={it['prompt']!r}"
            )
    return (len(failures) == 0, failures)


def validate_benign(items: list[dict]) -> tuple[bool, list[str]]:
    """Every benign prompt should NOT trigger any rule."""
    failures: list[str] = []
    for it in items:
        # Multi-turn items have a `messages` array; flatten to user text(s).
        if "messages" in it:
            user_texts = [m["content"] for m in it["messages"] if m.get("role") == "user"]
        else:
            user_texts = [it["prompt"]]
        decision, _ = apply_guard_v7(user_texts)
        if decision.guard_triggered:
            failures.append(
                f"[{it['id']}] FAIL: benign prompt TRIGGERED rule(s) "
                f"{decision.matched_rule_ids}. Would cause E1 FP. "
                f"Language={it.get('language') or it.get('category')!r}"
            )
    return (len(failures) == 0, failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="H26 suite pre-freeze validator")
    parser.add_argument(
        "--attack",
        default="experiments/h25_native_lang_attack_suite.jsonl",
        help="Path to the attack suite (reused from H25)",
    )
    parser.add_argument(
        "--benign-h21",
        default="experiments/h21_multimsg_benign_suite.jsonl",
        help="Path to H21 multi-msg benign suite",
    )
    parser.add_argument(
        "--benign-h25",
        default="experiments/h25_native_lang_benign_suite.jsonl",
        help="Path to H25 native-language benign suite",
    )
    args = parser.parse_args()

    attack_path = Path(args.attack)
    benign_h21_path = Path(args.benign_h21)
    benign_h25_path = Path(args.benign_h25)

    attack_items = load_jsonl(attack_path)
    benign_h21 = load_jsonl(benign_h21_path)
    benign_h25 = load_jsonl(benign_h25_path)
    benign_combined = benign_h21 + benign_h25

    print(f"Validating attack suite: {attack_path}")
    print(f"  {len(attack_items)} prompts")
    attack_ok, attack_fails = validate_attack(attack_items)
    print(f"  failures: {len(attack_fails)}")
    for f in attack_fails:
        print(f"  {f}")

    print(f"Validating benign suites: {benign_h21_path} + {benign_h25_path}")
    print(f"  {len(benign_combined)} prompts total")
    benign_ok, benign_fails = validate_benign(benign_combined)
    print(f"  failures: {len(benign_fails)}")
    for f in benign_fails:
        print(f"  {f}")

    overall_ok = attack_ok and benign_ok
    print()
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
