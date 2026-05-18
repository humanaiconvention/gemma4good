# Security Policy

## Scope

This repository contains the source for `gemma4good`, a Kaggle hackathon entry
demonstrating verifiable AI governance. The security-relevant components are:

- **`tools/v42_boundary_guard.py`** — the H18r4-promoted deterministic regex
  proxy that mediates between callers and the v42 language model. This is the
  central security boundary in the runtime path.
- **`tools/v42_boundary_guard_v2.py`** — H19 candidate with Unicode
  normalization and multi-message scanning (separate module; not yet promoted).
- **`maestro_gateway/app.py`** — reference FastAPI gateway. Defaults to
  fail-closed (`MAESTRO_LAUNCH_MODE=production`); tests opt in to test mode
  via env var.
- **`onchain/HAICAnchor.sol`** — Solidity contract for anchoring Merkle
  receipts to a public chain. Audited for the limited surface it exposes
  (`anchorReceipt(bytes32 root)` only).
- **`utils/merkle.py`** — SHA3-256 + Merkle root utilities used everywhere.

## Acknowledged limitations

The H18r4-promoted guard's known gaps are catalogued in
`docs/v42_guard_known_limitations_2026-05-15.md`. These are documented
publicly because:

1. The H18r4 canonical anchor only covers ASCII single-message attacks.
2. Pretending otherwise would violate the submission's discipline of
   only claiming what is anchored.
3. The H19 hypothesis (`docs/h19_precommit_hypothesis_2026-05-16.md`) is
   the predeclared path to closing those gaps.

In short: **the guard is a defense-in-depth layer over a base model that
has known weaknesses on adversarial inputs.** Do not deploy `guard + v42`
as a sole security boundary for untrusted public input.

## Reporting

Security issues that go beyond the documented limitations should be
reported via GitHub issues on [github.com/humanaiconvention/gemma4good](https://github.com/humanaiconvention/gemma4good)
with the `security` label, or by emailing the maintainers listed in
`README.md`.

For the runtime governance loop's broader threat model — the consent
sub-system, leakage receipts, federation viability gates — see
`docs/evaluation_doctrine.md` and `docs/runtime_grounding_loop_2026-05-11.md`.

## What we DO commit to

- **Predeclared, non-compensatory gates.** Every promoted security claim
  is anchored to a SHA3-256 self-hash over a canonical eval run with
  predeclared thresholds. Gates are not relaxed after seeing results.
- **Audit-stable hashing.** SHA3-256 (not SHA-256) for all Merkle work,
  to reduce length-extension surface and align with NIST's recent default.
- **No raw user text in logs.** The guard hashes user text before
  logging matched rule IDs; raw text never reaches the on-disk audit log.
- **Fail-closed defaults.** Reference gateway defaults to
  `MAESTRO_LAUNCH_MODE=production`. The dev-token shortcut requires
  explicit opt-in.

## What we do NOT commit to

- The guard is **not** a comprehensive WAF. It targets the 16 attack
  classes documented in `docs/v42_boundary_guard_precommit_2026-05-14.md`.
- The v42 model is **not** independently safety-aligned. Removing the
  guard exposes the base model's known concealed-compliance and
  jailbreak failure modes.
- The viability condition is **violated** for Gemma 4 E2B
  (`Ceff/E = 0.879`). This is a documented architectural finding, not a
  fixable bug. The framework's correct behavior is to flag this, not
  mask it.
