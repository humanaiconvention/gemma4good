# Repo Status

Date: 2026-05-18 (post-submission aftercare pass)

This file summarizes the state of the public Gemma4Good repository after the
Kaggle Gemma 4 Good submission.

## Repository Role

This repository is the curated public-facing source layer for Gemma4Good.

It is meant to contain:

- the Kaggle-facing notebooks
- the governance and viability source code
- experiment and training utilities that are useful as source-of-truth
- explanatory docs and writeups

It is not meant to contain:

- local logs
- secrets
- heavyweight deployment artifacts
- one-off scratch patch helpers
- generated zip bundles that can be rebuilt from source

## Submission Boundary

The Kaggle submission was filed on 2026-05-18 from `main` at commit
`ec7db2e final asset: 560x280 Kaggle card preview`.

The frozen submitted file set is indexed in
`docs/submission_manifest_2026-05-18.md`. Later commits are aftercare unless
explicitly labeled as a new submission or release.

## Release Scope

The public source release is intended to be readable, reproducible, and safe to
share with collaborators without bundling private machine state or heavyweight
runtime artifacts.

The current public posture is governance-first:

- **`guard-v7 + v42` is the submitted promoted candidate** (H26 PASS,
  2026-05-17). Guard-v7 serves as the runtime boundary layer; v42 weights are
  unchanged.
  Anchor: `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8`.
  Verdict: `docs/h26_verdict_2026-05-17.md`.
- v42 GGUF remains the base model. No weights were changed across the H-series.
- v58 and v59 are documented experimental appendix results, not promoted
  replacements.
- New model or guard claims require real artifacts, seeded canonical eval
  output, self-anchors, and predeclared non-compensatory gates.
- The limitations ledger is `docs/v42_guard_known_limitations_2026-05-15.md`.
  As of H26, L-01 through L-09 are closed, routed to a non-guard layer, or
  intentionally outside scope.
- H19 and H25 remain published FAIL verdicts. They are part of the evidence
  chain, not embarrassments to erase.

## Phase-2 Review (2026-05-16)

A second comprehensive code-review pass landed 46 findings; 30+ were
addressed in-tree. Highlights:

- Guard infra hardened (shared httpx client lifespan, 502 on upstream
  errors, SHA3 surrogate safety, catch-all proxy restricted to GET/HEAD
  allowlist). Matching behavior unchanged → H18r4 anchor preserved.
- pyproject.toml populated with declared runtime + ML extras.
- 31 stale experiment scripts moved to `experiments/archive/` with
  README; canonical rubric API extracted to `experiments/rubrics.py`.
- `maestro_gateway` defaults to production mode; chat endpoint size-
  capped.
- Doc rule-count corrected (was wrongly stated as 17, actual is 16).
- New `tests/test_v42_boundary_guard_proxy.py` covers proxy + streaming
  + upstream-error paths and asserts the exact 16-rule set.
- Test count: 668 → 679 passing.
- `tools/v42_boundary_guard_v2.py` implements the H19 candidate with
  Unicode normalization + multi-message scan (separate module, not
  modifying the H18r4-promoted v1 guard).

## Final H-Series Extension (2026-05-17)

After the phase-2 review, H20 through H26 extended the deterministic guard
with separate predeclared hypotheses:

- H20 closed Unicode bypass.
- H21 closed multi-message scan.
- H22 closed client-supplied system-role injection.
- H23 passed the encoded-payload behavioral defense and surfaced L-08.
- H24 closed L-08 with leet-fold matching.
- H25 failed and confirmed native-language attack bypass as L-09.
- H26 closed L-09 with eleven multi-language direct-injection rule families.

The final submitted guard is `tools/v42_boundary_guard_v7.py`; tests reached
797 passing by the final submission state.

## Push-Readiness Notes

At the time of this pass, the repository contains:

- core source modules
- tests
- notebook assets and helper scripts
- writeup and deployment docs
- curated experiment utilities

## Collaboration Note

The repository is organized as a curated source tree rather than an artifact
dump. If new material is added later, the same standard should apply:

- keep source, notebooks, maintained utilities, and explanatory docs
- exclude secrets, logs, caches, and heavyweight generated artifacts
