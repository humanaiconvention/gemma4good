# Repo Status

Date: 2026-05-16 (overnight phase-2 review pass)

This file summarizes the state of the public `0.1` Gemma4Good repository.

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

## Release Scope

Version `0.1` is the public source release. It is intended to be readable,
reproducible, and safe to share with collaborators without bundling private
machine state or heavyweight runtime artifacts.

The current public posture is governance-first:

- **`guard + v42` is the live promoted candidate** (H18r4 PASS, 2026-05-15).
  Guard serves on port 8082; v42 weights unchanged on port 8081.
  Anchor: `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`.
  Verdict: `docs/v42_guard_h18r4_verdict_2026-05-15.md`.
- v42 GGUF remains the base model. No weights were changed.
- v58 and v59 are documented experimental appendix results, not promoted
  replacements.
- New model claims require real artifacts, seeded canonical eval output,
  self-anchors, and predeclared non-compensatory gates.
- **Known H18r4 limitations** are documented in
  `docs/v42_guard_known_limitations_2026-05-15.md` — Unicode bypass and
  multi-message scan gaps that the H18 canonical attack set does not
  exercise. The H19 hypothesis to close those gaps is predeclared in
  `docs/h19_precommit_hypothesis_2026-05-16.md`.

## Phase-2 review (2026-05-16)

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
