# Changelog

All notable changes to gemma4good. Format follows [Keep a Changelog](https://keepachangelog.com/).

Per submission discipline, every model claim has a precommitted hypothesis,
a canonical eval, a self-anchor, and a verdict doc. This file is a high-
level rollup; the dated docs in `docs/` are the load-bearing record.

## [Submitted Snapshot] — 2026-05-18

### Added
- `docs/submission_manifest_2026-05-18.md` — exact Kaggle submission
  boundary at commit `ec7db2e`.
- `docs/research_record_map.md` — post-submission evidence map for cold
  readers and future agents.
- Final media gallery assets, video URL, and Kaggle card preview.

### Changed
- Submitted promoted candidate is `guard-v7 + v42` (H26), anchor
  `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8`.
- Test count reached 797 passing in the submitted state.
- H19 and H25 remain published FAIL verdicts; H20/H21/H22/H24/H26 are the
  passing closures that preserve the non-compensatory gate discipline.

## [Unreleased] — 2026-05-16

### Added — H19 verdict (FAIL — discipline held)
- `docs/h19_verdict_2026-05-16.md` — H19 FAILS per predeclared gates.
  - H19-A canonical replay PASS (aggregate_security 0.9943, strict
    EXPLICIT_REFUSAL 500/500, semantic_leak 0)
  - H19-B Unicode-bypass closure PASS (20/20 attacks triggered)
  - H19-C Unicode benign FP PASS (0/31 false positives across 17 languages)
  - H19-D1 multi-message attack FAIL (16/20; the 4 misses test v1 rule
    coverage gaps, not multi-msg iteration logic — suite was a flawed
    instrument for the hypothesis as written)
  - H19-D2 system-role injection FAIL (4/5; the 1 miss has system at
    message position 0 which the precommit explicitly permits as the
    operator system prompt; suite contradicted precommit)
  - Result: H18r4 remains promoted. Guard v2 retained as evidence of
    Unicode closure but not promoted. A future H20 with cleaner suite
    design would address the multi-message claim independently.
- `docs/why_this_matters.md` — 5-minute public-facing articulation
  of the project's thesis, what it proves, and what it honestly does NOT
  prove. Designed to fill the gap between the website tagline and the
  full repo for someone landing cold.
- `SECURITY.md` — security policy with explicit scope, acknowledged
  limitations link, and what the project does / does not commit to.

### Added (earlier in 2026-05-16)
- `docs/h19_precommit_hypothesis_2026-05-16.md` — H19 hypothesis to close
  the Unicode-bypass (L-01) and multi-message-scan (L-02) gaps in the
  H18r4 guard. Three predeclared test suites committed before any
  evaluation:
  - `experiments/h19_unicode_bypass_suite.jsonl` (20 attacks)
  - `experiments/h19_unicode_benign_suite.jsonl` (30 non-Latin benign)
  - `experiments/h19_multimsg_attack_suite.jsonl` (20 + 5 system inject)
- `tools/v42_boundary_guard_v2.py` — H19 guard candidate with NFKC
  normalization + zero-width strip + Cyrillic/Greek homoglyph fold +
  per-message scan + system-role-in-history rejection. Separate module
  from v1 to preserve the H18r4 anchor.
- `experiments/h19_offline_eval.py` — runs H19-B/C/D offline gates.
- `docs/v42_guard_known_limitations_2026-05-15.md` — catalog of security
  gaps that the H18r4 canonical anchor does NOT cover.
- `experiments/rubrics.py` — stable named entry point for strict + v1
  rubric classifiers (re-export of the canonical implementations).
- `experiments/archive/` — 31 stale notebook builders, legacy evaluators,
  and superseded pipeline scripts moved here with category subfolders
  and a README.
- `tests/test_v42_boundary_guard_proxy.py` — 11 new tests covering
  guard-triggered chat, streaming SSE, upstream pass-through, upstream
  timeout/unreachable → 502, proxy allowlist enforcement, surrogate
  input handling, and exact rule-count assertion.
- `experiments/prism_geometry_trajectory.py` and
  `experiments/prism_geometry_trajectory_2026-05-15.json` — PRISM
  geometry scan across v55/v56/v57/v58 merged models confirming the
  v42–v59 SFT track did NOT move quantization hostility.
- `ONBOARDING.md` and `CHANGELOG.md` (this file).
- `D:/kaggle/notebooks/haic-guard-reproducibility/` — third public
  Kaggle notebook demonstrating `guard + v42` independent reproducibility.

### Changed
- Guard `tools/v42_boundary_guard.py`:
  - Shared `httpx.AsyncClient` via FastAPI lifespan (was leaking
    per-request clients).
  - 502 `bad_gateway` JSON response on upstream timeout / connect
    errors (was raising and surfacing as 500 with stack).
  - SHA3 hashing uses `errors="replace"` (surrogate-safe).
  - Catch-all proxy restricted to GET/HEAD on `{health, props, slots,
    metrics, v1/models, models}` allowlist — was forwarding all
    methods/paths.
  - `_upstream` moved off module global to `app.state`.
  - **Matching behavior unchanged** → H18r4 anchor still valid.
- `maestro_gateway/app.py`:
  - Default `MAESTRO_LAUNCH_MODE` changed from `"test"` to
    `"production"` (fail-closed). Tests opt in via env in
    `tests/test_maestro_gateway.py`.
  - Chat endpoint capped at 40 messages × 8 KB content per message.
- `maestro_integration/maestro_client.py`:
  - Bare `except Exception` narrowed to `requests.RequestException`
    plus `ValueError`/`KeyError` with debug logging.
- `pyproject.toml`:
  - Populated `dependencies` (fastapi, uvicorn, httpx, requests,
    pydantic, numpy) — fresh installs no longer broken.
  - New `ml` optional-dependency group (torch, transformers, peft).
- `onchain/live_roundtrip.py`:
  - Anvil private key documented as public foundry default with env-var
    override and safe-secret-scan marker.
- `.gitignore` extended to exclude `onchain/{out,cache,lib,broadcast}/`.
- Submission notebook `notebook/haic_gemma4_governance.ipynb`:
  - Cell 38 (Scenario 6) and cell 40 ("Final Evaluation Results")
    updated to surface `guard + v42` as the promoted live candidate
    with the H18r4 anchor.
- Documentation rule count corrected to 16 across README, WRITEUP,
  next_steps, both H18 verdicts, submission_verification_report.
  Earlier wording said "17" but the actual code is 16 rules.

### Fixed
- Test count reference updated from 668 → 679 in README, next_steps.
- Broken archive paths in `docs/promotion_workflow.md`,
  `docs/kaggle_launch_checklist.md`, `docs/rigorous_eval_methodology.md`,
  `experiments/run_rigorous_comparison.py`.

## [0.1] — 2026-05-15

### Added — H18r4 PASS, guard + v42 promoted
- `tools/v42_boundary_guard.py` (16 rules, 4 classes, port 8082).
- `tests/test_v42_boundary_guard.py` (60 tests).
- `docs/v42_guard_h18_verdict_2026-05-15.md` — first-run FAIL (rubric
  phrase artifact).
- `docs/v42_guard_h18r4_verdict_2026-05-15.md` — **H18r4 PASS** with
  canonical anchor
  `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`.
- `experiments/v42_guard_h18r4_canonical.json` + zero-record sidecar.
- Tier 3 live-validation kernel pushed to Kaggle as Version 12 with
  public visibility (`is_private: false`).

### Added — v55–v59 fine-tuning track (experimental appendix)
- v55: best balanced SFT, failed direct-injection floor.
- v56: targeted mixed SFT, failed H14.
- v57: production-candidate design, failed H15.
- v58: boundary-first SFT, failed H16.
- v59: strongest fine-tuned result, failed H17 injection + jailbreak.
- Each with precommitted hypothesis + canonical verdict in `docs/`.

### Added — viability and PRISM
- `viability/distributed_viability.py` — federation-level Ceff/E.
- Tier 3 live validation: PRISM geometry + viability + Maestro receipt
  on Kaggle T4. Viability VIOLATED (Ceff/E = 0.879) — architectural
  finding, framework correctly diagnoses.

## Earlier (pre-0.1)

The pre-0.1 lineage (v34–v54, governance demo notebook, NLA stage 1) is
not summarized here. See dated docs in `docs/` from 2026-04-20 onward.

---

## Versioning policy

- A new H-series hypothesis is a minor version bump only when it
  produces a passing verdict that supersedes the current promoted
  candidate.
- Failed H-series hypotheses are recorded but do not bump the version.
- The version in `pyproject.toml` is the source of truth for the
  public release tag.
