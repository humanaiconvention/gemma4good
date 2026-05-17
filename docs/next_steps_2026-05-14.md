# Gemma4Good Next Steps — 2026-05-14

## Current Decision (updated 2026-05-16)

**`guard-v5 + v42` (H22) is the current promoted live candidate** as of
2026-05-16 late evening. All 16 non-compensatory predeclared gates
passed: H21's multi-message closure + Unicode normalization carried
forward, plus client-supplied `role: system` rejection (15/15 reject
at non-first positions, 10/10 legitimate pos-0 system prompts forwarded
correctly, 0/20 benign multi-turn FP across 13 languages). v42 weights
unchanged. Guard-v5 is `tools/v42_boundary_guard_v5.py` serving on port
8086. Anchor:
`5f2e796cf5afe1665c6084a7ccf9e43c419555178e08653f21c5d7234f359abc`.

Discipline trail (48 hours, four anchored PASSES + one anchored FAIL, no gate relaxation):
- **H18r4** PASS (2026-05-15) anchored `18e2c5a5…` — original guard, ASCII attacks only.
- **H19** FAIL (2026-05-16 early) — suite-design flaw confused multi-message with rule coverage AND precommit-vs-suite inconsistency on system-role. Published honestly.
- **H20** PASS (2026-05-16 afternoon) anchored `56ce960993f9…` — Unicode-bypass closure (L-01) isolated.
- **H21** PASS (2026-05-16 evening) anchored `d916ef63…` — multi-message attack closure (L-02) isolated.
- **H22** PASS (2026-05-16 late evening) anchored `5f2e796cf5af…` — client-supplied system-role rejection (L-02b) isolated, with explicit D2b predicate for legitimate operator pos-0 system prompts. **Current promoted candidate.**

**All eight documented limitations from `docs/v42_guard_known_limitations_2026-05-15.md` are now resolved.**

Per-item:
- L-01, L-02, L-02b, L-07: CLOSED via H-series anchors (L-01 H20, L-02 H21, L-02b H22) or removal (L-07).
- L-03 (rules walk perf): documented as intentional audit-metadata requirement.
- L-04 (rate limit): routed to deployment layer per `docs/gateway_deploy_plan.md`.
- L-05 (uvicorn access log): locked in by `tests/test_v42_boundary_guard_logging.py`.
- L-06 (single-chunk SSE): documented as intentional given short response length.

- Guard anchor: `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`
- Verdict: `docs/v42_guard_h18r4_verdict_2026-05-15.md`

The honest submission story:

- The governance loop is the primary contribution.
- v42 is the base model; guard + v42 is the promoted security endpoint.
- v58/v59 are valuable experimental appendix results (best model-only
  explicit refusal), but neither cleared promotion gates without the guard.
- The guard demonstrates that deterministic runtime governance can close
  injection/jailbreak/disclosure gaps that SFT alone could not.

## Immediate Work

1. Keep submission-facing docs aligned:
   - README points through v59/H17.
   - WRITEUP distinguishes v38/v39 governance-demo artifacts from the v42
     live security reference.
   - `docs/submission_alignment_2026-05-13.md` lists v58/v59 as appendix
     evidence, not production replacements.
   - `docs/submission_verification_report_2026-05-14.md` records the current
     Kaggle verification status, including unresolved public-visibility checks.

2. Run a final repo verification pass before submission:
   - `python -m pytest tests/`
   - `python experiments/runtime_loop_stress_test.py`
   - `bash verify_all.sh` where the shell environment supports it.

3. Check notebook-facing claims:
   - no stale "current production" language unless track-specific;
   - no v58/v59 promotion implication;
   - no performance claim without artifact, command, seed/sample count, and
     anchor.

4. Preserve live operations:
   - v42 should be served from
     `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf`.
   - v58/v59 GGUFs remain artifacts only.

5. Preserve response-level diagnostics:
   - Future canonical candidate evaluations should use
     `experiments/canonical_eval.py --failure-sidecar experiments/v<N>_failures_full.jsonl`.
   - The canonical JSON remains the promotion artifact; the sidecar is the
     required diagnostic artifact for failed candidates.

6. Deterministic boundary guard — **H18 PASSED 2026-05-15**:
   - Guard: `tools/v42_boundary_guard.py` (port 8082, 16 rules, 60 tests).
   - H18r4 verdict: `docs/v42_guard_h18r4_verdict_2026-05-15.md`.
   - Anchor: `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`.
   - All 13 gates PASS. `guard + v42` is now the promoted candidate.
   - The canonical submission endpoint is port 8082 (guard) → 8081 (v42).

## Verification Performed

Current pass on 2026-05-14:

- `python -m pytest tests/` passed: 679 passed (60 guard tests after DI-06
  addition + 2 new DI-06 trigger tests), 1 dependency deprecation warning.
- `python experiments/runtime_loop_stress_test.py` passed: 7 streams passed,
  0 failed.
- Runtime stress receipt:
  `895fe57109d260ca5e494be0a5346a7922c083f019356a7458a04abec0302cb8`.
- `bash verify_all.sh` could not run because local WSL failed to mount
  `E:/WSL/ext4.vhdx` (`ERROR_PATH_NOT_FOUND`). The same verification steps were
  run directly in PowerShell/Python:
  - 608 tests passed.
  - Runtime-loop coverage subset passed: 85 tests, 97% total coverage.
  - Runtime stress test passed: 7 streams, 0 failed.
  - Fresh clinic, classroom, and deforestation receipts matched the committed
    reference receipts after timestamp/self-anchor normalization.
- Live server check: `http://127.0.0.1:8081/health` returned OK and `/props`
  reported model path
  `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf` with
  `reasoning_format: none`.
- Notebook claim audit completed: the submitted notebook now distinguishes the
  historical v35/v38 governance-demo lineage from v42 as the live
  semantic-interviewer security reference.
- Tier 3 Version 11 completed on Kaggle with SGT `10.0`, `0` security fails,
  Merkle root
  `54ee8df6e57529d921467b2d863fc3e42faafe1f58e8f2b1f608414348f4fbcd`,
  and viability still false (`Ceff/E = 0.879055`). The result is retained in
  `experiments/tier3_v11_results_2026-05-15.json`.
- Tier 3 public visibility **resolved 2026-05-15**: Version 12 pushed at 17:03
  local after GPU quota reset, completed at 17:08, HTTP 200 confirmed
  unauthenticated. Results identical to v11 (deterministic). Artifact committed
  as `experiments/tier3_v12_results_2026-05-15.json`.

## Do Not Do By Default

- Do not launch v60 by momentum.
- Do not tune gates after seeing a failure.
- Do not replace v42 with v58 or v59.
- Do not cite unmeasured architectural hypotheses as project evidence.

## Possible Follow-Up Research

These are follow-ups, not submission blockers:

- Add a measured architecture/geometry diagnostic for the actual Gemma 4 E2B,
  v42, and v59 merged checkpoints to see whether fine-tuning changes the
  substrate-level signal or only surface behavior.
- Build a deterministic boundary guard around the v42 live path and evaluate it
  under the same canonical scenarios before considering further model-only SFT.
- Add a rubric audit that separates harmless refusal mentions of protected
  material from true prompt/protocol leakage, while keeping the original
  conservative gate for promotion decisions until a replacement is
  predeclared.

## Submission Posture

Lead with the auditable governance architecture:

- Gemma 4 tool calls execute wellbeing, consent/provenance, PRISM-style drift,
  NLA/mock-NLA, and Merkle receipt checks.
- Promotion is reproducible, seeded, anchored, and non-compensatory.
- Negative model results are included because the project is about reliable
  governance, not a prettier demo.
