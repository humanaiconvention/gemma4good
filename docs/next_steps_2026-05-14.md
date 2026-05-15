# Gemma4Good Next Steps — 2026-05-14

## Current Decision

Keep `haic-gemma4-v42` live as the semantic-interviewer / fine-tuning security
reference. Do not promote v58 or v59.

The strongest fine-tuned candidate is v59, but H17 failed two predeclared,
non-compensatory gates:

- `sgt_adversarial_inject`: 0.95 vs required >= 0.97.
- `sgt_jailbreak_dan`: 0.96 vs required >= 0.97.

The honest submission story is therefore:

- The governance loop is the primary contribution.
- v42 remains the live reference because no later candidate cleared the
  promotion gates.
- v58/v59 are valuable experimental appendix results showing that explicit
  concealed refusal can be greatly improved, but not yet without residual
  injection/jailbreak misses under strict gates.

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

6. Prefer deterministic boundary testing before more model-only SFT:
   - Guard implemented: `tools/v42_boundary_guard.py` (port 8082, 16 rules,
     58 tests all passing). See `docs/v42_boundary_guard_precommit_2026-05-14.md`.
   - Eval launcher: `experiments/run_h18_guard_eval.py`.
   - Next step: run H18 canonical evaluation (requires v42 llama-server live
     on port 8081 and GPU time). Do not promote without H18 PASS.

## Verification Performed

Current pass on 2026-05-14:

- `python -m pytest tests/` passed: 666 passed (including 58 guard tests),
  1 dependency deprecation warning.
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
- Tier 3 public metadata has been patched locally to `is_private: false`, but
  Kaggle rejected the live push because the account has reached the weekly GPU
  quota. Public visibility remains a live blocker until a metadata update can
  be accepted and the URL rechecked.

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
