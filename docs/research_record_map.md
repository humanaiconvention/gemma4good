# Research Record Map

This map is post-submission aftercare. It helps a cold reader navigate the
project without mistaking historical experiment notes for the final
submission state.

## Read First

1. `docs/submission_manifest_2026-05-18.md` — the exact submitted snapshot
   and load-bearing file set.
2. `WRITEUP.md` — the public submission narrative.
3. `docs/h26_verdict_2026-05-17.md` — the final promoted candidate verdict.
4. `docs/evaluation_doctrine.md` — the predeclared, non-compensatory gate
   discipline.
5. `docs/what_the_guard_actually_does.md` — the empirical v42-vs-guard
   explanation.
6. `docs/v42_guard_known_limitations_2026-05-15.md` — the limitations ledger;
   as of H26, all documented L-01 through L-09 items are closed, routed to
   another layer, or intentionally out of scope.

## Final Submission State

- The project was submitted to Kaggle on 2026-05-18.
- The submitted repo snapshot is commit `ec7db2e`.
- The promoted candidate is `guard-v7 + v42`.
- The H26 anchor is
  `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8`.
- The submission's strongest claim is not "fine-tuning won." It is that a
  governance loop can reject weak candidates and promote only the artifact
  that clears predeclared gates.

## Track Map

### 1. Governance Architecture

Core files:

- `notebook/haic_gemma4_governance.ipynb`
- `tools/haic_tools.py`
- `viability/viability_condition.py`
- `viability/session_gates.py`
- `viability/distributed_viability.py`
- `tools/diloco_fragment_verifier.py`
- `tools/enforcement_evidence_contract.py`
- `utils/merkle.py`

Key docs:

- `docs/viability_condition.md`
- `docs/runtime_grounding_loop_2026-05-11.md`
- `docs/diloco_integration_2026-05-11.md`
- `docs/TIER3_RUNBOOK.md`
- `docs/submission_verification_2026-05-16.md`

### 2. Fine-Tuning Appendix

The v50-v59 sequence is a negative/partial-results record, not the promoted
path.

- v50: DPO collapse into empty responses.
- v51: user-only SFT improved explicit refusal but regressed injection
  robustness.
- v52-v54: system/user format variants failed to bind refusal.
- v55: best balanced SFT, still failed direct-injection floor.
- v56: targeted mixed SFT failed H14.
- v57: production-candidate hypothesis failed H15.
- v58: boundary-first SFT improved refusal but failed H16.
- v59: strongest fine-tuned result, failed H17 on direct injection and
  jailbreak.

Primary docs:

- `docs/v50_canonical_verdict_2026-05-12.md`
- `docs/v51_canonical_verdict_2026-05-13.md`
- `docs/v52_canonical_verdict_2026-05-13.md`
- `docs/v53_canonical_verdict_2026-05-13.md`
- `docs/v54_canonical_verdict_2026-05-13.md`
- `docs/v55_canonical_verdict_2026-05-14.md`
- `docs/v56_canonical_verdict_2026-05-14.md`
- `docs/v57_canonical_verdict_2026-05-14.md`
- `docs/v58_canonical_verdict_2026-05-14.md`
- `docs/v59_canonical_verdict_2026-05-14.md`

### 3. Guard H-Series

This is the promoted path. Each step has a precommit/verdict discipline.

| Step | Result | Meaning |
|---|---|---|
| H18r4 | PASS | ASCII baseline guard promoted |
| H19 | FAIL | combined Unicode + multi-message attempt exposed suite flaws |
| H20 | PASS | Unicode bypass closed |
| H21 | PASS | multi-message scan closed |
| H22 | PASS | client-supplied system role closed |
| H23 | PASS | encoded-payload behavioral defense held; L-08 surfaced |
| H25 | FAIL | native-language attack bypass confirmed; L-09 surfaced |
| H24 | PASS | leet-fold closes L-08 |
| H26 | PASS | multi-language rules close L-09; final submitted candidate |

Primary docs:

- `docs/v42_boundary_guard_precommit_2026-05-14.md`
- `docs/v42_guard_h18r4_verdict_2026-05-15.md`
- `docs/h19_verdict_2026-05-16.md`
- `docs/h20_verdict_2026-05-16.md`
- `docs/h21_verdict_2026-05-16.md`
- `docs/h22_verdict_2026-05-16.md`
- `docs/h23_verdict_2026-05-16.md`
- `docs/h24_verdict_2026-05-16.md`
- `docs/h25_verdict_2026-05-16.md`
- `docs/h26_verdict_2026-05-17.md`

### 4. Reproducibility And Submission

- `docs/submission_manifest_2026-05-18.md`
- `docs/pre_submission_checklist_2026-05-18.md`
- `docs/submission_verification_2026-05-16.md`
- `assets/media_gallery/README.md`
- `video/README.md`

The public Kaggle guard reproducibility notebook currently demonstrates H18r4,
not the full H26 rule set. The submitted WRITEUP is clear that H26 is the
current promoted candidate and that H18r4 is the fastest public kernel demo.

## Historical Document Rule

Do not rewrite dated verdicts to remove the uncertainty that existed when they
were written. The project gains credibility because it preserves failures and
wrong turns. Only active reader-entry documents such as `README.md`,
`docs/ONBOARDING.md`, `docs/REPO_STATUS.md`, and this map should be kept in
the final current-state voice.
