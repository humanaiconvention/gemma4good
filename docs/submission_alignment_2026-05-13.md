# Submission Alignment Notes — 2026-05-13

This note aligns the public submission posture with
`docs/project_goal_2026-05-13.md`.

## Lead Claim

The submission should lead with HAIC as a verifiable governance loop for
grounded AI:

- Gemma 4 function-calling executes governance tools.
- Consent/provenance checks bound which signals can enter `C_eff(t)`.
- PRISM-style measurements provide an `E(t)` proxy.
- Merkle receipts make decisions and evaluations auditable.
- Promotion is gated by predeclared, non-compensatory criteria.

## Fine-Tuning Track Placement

The v50-v55 fine-tuning sequence should be presented as an experimental appendix,
not the primary submission claim.

Known results:

- v50 DPO collapsed into mostly empty responses under canonical eval.
- v51 user-only SFT substantially improved explicit concealed refusal but
  regressed injection robustness.
- v52 system-in-user format failed.
- v53 proper system+user at 60 steps failed.
- v54 must not be documented until its canonical eval artifact exists.
- v55 is a controlled test of mixed user-only SFT, not a guaranteed fix.

## Submission Documents To Treat As Load-Bearing

- `README.md`
- `WRITEUP.md`
- `notebook/haic_gemma4_governance.ipynb`
- `docs/viability_condition.md`
- `docs/evaluation_doctrine.md`
- `docs/promotion_workflow.md`
- `docs/runtime_grounding_loop_2026-05-11.md`
- `docs/diloco_integration_2026-05-11.md`
- `docs/project_goal_2026-05-13.md`
- `docs/v55_hypothesis_2026-05-13.md`

## Known Consistency Risk

Some older narrative docs describe earlier models such as v38/v39 as "current"
or "production" in their historical context. For the fine-tuning security track,
the current production reference is v42 unless a later candidate beats it under
canonical eval. Do not rewrite historical verdict docs to pretend they were
written later; instead, make the current status explicit in new summary docs.

## Rule For New Claims

Every new model claim needs:

- artifact path
- eval command
- seeds/sample counts
- JSON self-anchor
- precommitted predicate result
- honest pass/fail/partial verdict

No artifact means no verdict.
