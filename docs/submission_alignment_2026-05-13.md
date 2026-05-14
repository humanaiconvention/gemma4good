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

The v50-v59 fine-tuning sequence should be presented as an experimental appendix,
not the primary submission claim.

Known results:

- v50 DPO collapsed into mostly empty responses under canonical eval.
- v51 user-only SFT substantially improved explicit concealed refusal but
  regressed injection robustness.
- v52 system-in-user format failed.
- v53 proper system+user at 60 steps failed.
- v54 proper system+user at 100 steps failed.
- v55 mixed user-only SFT was the best balanced fine-tuned result, but failed
  the direct-injection floor.
- v56 targeted mixed SFT failed H14 and triggered the stop condition.
- v57 tested a new precommitted production-candidate design under H15 and
  failed. It should be presented as a negative result, not as a replacement for
  v42.
- v58 boundary-first SFT was a genuine improvement over v55-v57 and produced
  very high explicit concealed refusal, but failed non-compensatory H16 gates
  for direct adversarial injection and disclosure-preview markers.
- v59 targeted the residual v58 failures and became the strongest fine-tuned
  result to date, but still failed non-compensatory H17 gates for direct
  adversarial injection and jailbreak robustness.
- v42 remains the production reference.

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
- `docs/v55_canonical_verdict_2026-05-14.md`
- `docs/v56_hypothesis_2026-05-14.md`
- `docs/v56_canonical_verdict_2026-05-14.md`
- `docs/v57_production_candidate_plan_2026-05-14.md`
- `docs/v57_canonical_verdict_2026-05-14.md`
- `docs/response_failure_taxonomy_v42_v55_v57_2026-05-14.md`
- `docs/v58_precommit_plan_2026-05-14.md`
- `docs/v58_residual_failure_taxonomy_2026-05-14.md`
- `docs/v58_canonical_verdict_2026-05-14.md`
- `docs/v59_precommit_plan_2026-05-14.md`
- `docs/v59_canonical_verdict_2026-05-14.md`

## Known Consistency Risk

Some older narrative docs describe earlier models such as v38/v39 as "current"
or "production" in their historical context. For the fine-tuning security track,
the current production reference is v42 unless a later candidate beats it under
canonical eval. Do not rewrite historical verdict docs to pretend they were
written later; instead, make the current status explicit in new summary docs.

The current submission distinction is:

- v38/v39 remain historical governance-demo and promotion-doctrine artifacts.
- v42 is the live semantic-interviewer / fine-tuning security reference.
- v58/v59 are experimental appendix results, not production replacements.

## Rule For New Claims

Every new model claim needs:

- artifact path
- eval command
- seeds/sample counts
- JSON self-anchor
- precommitted predicate result
- honest pass/fail/partial verdict

No artifact means no verdict.
