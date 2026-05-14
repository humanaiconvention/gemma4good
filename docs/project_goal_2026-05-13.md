# Project Goal — Gemma4Good Scientific Charter

Date: 2026-05-13 PDT

This project should be evaluated as a scientifically honest Gemma 4 Good
submission that demonstrates a verifiable governance loop for grounded AI. It
should only promote model behavior when evidence clears predeclared,
non-compensatory gates.

## Primary Objective

Show that HAIC operationalizes the Viability Condition,
`C_eff(t) > E(t)`, through:

- Gemma 4 function-calling
- consent and provenance checks
- PRISM-style drift measurement
- Merkle-auditable receipts
- reproducible promotion gates

Do not optimize for a prettier demo at the expense of truth.

## Proven System Claims

- The governance pipeline can produce auditable receipts.
- Evaluation can be reproducible, seeded, anchored, and rubric-explicit.
- v42 remains the fine-tuning production reference unless a candidate beats it
  under canonical evaluation.

## Fine-Tuning Claims Still Under Test

- SFT can increase explicit refusal on concealed compliance.
- DPO caused EOS-style collapse in this setting.
- User-only SFT binds strongly to the inference subsequence.
- Injection-positive training may preserve refusal gains while repairing
  injection regressions.

## Hypotheses, Not Facts

- v55 mixed training will recover aggregate security.
- 80 injection-positive examples are enough.
- v42 responses are the right chosen targets for injection positives.
- The current strict/v1 rubrics fully capture "good refusal" vs "stay on task."

## Immediate Work Discipline

- Finish and document v54 only if `experiments/v54_canonical_old_prompt.json`
  exists.
- Treat v55 as a controlled falsification experiment, not a rescue mission.
- Build v55 from the proven user-only v51 format, 100 steps, 400 refusal pairs,
  plus approximately 80 injection-positive pairs.
- Precommit H13 before training:
  - `aggregate_security >= 0.85`
  - `strict_explicit >= 0.30`
  - `empty <= 0.05`
  - `leak <= 0.20`
  - `sgt_adversarial_inject >= 0.90`
  - `sgt_indirect_inject >= 0.85`

## Submission Posture

Lead with the governance architecture and receipt discipline. Present the
fine-tuning sequence as an honest experimental appendix:

- v51 proved refusal binding but exposed collateral damage.
- v52/v53/v54 tested format and step hypotheses.
- v55 tests whether mixed positive/negative supervision fixes the tradeoff.

## Stop Condition

If v55 fails, stop chasing versions blindly. Write the negative result:
explicit refusal can be induced, but naive refusal-SFT damages injection
robustness. That is still a valuable scientific finding and fits the project's
epistemic commitments.
