# v56 Hypothesis — Targeted Mixed SFT

Date: 2026-05-14 PDT

## Why v56 Exists

v55 was the best balanced fine-tuned result so far, but it failed the
precommitted H13 gate because `sgt_adversarial_inject` reached 0.88 against a
0.90 floor. Social engineering also remained below v42.

v56 is a narrow data-coverage test. It does not change the format or step count.

## Hypothesis H14

The remaining v55 gap is caused by insufficient positive coverage for direct
injection and social-engineering prompts. Increasing those positive examples
will improve the targeted scenarios without destroying explicit concealed
refusal.

## Design

- Base: v42 adapter warm-start
- Training format: user-only `apply_chat_template`, exactly like v51/v55
- Refusal data: 400 `v47_dpo_pairs.jsonl` prompt/chosen pairs
- Direct-injection positives: 80
- Indirect-injection positives: 40
- Social-engineering positives: 40
- Positive completions: deterministic v42 completions generated inside the
  Kaggle notebook before SFT and filtered for no obvious compliance/leak
- Training method: TRL `SFTTrainer`
- Loss: `completion_only_loss=True`
- Steps: 100
- LR: 5e-5

## Precommitted H14 Gates

| Predicate | Threshold |
|---|---:|
| H14a: aggregate_security | >= 0.87 |
| H14b: strict_concealed_refusal | >= 0.35 |
| H14c: empty_response_rate | <= 0.05 |
| H14d: strict_concealed_leak | <= 0.05 |
| H14e: `sgt_adversarial_inject` | >= 0.90 |
| H14f: `sgt_indirect_inject` | >= 0.90 |
| H14g: `sgt_social_engineering` | >= 0.65 |

## Stop Condition

If H14 fails, stop the fine-tuning chase for this submission. v55/v56 together
answer the data-coverage question well enough for the project goal: mixed SFT
is useful, but promotion remains governed by non-compensatory gates.

## Implementation Artifacts

- Builder: `experiments/build_v56_nb.py`
- Kaggle notebook path:
  `D:/kaggle/notebooks/haic-gemma4-v56-sft-targeted-mixed/`
- Quantization config:
  `experiments/quantize_warmstart_direct.py --version v56`
- Expected eval output:
  `experiments/v56_canonical_old_prompt.json`

## Run Notes

- v56 initial Kaggle attempt stopped before training. The guardrail collected
  80/80 direct-injection positives and 40/40 indirect-injection positives, but
  only 37/40 social-engineering positives. No adapter was produced and no
  model evidence should be inferred from that attempt.
- v56 v2 keeps the same H14 hypothesis, gates, format, step count, and
  acceptance filter. The only change is a larger social-engineering candidate
  prompt pool so the notebook can still require 40 clean v42 target responses
  before SFT begins.
