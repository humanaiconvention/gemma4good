# v55 Hypothesis — User-Only Mixed SFT

Date: 2026-05-13 PDT

## Why v55 Exists

v51 is the best fine-tuned result so far because it proved a real behavioral
lever: user-only SFT raised strict concealed-compliance explicit refusal from
13.8% to 63.4% with zero empty responses and only 2.0% semantic leaks.

The same run regressed aggregate security from v42's 0.88 to 0.7814. The main
regressions were:

- `sgt_adversarial_inject`: 0.97 -> 0.70
- `sgt_indirect_inject`: 0.94 -> 0.75
- `sgt_social_engineering`: 0.70 -> 0.54

v52 and v53 tested system-prompt format hypotheses and failed. v54 is still not
documentable until `experiments/v54_canonical_old_prompt.json` exists. As of
the first partial seed in the log, v54 is not trending toward rescuing explicit
refusal, but that is not a verdict.

## Hypothesis H13

Mixed supervision can preserve v51's user-only refusal binding while restoring
injection robustness if the training set includes positive examples where the
model must stay in the interviewer role under injection pressure.

## Design

- Base: v42 adapter warm-start
- Training format: user-only `apply_chat_template`, exactly like v51
- Refusal data: 400 `v47_dpo_pairs.jsonl` prompt/chosen pairs
- Injection-positive data: 80 held-out direct/indirect injection prompts
- Chosen injection-positive responses: deterministic v42 completions generated
  inside the Kaggle notebook before SFT
- Training method: TRL `SFTTrainer`
- Loss: `completion_only_loss=True`
- Steps: 100
- LR: 5e-5
- Effective batch: 8

The 80 injection-positive prompts must not be exact copies of canonical eval
prompts. This avoids training-on-test contamination while still testing the
same failure family.

## Precommitted H13 Gates

| Predicate | Threshold | Why it exists |
|---|---:|---|
| H13a: aggregate_security | >= 0.85 | Prevent a global security tradeoff |
| H13b: strict_concealed_refusal | >= 0.30 | Preserve real explicit refusal |
| H13c: empty_response_rate | <= 0.05 | Catch EOS/empty collapse |
| H13d: strict_concealed_leak | <= 0.20 | Do not regress beyond v42 leak level |
| H13e: `sgt_adversarial_inject` | >= 0.90 | Directly test v51's largest regression |
| H13f: `sgt_indirect_inject` | >= 0.85 | Directly test indirect-injection recovery |

## What Would Falsify H13

H13 is false if any of the following happen:

- Explicit refusal drops below 30%.
- Aggregate security remains below 85%.
- Direct or indirect injection stays meaningfully below the precommitted floors.
- Empty responses exceed 5%.
- Semantic leaks exceed 20%.

If H13 fails, the honest interpretation is not "try v56 immediately." The
result would show that user-only refusal-SFT can induce explicit refusal, but
the present training/rubric setup does not yet separate concealed-compliance
refusal from injection robustness.

## Implementation Artifacts

- Builder: `experiments/build_v55_nb.py`
- Kaggle notebook path: `D:/kaggle/notebooks/haic-gemma4-v55-sft-mixed/`
- Quantization config: `experiments/quantize_warmstart_direct.py --version v55`
- Expected eval output: `experiments/v55_canonical_old_prompt.json`
