# v39 Evaluation Runbook

*One-paste workflow for evaluating v39 once the Kaggle run completes.*

This runbook turns the doctrine into a sequence of concrete commands. The
output of step 4 is the load-bearing decision; everything else is plumbing.

---

## Pre-conditions

- Kaggle kernel `benhaslam/haic-gemma4-v39-pivot` has status `COMPLETE`
  (check via `kaggle kernels status benhaslam/haic-gemma4-v39-pivot`).
- BEAST GPU is free (`nvidia-smi`).
- You're in `D:/gemma4good` with the `master` branch checked out.

---

## Step 1 — Pull the v39 adapter from Kaggle

```bash
mkdir -p D:/kaggle/adapters/haic-gemma4-v39-adapter
kaggle kernels output benhaslam/haic-gemma4-v39-pivot \
    -p D:/kaggle/adapters/haic-gemma4-v39-adapter
```

Verify:

```bash
ls D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter/
# expect: adapter_config.json, adapter_model.safetensors, tokenizer files...
```

---

## Step 2 — LoRA-shape sanity check (catch SimSat-style null training)

```bash
cd D:/gemma4good
python -c "
from safetensors import safe_open
import collections
p = 'D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter/adapter_model.safetensors'
keys = []
with safe_open(p, framework='pt') as f:
    for k in f.keys(): keys.append(k)
lora = [k for k in keys if 'lora' in k.lower()]
print(f'LoRA tensors: {len(lora)}  (expect 490 for Gemma-4-E2B language_model-only LoRA)')
buckets = collections.Counter(
    'language_model' if 'language_model' in k else
    'vision_tower' if 'vision_tower' in k else
    'audio_tower' if 'audio_tower' in k else 'other'
    for k in lora
)
print(f'Buckets: {dict(buckets)}')
assert buckets.get('vision_tower', 0) == 0 and buckets.get('audio_tower', 0) == 0
print('OK: LoRA scoped to language_model only.')
"
```

If this fails (vision_tower or audio_tower count > 0), do not proceed —
the v33-class regression has returned. Diagnose before evaluating.

---

## Step 3 — Eval-set leakage check (Gate 2)

```bash
cd D:/gemma4good
# Extract v39 synthetic from the build script (v39 = v38 synthetic + 1 new example)
python -c "
import sys, json
sys.path.insert(0, 'D:/kaggle/scripts')
from build_v39_nb import SYNTHETIC_EXAMPLES
out = []
for ex in SYNTHETIC_EXAMPLES: out.append({'messages': ex['messages']})
with open('experiments/v39_synthetic_extracted.jsonl', 'w') as f:
    for o in out: f.write(json.dumps(o) + '\n')
print(f'Wrote {len(out)} v39 synthetic examples')
"

python -m tools.eval_leakage_check \
    --training D:/kaggle/datasets/v35_gov_push/v35_gov_final.jsonl \
              experiments/v39_synthetic_extracted.jsonl \
    --threshold 0.4 \
    --out experiments/v39_leakage_receipt.json
```

Expected: `Verdict: PASS`. If REVIEW_NEAR_MATCH or BLOCKED_EXACT_MATCH,
inspect the JSON for which scenario tripped the check.

---

## Step 4 — Run rigorous SGT (single command, full pipeline)

This is **the load-bearing step.** Single command, runs everything: rigorous
SGT (deterministic + n=20 sampling), leakage check, six-gate decision,
Merkle eval-receipt mint. Use the refined rubric (the doctrine default).

```bash
cd D:/gemma4good

# 1-turn rigorous, with baseline, n=20 (tightens the CIs that v38 couldn't clear at n=10)
python -u -m tools.evaluate_promotion \
    --model haic-gemma4-v39 \
    --base D:/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3 \
    --adapter D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter \
    --training D:/kaggle/datasets/v35_gov_push/v35_gov_final.jsonl \
              experiments/v39_synthetic_extracted.jsonl \
    --baseline \
    --n-samples 20 \
    --seed 42 \
    --rubric refined \
    --profile default \
    --out experiments/v39_evaluation.json \
    2>&1 | tee experiments/v39_evaluation.log
```

Wall-clock: ~3 hours on RTX 2080 (v39 ~25 min sampling + base ~2.5 hr
sampling — base produces longer outputs).

The CLI prints the gate-by-gate verdict and exits with:
- `0` → PROMOTED
- `1` → BLOCKED
- `2` → INDETERMINATE

---

## Step 5 — Mint the Merkle eval receipt (already done in step 4)

`experiments/v39_evaluation.json` contains the full pipeline result.
For the canonical receipt format (matching v38's), regenerate via:

```bash
# Extract the SGT pass and decision from the orchestrator output, then
# run the canonical receipt minter:
python -c "
import json
d = json.load(open('experiments/v39_evaluation.json'))
json.dump(d['sgt'], open('experiments/v39_sgt_rigorous.json', 'w'), indent=2)
json.dump(d['decision'], open('experiments/v39_promotion_decision.json', 'w'), indent=2)
"

python -m tools.eval_receipt \
    --sgt experiments/v39_sgt_rigorous.json \
    --leakage experiments/v39_leakage_receipt.json \
    --decision experiments/v39_promotion_decision.json \
    --out experiments/v39_eval_receipt.json
```

Note the printed receipt root — that's the canonical anchor for v39.

---

## Step 6 — Compare to v38

```bash
cd D:/gemma4good
python -c "
import json
v38 = json.load(open('experiments/v38_sgt_rigorous_2turn_with_baseline_refined.json'))
v39 = json.load(open('experiments/v39_sgt_rigorous.json'))

def grab(rep, side, which, key):
    return rep[side][which].get(key)

print('               v38 (2-turn refined)        v39 (1-turn refined)')
print('grounding samp', grab(v38,'finetune','sampling','grounding_pass_rate'),
      '                     ',
      grab(v39,'finetune','sampling','grounding_pass_rate'))
print('grounding CI95', grab(v38,'finetune','sampling','grounding_ci95'),
      '             ',
      grab(v39,'finetune','sampling','grounding_ci95'))
print('security samp ', grab(v38,'finetune','sampling','security_pass_rate'),
      '                       ',
      grab(v39,'finetune','sampling','security_pass_rate'))
print('security CI95 ', grab(v38,'finetune','sampling','security_ci95'),
      '         ',
      grab(v39,'finetune','sampling','security_ci95'))
"
```

Note the methodological asymmetry: v38's full data was 2-turn rigorous.
v39 starts at 1-turn (the more demanding test). For apples-to-apples
2-turn comparison, additionally run:

```bash
python -u -m experiments.run_v38_sgt_2turn \
    --adapter D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter \
    --baseline \
    --n-samples 20 --seed 42 \
    --out experiments/v39_sgt_rigorous_2turn_with_baseline.json
```

(That's another ~2.5 hours but gives the apples-to-apples comparison.)

---

## Step 7 — Commit

```bash
cd D:/gemma4good
git add experiments/v39_evaluation.json \
        experiments/v39_evaluation.log \
        experiments/v39_sgt_rigorous.json \
        experiments/v39_promotion_decision.json \
        experiments/v39_leakage_receipt.json \
        experiments/v39_eval_receipt.json \
        experiments/v39_synthetic_extracted.jsonl
git commit -m "data(v39): rigorous evaluation receipts — <DECISION> on <DATE>"
```

If PROMOTED, also update WRITEUP.md to point at v39 with the new
`eval_receipt_root`. If BLOCKED, add a note to `docs/v39_recipe.md`
naming which gate(s) failed and what the next iteration would change.

---

## Hypothesis tracker

For each falsifiable prediction in [`docs/v39_recipe.md`](./v39_recipe.md),
record the verdict here after the run completes.

| Prediction | Status | Evidence |
|---|---|---|
| Restoring `train_on_responses_only` lifts grounding signal density | TBD | (compare v39 sampling 1-turn grounding vs v38's 36.7%) |
| Synthetic ×1 + response-only mask is sufficient (drop ×3) | TBD | (compare to v38; v39 should not regress) |
| Single Paris-refusal example closes the 1/20 leak | TBD | (look for "capital is Paris" patterns in v39 sampling security responses) |
| In-kernel mini-SGT tracks within tolerance of offline rigorous | TBD | (compare cell 6 output to step 4 numbers) |
| n=20 samples make Gate 1 CIs disjoint at v39's lift size | TBD | (look at finetune CI vs baseline CI from step 4) |

If all five fall on the predicted side, the recipe is validated. If
any fall against, that's a real lesson for v40 — name the failure mode.

---

*Author: Claude Opus 4.7 · 2026-05-09 · while v39 runs on Kaggle.*
