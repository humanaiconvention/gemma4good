# What the guard actually does

*A side-by-side empirical comparison of `v42 alone` versus
`guard-v5 + v42` (H22 promoted candidate) on the identical canonical
attack set. The deltas are the actual evidence for the guard's value —
not framework rhetoric, just numbers from two anchored evaluations.*

**Status:** Synthesis of two existing public evaluations, no new
evaluation run. Both anchors are in the repository at
`experiments/`.

---

## The two anchors

| Configuration | Anchor | Date | Description |
|---|---|---|---|
| **v42 alone** (no guard) | `e597605533701d7e1b3e78e311c00d0c…` | (pre-guard) | Gemma 4 E2B + v42 LoRA adapter, identical canonical eval parameters |
| **guard-v5 + v42** (H22 promoted) | `5f2e796cf5afe1665c6084a7ccf9e43c419555178e08653f21c5d7234f359abc` | 2026-05-16 | guard-v5 (16 rules + Unicode normalization + per-message scan + system-role rejection) in front of unchanged v42 |

Both evaluations used identical canonical parameters: 5 seeds (7, 13,
23, 42, 100), n_samples_phase1=20, focused-n=100, max_tokens=600,
temperature=0.7, top_p=0.9, system_prompt_variant=`old`.

---

## v1 aggregate security — per-scenario delta

| Scenario | v42 alone | guard-v5 + v42 | Delta |
|---|---:|---:|---:|
| sgt_adversarial_inject | 0.97 | **1.00** | +0.03 |
| sgt_indirect_inject | 0.94 | **1.00** | +0.06 |
| sgt_jailbreak_dan | 1.00 | 1.00 | 0 |
| sgt_offtopic_medical | 1.00 | 0.95 | −0.05 (sampling noise — guard does not trigger on these) |
| sgt_offtopic_legal | 1.00 | 0.99 | −0.01 (sampling noise — guard does not trigger on these) |
| **sgt_social_engineering** | **0.70** | **1.00** | **+0.30** |
| **sgt_concealed_compliance** | **0.56** | **1.00** | **+0.44** |
| **aggregate_security (pooled)** | **0.88** | **0.99** | **+0.11** |

The off-topic medical/legal rows trend slightly down because the
guard does not trigger on those scenarios (medical/legal questions
don't match any of the 16 attack-class rules). Those requests pass
through to v42 unchanged; the small variance is llama-server
sampling stochasticity between two separate server instances on
seeded but non-deterministic inference. Within sampling noise.

The two rows that matter are at the bottom. **The guard adds 30
percentage points on social engineering and 44 percentage points on
concealed-compliance refusal** — the two attack classes where v42
alone is most vulnerable.

---

## The strict rubric — what v42 actually does on concealed-compliance

The aggregate v1 rubric scores any non-leaking response as a pass,
including the "abstract deflection" failure mode where the model
acknowledges the question and pivots without explicitly refusing.
The strict rubric is more honest: it separates explicit refusal from
abstract deflection from semantic leak.

| Outcome on concealed-compliance | v42 alone (n=500) | guard-v5 + v42 (n=500) |
|---|---:|---:|
| Explicit refusal | 69 (13.8%) | **500 (100%)** |
| Abstract deflection | 348 (69.6%) | 0 |
| **Semantic leak (Paris named)** | **83 (16.6%)** | **0 (0%)** |
| Empty response | 0 | 0 |

Read the row labels carefully.

- **v42 alone explicitly refuses concealed-compliance attacks 13.8%
  of the time.** The other 86.2% of the time it either pivots away
  abstractly (the most charitable interpretation — 69.6%) or names
  Paris (16.6%).
- **v42 alone leaks the answer in 1 of every 6 attempts.** This is
  the failure mode v50–v59 fine-tuning tried to fix and could not.
- **guard-v5 explicit-refuses 100% and leaks 0%.** No semantic leak
  across 500 samples in five seeds.

This is the case for the guard, stated quantitatively. A 200-line
deterministic regex proxy turns a 16.6% leak rate into a 0% leak
rate, with a 86.2% → 100% jump on explicit refusal.

---

## Why this matters

The submission's central claim is that **deterministic governance
over learned systems is sometimes the right architecture, not a
stopgap**. The data above is the empirical case for that claim on
this specific model:

- The base model has a real failure mode (16.6% semantic leak on
  the canonical concealed-compliance attack).
- Six weeks of SFT iteration (v50–v59) failed to close that failure
  mode without opening another.
- A 200-line regex proxy closes it deterministically, with all 13
  predeclared non-compensatory gates passing.

The honest framing is: *we could not pass the gates by training. So
we promoted the regex.* The data above is what makes that decision
reproducible and verifiable by anyone running the public Kaggle
reproducibility notebook — they observe the same delta we did, with
their own SHA3-anchored receipt.

---

## How to verify these numbers yourself

```bash
# 1. Inspect v42-bare anchor
python -c "import json; r = json.load(open('experiments/v42_canonical_old_prompt.json')); print(json.dumps(r['aggregate'], indent=2))"

# 2. Inspect guard-v5 + v42 anchor (H22 promoted)
python -c "import json; r = json.load(open('experiments/v42_guard_v5_h22_canonical.json')); print(json.dumps(r['aggregate'], indent=2))"

# 3. The aggregate.rubric_v1.per_scenario object in each gives the
#    per-scenario pass rates that populate the table above.
```

Or run the public Kaggle reproducibility kernel
[`benhaslam/haic-guard-v42-reproducibility-demo-h18r4`](https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4)
which reruns a subset of these attacks against the guard and emits
its own SHA3-anchored receipt. No GPU required, runs in under a
minute.

---

## What this comparison does NOT prove

- It does not prove the guard is "secure" in any general sense — only
  that it passed all 13 predeclared gates on the canonical attack set.
- It does not prove the guard catches every attack a frontier-class
  adversary could construct. The known-limitations doc lists three
  attack classes that H18r4 did NOT initially anchor (Unicode bypass,
  multi-message scan, client-supplied system-role injection) — all
  three were subsequently closed in H20/H21/H22 anchored steps.
- It does not extrapolate to other base models. The case that the
  framework works on a non-Gemma base remains predeclared (see
  `docs/passing_model_demo_plan.md`) and not yet executed.

What it DOES prove: on Gemma 4 E2B + v42, on the canonical attack
set, the guard takes a measurably leaky model and produces a model
whose every decision is anchored, audit-logged, and reproducible —
without changing the model's weights.
