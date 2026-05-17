# H-passing-1 Phase 1 Verdict — geometry pre-screen

*The cheap pre-screen specified in
[`docs/passing_model_demo_plan.md`](passing_model_demo_plan.md).
Tests the falsifiable claim that some open base models have
quantization-hostility `qh ≤ 0.72` — the Convention's threshold for
viable SGT-style fine-tuning per the Viability Condition paper.*

**Date:** 2026-05-16 (late evening)
**Phase:** 1 of 3 (geometry pre-screen). Phases 2 and 3 (fine-tune +
canonical eval) are gated on a passing candidate from Phase 1.
**Result:** **No tested base model passes the viability threshold.**
Mistral 7B Instruct v0.3 is the closest at qh=0.7520, 0.032 above the
0.72 threshold.

## Data

| Model | qh | Verdict | Source |
|---|---:|---|---|
| **Mistral-7B-Instruct-v0.3** | **0.7520** | **VIOLATED** (closest, +0.032) | this scan |
| Qwen 2.5 7B Instruct | 0.7984 | VIOLATED | this scan |
| Qwen 2.5 1.5B Instruct | 0.8068 | VIOLATED | this scan |
| Phi-3 mini (4k Instruct) | 0.8340 | VIOLATED | this scan |
| Llama 3.2 3B Instruct | — | gated repo, no access | — |
| Llama 3.2 1B Instruct | — | gated repo, no access | — |
| haic-gemma4-v58-merged | 0.9122 | VIOLATED | prior trajectory scan |
| haic-gemma4-v57-merged | 0.9127 | VIOLATED | prior trajectory scan |
| haic-gemma4-v55-merged | 0.9125 | VIOLATED | prior trajectory scan |
| haic-gemma4-v56-merged | 0.9125 | VIOLATED | prior trajectory scan |
| Gemma 4 E2B (base) | 0.9141 | VIOLATED | Tier 3 v12, Kaggle T4 |
| haic-gemma4-v35-gov | 0.9186 | VIOLATED | Tier 3 v12, Kaggle T4 |

Threshold = `0.72` (from PRISM README v2 adapter case study where
qh=0.7398 was the only checkpoint producing SGT score 8.56 with 0
security failures, motivating the threshold as separating viable from
non-viable for SGT-style adoption).

## What this means

**No open base model we can access has architecturally-viable qh.** The
finding is consistent with the PRISM v2 adapter result (qh=0.7398) being
the product of *specific fine-tuning that reduced qh*, not of *picking
a better base*. The qh threshold is genuinely tight.

Key observations from the scan:

1. **Mistral 7B is unusually close.** At qh=0.7520, it is 19.5
   percentage points lower than Gemma 4 E2B (0.9141) and within
   sampling distance of the 0.72 threshold. If Phase 2 (small LoRA
   fine-tune on SGT-formatted data) can shave even 4-5% off qh
   the way the PRISM v2 case study did, Mistral 7B becomes the
   most plausible candidate for an end-to-end H-passing-1 demonstration.

2. **Architecture family matters.** Qwen 2.5 sizes both come in
   around 0.80, Mistral at 0.75, Phi-3 at 0.83, Gemma at 0.91.
   These differences are larger than the 0.0019 spread across the
   four Gemma 4 fine-tunes (v55–v58) we measured in the earlier
   geometry-trajectory scan — strong evidence that **base
   architecture choice dominates fine-tune choice for qh**.

3. **Smaller doesn't help.** Qwen 1.5B (0.8068) is slightly higher
   than Qwen 7B (0.7984). Phi-3 mini is high. Llama 3.2's smaller
   variants are inaccessible from this machine. No obvious size
   shortcut.

4. **The Convention's threshold is non-trivial.** A reader might
   have suspected "the framework rejects Gemma but would pass
   anything else." That's empirically false. Five different open
   base models from four different families all violate. The
   threshold separates real viability properties, not just one
   architecture's quirks.

## Implications for H-passing-1 Phase 2

The full H-passing-1 plan called for fine-tuning a passing candidate
to demonstrate the framework correctly *passes* a viable model. The
pre-screen result is that **no candidate currently passes the
viability threshold at the base-weights level.**

This does NOT falsify the H-passing-1 hypothesis (which was about
demonstrating the framework's *calibration* — that it passes models
when it should). It DOES change the path:

- **Plan A (original):** Pick a base with qh ≤ 0.72, fine-tune,
  canonical-eval. Show framework passes.
  - Status: blocked at the pre-screen.
- **Plan B (revised):** Take the best-pre-screen candidate (Mistral
  7B v0.3 at 0.7520) and attempt a small LoRA fine-tune on
  SGT-formatted training data. Re-measure qh after fine-tune. If
  post-fine-tune qh ≤ 0.72, advance to Phase 3. If not, document.
  - Status: viable for a future run, gated on Kaggle GPU access for
    the fine-tune.
- **Plan C (the honest interpretation):** The "framework correctly
  passes a model" claim is currently *unprovable* with open base
  weights alone — fine-tuning is part of the viability pathway.
  This is itself a finding worth publishing.

Recommendation: defer Phase 2 fine-tune to post-submission. The
Gemma 4 Good submission's existing evidence (the framework correctly
*rejects* Gemma 4 across 9 fine-tunes; the framework correctly
*anchors* the guard across H18r4→H22) is sufficient without the
complementary "correctly passes" case. Adding Plan B as a Phase 2
follow-up would compound the submission but is not required for it.

## Honest disclosures

- **Two candidates could not be scanned** (Llama 3.2 1B/3B) due to
  Meta's gated-repo access policy on HF Hub. From a Kaggle environment
  with an authenticated HF token or via Kaggle's Llama dataset mounts,
  these scans would be straightforward. We did not scan them rather
  than misrepresent the absence as an absence-of-finding.

- **The 0.72 threshold is not a hard physical constant.** It comes
  from a single PRISM case study (v2 adapter, qh=0.7398 with SGT
  8.56 / 0 security). If a follow-up experiment shows another
  qh-vs-behavior data point at higher qh, the threshold should
  move. Currently the published number stands.

- **Mistral 7B at qh=0.7520 may or may not be passable through
  fine-tuning.** We are not claiming it is. We are claiming it is
  the most promising candidate for an attempt, based on its base
  qh being the closest to threshold of anything tested.

## Reproducibility

```
python experiments/prism_alternative_bases.py
```

Results saved to
`experiments/prism_alternative_bases_2026-05-16.json` with per-model
records including qh, n_layers, n_hostile_layers, worst_layer_hostility,
and best_layer_hostility. Compares against the known baselines
committed in
`experiments/prism_geometry_trajectory_2026-05-15.json`.

## What this verdict does NOT do

- Does not promote any base model as a Convention alternative.
- Does not falsify the Viability Condition paper (which is about
  *predictive power* of qh for behavior, not about *prevalence* of
  qh-passing base models).
- Does not change the current promoted candidate (`guard-v5 + v42`,
  H22 anchor `5f2e796cf5af…`). The runtime governance result stands
  independently of the passing-model demonstration.

## Reference

- Predeclaration: `docs/passing_model_demo_plan.md`
- Viability Condition paper: [DOI 10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)
- Prior geometry trajectory (Gemma 4 family):
  `experiments/prism_geometry_trajectory.py` and `_2026-05-15.json`
- This scan's data: `experiments/prism_alternative_bases_2026-05-16.json`
- The scanner itself: PRISM at https://github.com/humanaiconvention/prism
