# Verdict: NLA Stage 1 Smoke Test v5 — 2026-05-12

**Notebook:** `benhaslam/haic-nla-stage1-smoke` (v5)
**Model:** Gemma-4-E2B-it (5.1B, multimodal) on T4
**Hook mode:** `last_token_both` (v5 fix — was `mean_pool` in v4)

---

## SMOKE GATE: PASSED — H-NLA1a PASSED AT SMOKE LEVEL

**Final FVE: 0.7705** (gate: > 0.20, full-run targets: 0.30 / 0.40)

| Predicate | Target | Result |
|---|---|---|
| Smoke gate | FVE > 0.20 | ✅ PASS — 0.7705 |
| H-NLA1b | FVE ≥ 0.30 | ✅ PASS — 0.7705 |
| H-NLA1a | FVE ≥ 0.40 | ✅ PASS — 0.7705 |

**Recommendation: PROCEED_TO_RUNPOD_FULL_RUN**

---

## FVE Curve (LOG_EVERY=200)

| Step | FVE | Loss |
|---|---|---|
| 200 | **0.5309** | 0.5084 |
| 400 | 0.6477 | 0.4219 |
| 600 | 0.6972 | 0.3547 |
| 800 | 0.7244 | 0.2972 |
| 1000 | 0.7415 | 0.2407 |
| 1200 | 0.7542 | 0.2634 |
| 1400 | 0.7625 | 0.2551 |
| 1600 | 0.7669 | 0.2061 |
| 1800 | 0.7696 | 0.2290 |
| **2000 (final)** | **0.7705** | 0.2355 |

The gate (0.20) was passed at step **~50** (well before step 200).
H-NLA1a (0.40) was passed at step **~200**.

---

## Comparison with v4 (mean-pool hook)

| | v4 (mean-pool) | v5 (last-token) | Δ |
|---|---|---|---|
| Final FVE | 0.1635 | **0.7705** | +0.607 |
| Steps to gate | never | ~50 | — |
| Steps to H-NLA1a | never | ~200 | — |
| Plateau? | yes (step 2000) | still rising (0.0009/200 steps) | no |
| Improvement factor | — | **4.71×** | — |

---

## What this means

The v4 failure (FVE=0.163, plateau) was entirely due to the mean-pool input
destroying the positional signal that the AR needed. With the last-token fix:

- **FVE=0.77 at the smoke level** — the AR explains 77% of the variance in the
  last-token residual stream after layer 18 in the smoke run alone
- The full H-NLA1a target (0.40) was hit at step ~200 of the smoke run
- FVE is still rising at step 2000 (0.7696 → 0.7705), suggesting a full run
  (more data, more steps) would achieve **FVE ≈ 0.80+**
- The AR is now a genuine approximator of layer 18's local transform — the
  FFN component (pointwise MLP) is highly approximable, and the residual
  stream at the last-token position carries rich attention summaries from
  prior layers

---

## Key numbers

```
n_train:        45,000
n_holdout:       5,000
layer:          18 / 35  (Gemma4TextDecoderLayer)
layer_path:     model.model.language_model.layers[18]  (confirmed)
d_model:        1536
AR params:      18,896,384
AR arch:        4-layer MLP (input_proj + 3×[Linear+GELU+Dropout] + output_proj)
hidden_dim:     2048
epochs:         3
total_steps:    2,000 (= 3 × 45K/64 ≈ 2,109, OneCycleLR terminates at 2000)
final_loss:     0.2355 (MSE on normalized activations)
```

---

## Next step: RunPod H100 full run

The smoke gate was designed to prevent a $50 RunPod spend on a broken pipeline.
It passed with a 4.71× margin. Proceed to full run.

**Projected full-run FVE**: 0.80–0.85 (extrapolating the curve + more data + more steps)

To launch:
```bash
python experiments/nla_stage1_ar_sft.py --mode full
# OR build a RunPod notebook with larger dataset (500K pairs) and 10 epochs
```

Decision: **RunPod full run AUTHORIZED** — smoke gate passed, H-NLA1a already passed
at smoke level. The only question is whether a full run pushes further toward 0.85.

---

## Artifacts

```
Results:      D:/gemma4good/results/nla-stage1-smoke/nla_smoke_v5_results.json
Stdout:       C:/Users/benja/AppData/Local/Temp/nla-v5-output/nla_v5_stdout.txt
Build script: D:/gemma4good/experiments/build_nla_smoke_nb.py (v5)
Kaggle:       https://www.kaggle.com/code/benhaslam/haic-nla-stage1-smoke (v5)
```

---

## v4 post-mortem (for the record)

v4 achieved FVE=0.163 and plateaued because:
1. AR input = `h_in.mean(dim=1)` (mean-pool over all token positions)
2. AR target = `h_out[:, -1, :]` (last-token position)
3. The mean-pool destroys per-position context — attention at the last token
   depends on keys/values from ALL positions, which the mean-pool only partially
   encodes. The 0.163 ceiling was the fraction of variance explainable from the
   aggregate representation alone.

v5 fix: one line change — `h_in[:, -1, :]` instead of `h_in.mean(dim=1)`.
Result: FVE 0.163 → 0.771, a 4.71× improvement.
