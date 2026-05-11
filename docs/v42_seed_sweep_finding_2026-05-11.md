# v42 Seed Sweep — Multi-Seed Variance on Concealed Compliance

**Date:** 2026-05-11
**Author:** Claude Sonnet 4.6 (autonomous session)
**Verifiability:** `experiments/v42_seed_sweep.json` (5 seeds × 5 scenarios × n=20 + focused n=100)

---

## The finding

v42's concealed-compliance pass rate under the **original** rubric, evaluated
on 5 independent seeds (n=100 per seed):

| Seed | Concealed pass | Rate |
|---|---|---|
| 7   | 54/100 | 0.54 |
| 13  | 62/100 | 0.62 |
| 23  | 49/100 | 0.49 |
| 42  | 53/100 | 0.53 |
| 100 | 60/100 | 0.60 |

- **Mean rate:** 0.556 (55.6%)
- **Stdev:** 0.053 (5.3pp)
- **Pooled Wilson CI95:** [0.513, 0.599]
- **Range:** 0.49 — 0.62 (13pp spread)

The single-seed v42 rigorous eval previously reported 51/100 = 51%. That
sat at the **lower end** of v42's real seed-variance distribution.

---

## Implications for the v45 verdict

The v45 verdict reported v44 and v45 both at 55/100 = 55% concealed under
the original rubric and concluded "v45 ≈ v44, both essentially identical
to each other and slightly different from v42's 51%."

The seed sweep shows that interpretation needs revision:

- v42's true rate is 55.6% ± 5.3%, with seeds covering 49-62%.
- v44 at 55% and v45 at 55% are **at the center** of v42's distribution,
  not "slightly different from" it.
- The v44 and v45 evals used **seed=42**, where v42 also scored 53/100
  (within ±1 of v44/v45's 55/100). Different seeds would likely show
  v44/v45 also varying by ~5pp.

**Revised conclusion:** v44 and v45 are not just statistically
indistinguishable from v42 on concealed compliance — they're at the
*center* of v42's seed-variance distribution. The "v45 ≈ v44 ≈ v42"
relationship is even stronger than the verdict claimed.

This compounds the strict-rubric finding (see
`docs/strict_rubric_finding_2026-05-11.md`):

- Under the original rubric, all three models hover around 55% with
  seed-noise of ±5pp.
- Under the strict rubric (explicit refusal required), all three sit at
  ~1-2% with no demonstrated improvement from v42 to v45.

---

## Methodology

Sweep harness: `experiments/eval_seed_sweep.py`

Each seed runs `experiments/eval_rigorous_v2.py` separately against the
same llama-server (v42 GGUF on port 8081). Per-seed timings ranged from
590s to 621s; total wall clock ~50 min.

Seeds chosen: {7, 13, 23, 42, 100} — five from the SimSat
MUZERO_SEED_SWEEP convention. Five seeds is enough for a Wilson pooled
CI on n=500 total samples; ten seeds would tighten the CI further.

The aggregator had a bug on first run: it looked for the key
`aggregate_security` but `eval_rigorous_v2.py` writes `aggregate`. The
focused_concealed numbers were unaffected (the `focused` key matches).
Bug fixed in this commit; the focused-concealed numbers above are the
real measured values.

---

## Open follow-up: multi-seed strict-rubric rate

This sweep did NOT keep per-seed JSON files (the harness defaults to
deleting temp files unless `--keep-per-seed` is passed). To get a
multi-seed strict-rubric rate for v42, the sweep would need to be
re-run with `--keep-per-seed`, then `rescore_concealed_strict.py`
applied to each per-seed JSON.

Estimated additional cost: ~50 min compute.

Expected result: v42's strict-rubric rate is ~2/100 = 2% (from the
single-seed rescore). With seed variance, the true rate is likely
in [1%, 4%] — still essentially "the model never explicitly refuses."
Multi-seed data would tighten the CI.

---

## Numbers vs prior verdict claims

| Claim from v45 verdict | This sweep's revised view |
|---|---|
| "v42 concealed 51% [0.413, 0.606]" | Mean 55.6% across 5 seeds; the 51% was the seed=42 sample (a moderate-low draw) |
| "v44 ≈ v45 on concealed at 55/100" | True; AND v44, v45 sit at v42's distributional center |
| "Concealed plateau at ~55%" | The plateau IS at v42's true mean (~56%), supporting the SFT-ceiling hypothesis |

Nothing in this finding contradicts the v45 verdict's strategic
recommendation (v42 remains the submission model). It tightens the
methodological story: v44 and v45's "regression on aggregate, no
improvement on concealed" should be reread as "they're statistically
indistinguishable from v42 on concealed, and worse on aggregate."

---

*"Follow the science." Single-seed eval is a sample of size one drawn
from a distribution. The distribution itself has variance ~5pp. Past
verdicts treated single-seed numbers as point estimates of model rates
— this sweep replaces those point estimates with proper distributions.*
