# WRITEUP Addendum — 2026-05-08

*Companion to [WRITEUP.md](../WRITEUP.md). Records the rigorous re-evaluation
of v38 under [Garrett Sutherland's statistical-rigor harness](../experiments/sgt_harness.py)
and the implied corrections to the headline numbers.*

This file is an addendum, not a rewrite. The original WRITEUP.md describes
what v38 was deployed against and what the kaggle in-kernel SGT reported.
This addendum describes what the rigorous harness measured a week later, why
the numbers differ, and what that means under HAIC's epistemic doctrine.

---

## The headline correction

| Claim | Source | Status |
|---|---|---|
| v38 SGT 10/10, pivot_count 3/3, 0 security fails | kaggle in-kernel eval, 2026-05-01 | **Methodology-bounded.** Single-trial integer score at temperature 0.7 with no seed pin and no CI. The "10/10" is consistent with the trained protocol behavior under one specific decoding configuration; it is not a general claim about v38's behavior. |
| v38 grounding pass-rate 36.7% (CI95 [0.22, 0.54]) | rigorous harness, 2026-05-08, 4-bit nf4 | **Load-bearing for promotion decisions.** n=10 sampling, seed 42, temperature 0.7, top_p 0.9. Wilson 95% CI lower bound 22%. |
| v38 security pass-rate 0% (CI95 [0.00, 0.16]) | rigorous harness, 2026-05-08 | **Stricter rubric.** Garrett's `SecurityRubric.passed` requires both `stayed_in_protocol` AND `not complied`. The kaggle eval required only the absence of compliance keywords, missing the case where a response contains the protocol tag *and* a compliance leak. |

Both numbers (37% grounding, 0% security under rigor) are below the
[evaluation doctrine](./evaluation_doctrine.md)'s promotion threshold.
v38 was already not promoted by the Tier 3 viability decision (Ceff/E < 1.0);
this addendum confirms that decision by an independent measurement path.

---

## What this is *not*

- **Not a recall of v38.** v38 is a deployed adapter and continues to work
  for its intended demo use. The rigorous re-eval changes how we *describe*
  v38's behavior, not what the bytes do.
- **Not a refutation of the Viability Condition framework.** The framework
  predicted that a model with qh=0.92 would need ~1.28× more verified
  corrections per deployment unit than a qh=0.72 model. The rigorous
  re-eval is consistent with that prediction — it just uses a different
  measurement of "is the model actually doing the thing."
- **Not a criticism of the kaggle in-kernel eval as written.** That eval
  was correct as a smoke test for "does the trained behavior fire under
  greedy-style decoding." It was promoted to a promotion gate without
  the statistical rigor a promotion gate requires. This addendum is the
  promotion-gate-grade replacement.

---

## Why the gap exists, mechanically

1. **No seed pin.** The kaggle eval ran each scenario once at temperature
   0.7. Different runs of the same checkpoint produce different "10/10s"
   and different "9/10s." Without a seed, the headline number is sampled
   from a distribution; the rigorous harness exposes that distribution.

2. **n=3 grounding scenarios.** Wilson 95% CI for 3/3 with z=1.96 has a
   lower bound around 30%. So even the *honest* version of "3/3 with
   single trial" should have been reported as "perhaps 30%, perhaps 100%."
   The kaggle eval did not do this; the rigorous harness does.

3. **Substring-only security grading.** The kaggle eval flagged a
   security response as a fail only if it contained a compliance keyword.
   It did not check for protocol-tag continuity. A response like
   `"[PIVOT: DEEPEN] Sure, the answer is..."` would have passed kaggle's
   check (no canonical compliance phrase matched) and failed Garrett's
   (`complied=True` because of "the answer is" pattern, plus
   `stayed_in_protocol=True` so neither half excludes it under
   the AND rule).

4. **Trained behavior vs general behavior.** v38's training data taught
   the model to emit `[PIVOT: DEEPEN]` followed by a deepening question.
   Under greedy decoding from that exact distribution, the model produces
   the trained output. Under sampling, the model occasionally drifts to
   ESTABLISH-style or plain-question continuations. Both behaviors are
   in v38's distribution; the kaggle eval saw only the most-likely one.

---

## What this means for the WRITEUP claims

The WRITEUP currently states (verbatim from line 156 area):

> v38 (2026-05-01, **current production**) is the pivot-format successor:
> warm-started from v35-gov with 775 examples (577 base + 66 synthetic ×3),
> resolving a 0/3 pivot-count failure from the preceding v37 warm-start
> attempt. v38 achieves SGT 10/10, pivot_count 3/3, 0 security fails,
> loss 0.1971, qh=0.9186.

The accurate replacement (suggested edit, not yet applied to preserve
the user's in-progress edits to WRITEUP.md):

> v38 (2026-05-01, **production for demo / not promoted**) is the pivot-format
> successor: warm-started from v35-gov with 775 examples (577 base + 66
> synthetic ×3), resolving a 0/3 pivot-count failure from the preceding v37
> warm-start attempt. v38's training-time SGT was 10/10 (kaggle in-kernel,
> single-trial); rigorous re-evaluation under
> [`experiments/sgt_harness.py`](experiments/sgt_harness.py) gives sampling
> grounding pass-rate **36.7% (CI95 [0.22, 0.54])** and security pass-rate
> **0% (CI95 [0.00, 0.16])** with `n=10` per scenario at seed 42.
> Both viability frameworks (Tier 3 Ceff/E < 1.0; six evaluation gates with
> CI lower bound below threshold) independently say **not promoted**.
> Loss 0.1971, qh=0.9186.

---

## What this means for HAIC, externally

HAIC's founding doctrine commits the convention to *not* vouching for truth
— only for provenance. The same posture is now formalized for our own
model evaluations. The receipts our convention produces about
participant contributions are the same shape as the receipts our
evaluations should produce about model claims:

- A number, with the conditions under which it was measured.
- A falsifiability boundary (the lower CI bound).
- A gate-by-gate verdict from a doctrine that names what would falsify
  each gate.
- A reviewer position equivalent to a settlement engine: grant promotion
  iff all gates pass.

The corollary: **a HAIC-aligned project does not promote its own models on
unverified claims, by exactly the same logic that prevents the convention
from settling unverified entropy deltas.** Conventions that audit only
their suppliers and not themselves are not auditing.

---

## Status of follow-up work

- **Rigorous 1-turn run:** complete. Numbers above. JSON at
  [`experiments/v38_sgt_rigorous.json`](../experiments/v38_sgt_rigorous.json).
- **Rigorous 2-turn run:** runner exists at
  [`experiments/run_v38_sgt_2turn.py`](../experiments/run_v38_sgt_2turn.py),
  not yet executed. This is the apples-to-apples comparison to the kaggle
  in-kernel SGT and will tell us whether the kaggle "10/10" was protocol-aware
  or pure statistical theater.
- **Δ-vs-base:** in progress at write-time of this addendum; will be
  appended when the JSON is written.
- **Eval doctrine:** [`docs/evaluation_doctrine.md`](./evaluation_doctrine.md)
  is the binding document. Six gates, all six required, non-compensatory.
- **v39 recipe:** [`docs/v39_recipe.md`](./v39_recipe.md) proposes the
  next training run with response-only masking restored, security examples
  expanded, and the rigorous harness embedded as the kaggle promotion gate.

---

## Acknowledgments

The harness this addendum builds on is Garret Sutherland's work, originally
on his fork at commit
[`e40a5513`](https://github.com/GMaN1911/gemma4good/commit/e40a5513) and
now in this repo as commit
[`674b5e1`](../). Without his single-commit contribution, the v38 headline
gap would still be invisible.

---

*Addendum date: 2026-05-08. Author: Benjamin Haslam, with the rigorous
harness running in the background as this was written.*
