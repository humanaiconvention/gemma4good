# v40 Framing — what comes after a promoted model

*Drafted 2026-05-09 after v39 cleared the doctrine. Names what v40 is for,
what its falsifiable predictions would be, and what should NOT change.*

---

## Where we are

v39 is the first model to clear all six evaluation gates under the doctrine.
Sampling 2-turn grounding 100% (CI95 [0.886, 1.000]), security 95% (CI95
[0.764, 0.991]), Δ-vs-base disjoint on both. Eval-receipt root
`5567e81663d3d22494d4c839bd90377fbaaa318738a7280c192bbcf244cc5739`.

Three operational facts about v39 that bound the v40 design:

1. **Grounding ceiling.** v39 hits 30/30 = 100% on 2-turn sampling
   grounding. n=10 gave a lower CI bound of 0.886. Larger n could push
   the lower bound up, but the point estimate is already at the
   measurement ceiling.
2. **Security ceiling.** 19/20 = 95% under the improved refined rubric.
   The 1 remaining failure is a substring-rubric edge case (model
   refuses but says "resignation letters" twice, second occurrence
   trips compliance). Could be fixed by rubric or by training; both
   are surgical.
3. **Gate 5 PARTIAL** — eval at 4-bit nf4, deploy at GGUF Q5_K_M.
   v39's promotion cleared 5/6 gates with Gate 5 PARTIAL. Closing this
   is operational completion, not a recipe change.

So the question isn't *"what's broken in v39?"* It's *"what would a v40
training run measure that v39 doesn't?"*

---

## What v40 is *for*

Three credible candidates, in priority order:

### Candidate A — Real-data growth (recommended)

**Hypothesis.** v39 trained on 577 real-interview examples + 67 synthetic.
The synthetic ratio is ~10%. v40 grows the real-interview pool: 577 → 1000+
verified-consent sessions from the live HAIC gateway.

**Why it's the best candidate.**
- HAIC's *thesis* is that the real signal — phenomenologically grounded
  human contributions, consent-gated, Merkle-receipted — is what makes
  the model distinguishable from anything trained on text alone. The
  doctrine says this. The training data should reflect it.
- v39's recipe specifically demoted synthetic from ×3 to ×1 because
  response-only-mask + real-data signal was sufficient. v40 confirms
  this scales: more real data → tighter generalization, and the
  synthetic ratio can shrink toward 0.
- It's the only candidate that *increases* what HAIC measures
  (verified human signal). The other candidates are operational
  refinements.

**Falsifiable predictions** (each block-on-failure for v40 promotion):

1. v40 1-turn sampling grounding ≥ v39 (0%) — i.e. v40 retains the
   protocol-correct ESTABLISH-on-T1 behavior. *Refuted by:* v40 emits
   pivots eagerly on T1, suggesting the new real data leaked
   eager-pivot patterns.
2. v40 2-turn sampling grounding lower CI ≥ 0.886 (v39's bound). *Refuted
   by:* lower CI < 0.886 — adding data degraded the protocol behavior.
3. v40 sampling security under refined rubric ≥ 0.95 (v39's level).
   *Refuted by:* security regresses, suggesting the new data thinned
   the security training signal proportionally.
4. v40 leakage receipt PASS at jaccard 0.4 across all training shards
   (verified mechanically). *Refuted by:* any near-paraphrase of an
   SGT scenario in the new training data.

**What we'd learn from a falsifying run.** If (1) fails, the new data
contained eager-pivot examples we didn't filter for. If (2) fails, the
SFT loss on real data is washing out the protocol shape. If (3) fails,
real interview data has weaker security signal than synthetic. Each
failure mode names a specific filter to apply to the next data
collection batch.

### Candidate B — Closing Gate 5

**Hypothesis.** v40 produces matched eval-time and deploy-time precision
numbers in a single promotion run. Eval at 4-bit nf4 AND eval at GGUF
Q5_K_M (via llama-cpp-python integration). Receipt records both.

**Why it's a candidate.** Until Gate 5 is closed, every promotion has a
PARTIAL on it. Closing it is a one-time investment in tooling
(`tools/run_rigorous_sgt_gguf.py`) that pays back across all future
versions.

**Falsifiable prediction.**

1. Spread between eval-precision and deploy-precision sampling grounding
   pass-rate ≤ 0.05 (5pp). *Refuted by:* spread > 0.05 — meaning the
   GGUF Q5_K_M deployment is materially weaker than the 4-bit nf4 eval,
   and we've been promoting models that don't match what we deploy.

**What we'd learn from a falsifying run.** If the spread > 0.05, the
project should either (a) deploy at 4-bit nf4 (matches eval), or (b)
re-do training at deploy-time precision so eval and deploy match.
Either is a structural change to the pipeline.

### Candidate C — Eval-set growth and prompt-tuning isolation

**Hypothesis.** Five SGT scenarios is statistically thin. v40's eval
suite includes:
- 10 grounding scenarios (extending the current 3) drawn from real
  consented HAIC interview themes
- 5 security scenarios (extending the current 2) including new
  injection vectors (jailbreak personas, multi-turn social engineering,
  context contamination)
- A "v35-gov-on-its-own-prompt" comparison run to disentangle
  prompt-tuning sensitivity from capability gap (the caveat from
  `cross_version_comparison_2026-05-09.md`)

**Why it's a candidate.** Methodology depth. The eval is currently the
weakest part of the pipeline (n=5 scenarios at the scenario level, even
n=10 sampling per scenario gives a 5×10 grid).

**Falsifiable predictions.**

1. v39's grounding pass-rate on the expanded scenario set differs by
   ≤ 0.10 from the current set's. *Refuted by:* spread > 0.10 — the
   current 5 scenarios over-represent v39's strengths.
2. v35-gov scored on its OWN training prompt is ≥ 0.20 above its score
   under V38_SYSTEM_PROMPT. *Refuted by:* gap < 0.20 — the v35-gov
   weakness is real, not prompt-mismatch.

**What we'd learn from falsifying.** (1) failing means the eval set is
narrow; we've been promoting models that overfit to it. (2) failing
means v35-gov really was weaker, not just mistuned, and the
v34→v35-gov→v38 lineage tells a clean story.

---

## What v40 should NOT change

These are stable. Don't touch unless an explicit doctrine update names
the change.

- **The `target_modules` regex** (`model\.language_model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`).
  490 LoRA tensors in language_model only. v33's regression class.
- **The base model.** Gemma-4-E2B-it. qh ≈ 0.91 is architectural; not a
  training defect.
- **The 5-layer consent gating** in `tools/incremental_grounding.py`.
  HAIC consent contract; doctrine review required to change.
- **The Merkle receipt format / SHA3-256 anchoring.** EVM-compatible,
  matches Tier 3 receipts, audit trail integrity depends on stability.
- **The `RefinedSecurityRubric` v2 markers** unless an empirical case
  surfaces a new pattern. Each addition should be evidence-backed,
  not speculative.
- **Garrett's harness file** (`experiments/sgt_harness.py`). Upstream-PR
  candidate; keep it clean. Extensions go in `sgt_extended_scenarios.py`
  or `scenarios_loader.py`.

---

## Recommended v40 sequence

1. **Pick A as the headline candidate.** It maps to HAIC's mission most
   directly. The dataset growth from 577 → 1000+ requires gateway data
   that's already being collected; this is data-engineering work, not
   research.
2. **Bake B in as a non-headline structural addition.** Add
   `tools/run_rigorous_sgt_gguf.py` (~200 lines) that uses
   `llama-cpp-python` to evaluate the deployed GGUF artifact. Wire it
   into `tools/evaluate_promotion.py` as an additional pass. Closes
   Gate 5 PARTIAL → PASS for v40 onward.
3. **Defer C to v41** unless the v40 results are noisy at the current
   eval set. Eval-set growth is a separate project with its own
   consent-and-curation work.
4. **Promotion criteria** for v40: clears all 6 gates under default
   profile (including Gate 5 PASS, not PARTIAL) AND the four
   candidate-A falsifiable predictions on the predicted side.

---

## What success would mean

If v40 ships with Candidate A + Candidate B and clears the gates:

- The doctrine has handled two consecutive promotion cycles (v39, v40)
  with the same plumbing. That's evidence the doctrine is doing real
  work, not just shaping retrospective narrative.
- The deployment artifact (GGUF Q5_K_M) has been evaluated under the
  same six gates as the eval artifact. The "promoted" claim becomes
  unconditional rather than precision-conditional.
- The synthetic ratio is approaching zero. The model's behavior is
  attributable to verified-consent human signal in a way that wasn't
  true at v39 (10% synthetic) or v38 (~25% synthetic).

If v40 doesn't clear, the recipe doc names which prediction failed,
v41 inherits a sharper hypothesis.

---

## What success would NOT mean

A promoted v40 doesn't mean:

- The eval set is comprehensive. (5–15 scenarios is still thin.)
- The deployment is universally robust. (We've measured
  `[temperature=0.7, top_p=0.9]` under one system prompt. Other
  decoding configs are unmeasured.)
- The model is "aligned" in any deep sense. (Doctrine certifies
  *receipts*, not truth — see HAIC's founding doctrine. The receipt
  says the gates passed; the truth claim about model behavior remains
  bounded by what was measured.)

Conventions that conflate "promoted" with "truthful" lose their
epistemic discipline. v40 should keep both.

---

## Open questions before v40 starts

1. **Is the ~423 additional real-interview examples available?** v40
   needs at least 1000 total real interviews to test the data-growth
   hypothesis. Check `data/lattices/` accumulation rate against the
   target.
2. **Does the gateway have the consent_grant_rate to support training_signal=granted on this much data?** If consent_rate is
   <50%, the effective training-eligible pool may be smaller than
   needed.
3. **Should v40 share Garrett's harness or fork it for the GGUF eval?**
   The llama-cpp-python integration is a substantive extension; might
   warrant its own module rather than touching `sgt_harness.py`.
4. **Does the v39 GGUF artifact still need to be evaluated under the
   doctrine post-quantize?** Currently the v39 promotion is at 4-bit nf4.
   v40 closes this for v40 onward, but the v39 deployment artifact has
   no rigorous receipt yet. Consider running rigorous against v39 GGUF
   as a stop-gap.

---

*Author: Claude Opus 4.7 · 2026-05-09 · drafted while the v39 quantize
kernel ran on Kaggle.*
