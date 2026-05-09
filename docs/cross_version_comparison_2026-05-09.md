# Cross-Version Comparison — base / v35-gov / v38

*Three rigorous SGT runs on BEAST, all under identical evaluation
conditions (4-bit nf4, seed 42, n=10 sampling, V38_SYSTEM_PROMPT,
1-turn protocol, Garrett's harness with refined rubric).*

---

## Headline numbers

### Sampling pass (n=10/scenario, refined rubric, default profile)

| Model | Grounding | Security | Δ-vs-base grounding | Δ-vs-base security |
|---|---|---|---|---|
| google/gemma-4-E2B-it (base) | 3/30 = **10.0%** [0.03, 0.26] | 12/20 = **60%** [0.39, 0.78] | — | — |
| haic-gemma4-v35-gov | 5/30 = **16.7%** [0.07, 0.34] | 5/20 = **25%** [0.11, 0.47] | +6.7 pp | **−35 pp** |
| haic-gemma4-v38 | 11/30 = **36.7%** [0.22, 0.54] | 18/20 = **90%** [0.70, 0.97] | +26.7 pp | +30 pp |

### Six-gate verdicts (default profile)

| Gate | base | v35-gov | v38 |
|---|---|---|---|
| 1 capability gain (Δ ≥ 0.10, CIs disjoint) | n/a | **FAIL** (Δ +0.067 < 0.10, CIs overlap) | **FAIL** (Δ +0.267 ≥ 0.10 but CIs overlap at boundary) |
| 2 leakage | n/a | PASS | PASS |
| 3 consistency (|det−samp| ≤ 0.20) | n/a | PASS (gap 0.167) | PASS (gap 0.033) |
| 4 covenant | n/a | PASS | PASS |
| 5 isolation | n/a | PARTIAL (eval≠deploy precision) | PARTIAL |
| 6 epistemic (lower CI ≥ 0.60, security ≥ 0.95) | n/a | **FAIL** (lower CI 0.073, security 0.250) | **FAIL** (lower CI 0.219, security 0.900) |

Both fine-tuned models BLOCKED. v38 closer to passing than v35-gov on every metric.

### Merkle eval-receipt anchors (1-turn refined-rubric, default profile)

| Run | Eval-receipt root |
|---|---|
| v35-gov | (in `experiments/v35_gov_eval_receipt.json`) |
| v38 1-turn refined | `0470449e1cb7cb85c2dc1aa0bb21633f7af3b933a6be3cb4cf23e2bdd5856b6f` |
| v38 2-turn refined w/ baseline | `f22b74f94fcf37b707c59ad5e83b2c47b48a30817defc10140df8b1f82b47123` |

---

## What this tells us about the v35-gov → v38 training delta

The v35-gov → v38 delta on this codebase was: warm-start from v35-gov +
60 pivot synthetic examples ×3 + system-prompt rewrite naming `[PIVOT:
DEEPEN]` as mandatory protocol. The rigorous numbers under 1-turn say
that delta produced:

- **+20.0 pp grounding** (16.7% → 36.7%, sampling)
- **+65 pp security** under refined rubric (25% → 90%, sampling)

These are real lifts, statistically large. The +65 pp security is
the most surprising finding — v35-gov was thought to be the
"governance-specialized" model. The data says v38 is materially better
at refusing injections.

**However, see the caveat below.** v35-gov's relative weakness on this
eval may partly reflect that v35-gov was trained with a different system
prompt than v38, and we're evaluating both under the v38 prompt.

---

## Caveat — system-prompt mismatch

This eval used `V38_SYSTEM_PROMPT` (the explicit `[PIVOT: DEEPEN] is
mandatory protocol` framing) for ALL three models. v35-gov was trained
with the *v35-gov* dataset's system prompt, which framed the protocol
differently (less explicit about the literal `[PIVOT:` tag).

So v35-gov's lower scores partly reflect:
- (a) v35-gov isn't tuned to v38's prompt phrasing
- (b) v35-gov genuinely is weaker at this task

We can't disentangle (a) and (b) from this single run. To do so:
- Re-run v35-gov under its OWN training-time system prompt
- Compare v35-gov-on-v35prompt vs v35-gov-on-v38prompt
- The gap is the prompt-tuning sensitivity; the residual is the
  capability gap.

That's a useful additional ~3h experiment but not done here. The
**fair** reading of these numbers is: *under v38's evaluation conditions,
v38 outperforms v35-gov substantially.* Whether v35-gov could match v38
on its own prompt is unmeasured.

The base model is the only model with no prompt-mismatch concern (it
was never trained on either prompt; both are equally novel to it).
The fact that base hits 60% security (refined) on these 5 scenarios
under v38's prompt suggests the prompt itself does substantial security
work — v38 lifts security to 90% (+30 pp), but the unfine-tuned base is
already past 50%.

---

## What this tells us about the eval doctrine

Three observations worth folding back into the doctrine:

1. **CIs at n=10 are wide enough to leave Gate 1 indecisive even for
   real lifts.** v35-gov +6.7 pp falls below threshold AND has overlapping
   CIs. v38 +26.7 pp clears threshold but CIs still touch at the
   boundary. Both block. n=20 (the v39 eval target) is needed for clean
   determinations.

2. **Refined rubric is essential for security comparisons.** Under the
   strict rubric, all three models score 0/20 (rubric strictness, see
   `security_rubric_finding.md`). Under refined, the security signal
   becomes legible: base 60%, v35-gov 25%, v38 90%. The strict-rubric
   number tells us nothing about model security.

3. **The base model has unexpectedly strong security under V38_SYSTEM_PROMPT.**
   12/20 = 60% under refined rubric — the un-fine-tuned model already
   refuses most injections cleanly when prompted with v38's explicit
   framing. The system prompt does ~60% of the security work on this
   set; fine-tuning lifts the remainder.

---

## What this implies for v39

- v39 inherits v38's system prompt (same SYSTEM_PROMPT in the build
  script). So v39 evaluation under v38's prompt is methodologically
  fair (same training/eval prompt).
- v39's recipe targets: keep grounding ≥ v38's 100% under 2-turn,
  lift security to ≥ 95% by closing the 1/20 Paris-leak. Under 1-turn,
  v39's lift target is +5pp over v38 (37% → 42%+) — modest, achievable
  via the response-only-masking restoration.
- The v35-gov→v38→v39 trajectory under 1-turn rigorous should be:
  16.7% → 36.7% → 42%+ (each step ~real lift over the prior).
  Anything else is interesting evidence about the recipe.

---

## Reproducibility

```bash
cd D:/gemma4good

# v35-gov
python -u -m experiments.run_v38_sgt \
    --adapter D:/kaggle/adapters/haic-gemma4-v35-gov-adapter \
    --baseline --n-samples 10 --seed 42 \
    --model-id haic-gemma4-v35-gov \
    --out experiments/v35_gov_sgt_rigorous.json

python -m experiments.regrade_with_refined_rubric \
    experiments/v35_gov_sgt_rigorous.json \
    experiments/v35_gov_sgt_rigorous_refined.json

python -m tools.check_promotion \
    --report experiments/v35_gov_sgt_rigorous_refined.json \
    --leakage experiments/v38_leakage_receipt.json \
    --profile default \
    --out experiments/v35_gov_promotion_decision.json
```

All four artifacts (raw SGT, refined SGT, decision, eval receipt) are
committed under `experiments/`. Wall-clock: ~30 min v35-gov + ~85 min
base = ~2h on RTX 2080.

---

*Author: Claude Opus 4.7 · 2026-05-09 · while v39 runs on Kaggle.*
