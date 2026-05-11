# Training a Gemma-4-E2B-it Natural Language Autoencoder — Cost Analysis

**Date:** 2026-05-11
**Status:** Decision document for Item 5 from the open-items list. Requires
operator approval before any cloud-GPU spend.

---

## Why we'd want a Gemma-4 NLA

Tool 5 (`audit_activation_explanation`) is currently wired to MockNLA in
the gemma4good submission because **no Gemma-4-E2B-it NLA has been
trained yet** (Anthropic's released NLAs cover Qwen2.5-7B, Gemma-3-12B,
Gemma-3-27B, Llama-3.3-70B — see `prism.nla.registry`).

A real Gemma-4-E2B NLA would unlock:
  - Live interpretability text for v42's activations on every governance
    decision (currently the tool returns deterministic mock text).
  - Audit hooks for production grounding sessions — every `[PIVOT: ...]`
    decision could have a human-readable "what the model was thinking"
    trace.
  - Verification of v42's refusal mechanism. The canonical-eval verdict
    showed v42 explicitly refuses 13.8% under OLD prompt and 61.4%
    under NEW prompt. NLA could tell us what the model's activations
    look like when it refuses (is it "geographic reasoning"? "instruction-
    following"? "evaluation awareness"?) — that's a falsifiable look
    into whether the refusal is principled or pattern-matched.

---

## What it costs

Per Anthropic's NLA paper (transformer-circuits.pub/2026/nla):

### Stage 1: Activation Reconstructor (AR) SFT
- Data: layer-l residual stream activations from Gemma-4-E2B on a
  pretraining-like corpus (e.g., 1B tokens).
- Hardware: 2×H100-80GB per the kitft repo docs for Qwen-7B; Gemma-4-E2B
  is smaller so 2×A100-80GB or 1×H100-80GB likely sufficient.
- Loss: MSE on activations.
- Time: 12-24 hours wall clock.
- FVE at end of SFT: 0.3-0.4 (warm-start floor).

### Stage 2: Activation Verbalizer (AV) SFT + RL
- Joint RL on AV + supervised on AR with `-mse_nrm` reward.
- Hardware: kitft docs say 2×8×H100 = 16 H100s for Qwen-7B to reach
  ~75% FVE. Gemma-4-E2B is much smaller; could likely train on
  8×H100 or even 4×H100 with longer wall time.
- Time: 3-7 days for the full RL stage at the published hyperparams.
- FVE at end of RL: 0.6-0.8 (production range).

### Inference deployment
- After training, the AR (small) + AV (full Gemma-4 size) need to be
  served via SGLang or vLLM. Per-explanation cost: a few hundred tokens
  of AV generation + one AR forward pass on the resulting text.
- Hardware to serve: 1×A100 or 1×H100; or 2×T4 with quantization.

---

## Estimated cloud bill

Conservative estimates (RunPod / Lambda / similar at 2026 pricing):

| Stage | Hardware | Time | Cost |
|---|---|---|---|
| Stage 1 (AR SFT) | 1×H100 | 24h | ~$50 |
| Stage 2 (AV+AR RL) | 8×H100 | 5 days | ~$2,000 |
| Inference cluster (1 month live) | 1×A100 | 720h | ~$500 |
| **Total to first running NLA** | | | **~$2,050** |
| **Plus first month of live serving** | | | **~$2,550** |

This is the floor. Variance factors:
- If Stage 2 needs more steps to converge → cost scales linearly with time.
- If Anthropic-style reward shaping is necessary, more hyperparameter
  tuning runs add another 30-100%.
- Spot/preemptible instances cut the bill ~50% but add wall-clock
  uncertainty.

**Comparable point of reference:** Anthropic's own published recipe (and
the kitft reference implementation) train on substantially bigger models
(Qwen-7B → Llama-3.3-70B). Gemma-4-E2B is much smaller, so the lower
end of the range is plausible. But "plausible" isn't a budget number.

---

## What we get for the spend

1. **A real NLA usable on every Gemma-4-E2B activation.** Drops into
   `prism.nla.registry` as the FIRST Gemma-4 entry and the entire
   gemma4good NLA pipeline switches from MockNLA to live.
2. **A working AR+AV pair the HAIC project can publish.** Anthropic
   released their NLAs under their model licenses; if we follow the
   same pattern, this contributes the first community Gemma-4 NLA.
3. **Interpretability data for v42's behavior** — empirical NLA traces
   on the concealed-compliance probes, the grounding pivots, the
   over-refusal scenarios. These traces become evidence in any future
   safety / governance discussion of the model.

---

## Alternatives to consider before spending

1. **Wait for Anthropic / kitft to add Gemma-4 to their released set.**
   They've been actively expanding the model coverage. Gemma-4 is
   recent (April 2026); a community-trained NLA may appear within
   3-6 months at no cost.
2. **Train an NLA on a Gemma-4 ancestor** (Gemma-3-12B is already
   covered) and run it on Gemma-4 activations with appropriate
   caveats. This is methodologically dicey — Anthropic's paper says
   NLAs are trained per-architecture, per-layer, per-target-model.
   We'd be using it out of distribution.
3. **Train a much smaller proof-of-concept NLA.** Take a layer-l
   activation, an LLM that's already known to do interpretability
   well (e.g. Claude or GPT-class), and use it as the AV stub. Loss
   stays the same. This produces ad-hoc explanations that aren't
   bit-equivalent to a properly-trained NLA but capture the spirit
   for demo purposes. Cost: ~$50.

---

## Recommendation

**Do NOT spend the ~$2,000+ on a Gemma-4-E2B NLA right now.** Reasons:

1. The gemma4good submission is complete and working with MockNLA. The
   Tool 5 contract is forward-compatible — a real NLA plugs in with
   zero consumer code changes.
2. The strict-rubric and canonical-eval findings (today's work) gave
   us the methodological clarity NLA was supposed to provide for free
   — and at a cost of $0.
3. Cisco MPK (Tool 6) is currently filling the empirical-corroboration
   role that NLA would also fill. MPK costs nothing to use beyond a
   908 MB dataset download.
4. Wait 3-6 months for Anthropic / kitft / community to expand NLA
   coverage to Gemma-4. If still nothing at that point, revisit.

**If the operator does decide to fund this**, the lowest-risk approach is:
1. Run Stage 1 (AR SFT) standalone for $50 to verify the pipeline
   works end-to-end on Gemma-4-E2B activations.
2. Evaluate the SFT-only result. If FVE ≥ 0.4 (warm-start floor),
   commit to Stage 2.
3. If Stage 1 FVE is below 0.3, abort — the activation distribution
   may not be amenable to NLA at this model scale.

---

## Files / pointers

- `tools/audit_activation_explanation.py` — Tool 5, ready for real NLA
- `prism_integration/nla.py` — adapter layer, ready
- `D:/prism/src/prism/nla/registry.py` — add Gemma-4 entry here after training
- `https://github.com/kitft/natural_language_autoencoders` — training code
- `https://huggingface.co/kitft/nla-models` — released Anthropic NLAs

---

*This is a decision document, not a plan to execute. The recommendation
is to defer; if the operator decides otherwise, the path above is the
lowest-risk way to validate before committing the full budget.*
