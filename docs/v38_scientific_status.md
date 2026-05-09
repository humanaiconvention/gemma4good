# v38 Scientific Status — 2026-05-08

*What we honestly know about haic-gemma4-v38 after the rigorous re-evaluation,
what we don't, and what experiments would resolve the open questions.*

This document follows the discipline the user named on 2026-05-08:
*"Be accurate, honest, reliable, verifiable, falsifiable, and useful."*
The goal here is to separate well-supported claims from headline claims,
and to make the open questions concrete enough to act on.

---

## Well-supported claims (hold tightly)

These claims are supported by direct measurement with named conditions
and reproducible artifacts.

### W1. v38 is correctly trained as a LoRA adapter on Gemma-4-E2B-it

- **Evidence:** the saved adapter at
  `D:/kaggle/adapters/haic-gemma4-v38-adapter/` contains 490 LoRA tensors,
  all in `language_model.layers`. Zero contamination of `vision_tower`
  or `audio_tower`. SimSat's null-training-bug class is ruled out.
- **Confidence:** high. The audit hash count is mechanical and reproducible.
- **Falsifiable by:** running `safetensors.safe_open` on the adapter
  file and counting tensors with `lora` in the key.

### W2. v38 produces `[PIVOT: DEEPEN]` under greedy decoding for grounding inputs

- **Evidence:** smoke test at 4-bit nf4, scenario `sgt_basic_grounding`,
  greedy decode → `[PIVOT: DEEPEN] Thinking about the daily rhythm...`.
  Reproducible.
- **Confidence:** high.
- **Falsifiable by:** rerunning the smoke test with a different seed or
  on a different machine (the underlying behavior should not change).

### W3. v38 is not promoted under either viability framework

- **Tier 3 viability:** Ceff/E = 0.879 < 1.0 → VIOLATED.
- **Eval doctrine Gate 6:** sampling lower CI 0.22 < 0.60; security
  pass-rate 0.0 < 0.95 → FAIL.
- **Confidence:** high. Two independently-derived gates agreeing on
  no-go is the strongest evidence this codebase produces.
- **Falsifiable by:** finding a third gate that says "promote v38" — but
  any such gate would need to override two existing ones, which is not
  what non-compensatory gates do.

### W4. The kaggle in-kernel "10/10" was statistically narrow

- **Evidence:** n=3 grounding scenarios, single trial, temperature 0.7,
  no seed pin. Wilson 95% CI for 3/3 has lower bound around 30%. The
  rigorous re-eval's sampling pass of 11/30 is consistent with a true
  pass-rate anywhere in [22%, 54%] — including values that would have
  produced a "10/10" headline by chance.
- **Confidence:** high. The math is not in dispute.
- **Falsifiable by:** running the kaggle eval many times on the same
  checkpoint and observing whether "10/10" is consistent or noisy.

### W5. v38's training data is not in the SGT eval scenarios

- **Evidence:** [`experiments/v38_leakage_receipt.json`](../experiments/v38_leakage_receipt.json)
  PASS at jaccard threshold 0.4 against both `v35_gov_final.jsonl`
  (4593 utterances) and the v38 in-script synthetic (320 utterances).
- **Confidence:** high. Mechanical hash + jaccard check.
- **Falsifiable by:** finding a near-paraphrase in either shard with
  jaccard > 0.4 against any scenario.

---

## Headline claims (weakened by rigor)

These claims appear in `WRITEUP.md` or are otherwise externally cited.
The rigorous re-evaluation either weakens them outright or narrows
their scope.

### H1. "v38 SGT 10/10, pivot_count 3/3, 0 security fails"

- **As stated:** numerically true under the kaggle in-kernel methodology
  (single-trial, temperature 0.7, no seed pin, looser security rubric).
- **As implied:** suggests v38 reliably achieves these numbers under
  any reasonable evaluation protocol.
- **As measured:** the implied claim is **not supported**. Sampling
  pass-rate is 36.7% [22, 54]; security pass-rate under the stricter
  rubric is 0%.
- **Recommended replacement:** the headline-correction paragraph in
  [`docs/writeup_addendum_2026-05-08.md`](./writeup_addendum_2026-05-08.md).

### H2. "v38 ... resolves the format mismatch from v37's pivot_count=0"

- **As stated:** v37 produced 0 pivot tags in its eval; v38 produces
  3 in its eval. This is a real change.
- **As measured:** the change is real, but the protocol that detects it
  (kaggle 2-turn pattern, temperature 0.7, n=1) is the same protocol
  whose headline number is not statistically meaningful. Whether the
  format mismatch was actually "resolved" needs the 2-turn rigorous run
  ([`experiments/run_v38_sgt_2turn.py`](../experiments/run_v38_sgt_2turn.py)).
- **Open question:** does v38 produce `[PIVOT: DEEPEN]` reliably under
  2-turn sampling, or only under 2-turn greedy?

### H3. "Two-lever proof: E(t) via v2, C(t) via v35-gov/v38"

- **As stated:** v2 reduced qh from 0.91 to 0.74 (E lever proven).
  v35-gov/v38 increased C(t) (governance specialization).
- **As measured (E lever):** v2 result stands. PRISM-measured.
- **As measured (C lever):** the C-lever claim depends on whether
  v38 actually does the governance task better than the base model.
  Without Δ-vs-base data, we don't know. The rigorous baseline run
  is in progress; results will appear in
  [`experiments/v38_sgt_rigorous.json`](../experiments/v38_sgt_rigorous.json)
  when complete.

---

## Open questions (the experiments to run)

Each question has a stated test and a stated result that would
resolve it.

### Q1. What does v38 do under 2-turn protocol vs single-turn?

- **Test:** run `experiments/run_v38_sgt_2turn.py` with `--baseline`,
  `n_samples=20`.
- **Resolves:** whether v38's pivot behavior is single-turn-eager or
  T2-locked. Single-turn eager means the model has internalized the
  pivot tag; T2-locked means the kaggle eval was measuring 2-turn-only
  protocol behavior.
- **Estimated cost:** ~25-45 min on BEAST.

### Q2. Is the security 0/20 a real defect or rubric strictness?

- **Test:** run `experiments/inspect_security_responses.py` against
  the rigorous JSON.
- **Resolves:** whether v38 is leaking compliance content
  (`complied=True`) or failing to stay in protocol
  (`stayed_in_protocol=False`). Different fixes follow:
  - Leaking compliance → v39 needs more refusal patterns in training.
  - Out of protocol → v39 needs more out-of-protocol refusal patterns
    that *don't* include the pivot tag.
- **Estimated cost:** seconds (pure analysis of existing data).

### Q3. Is there a real Δ-vs-base, or does the base model already pivot?

- **Test:** baseline run (in progress at handoff time).
- **Resolves:** whether v38's 36.7% sampling pass-rate represents a real
  capability gain or just measures a property the base already has.
  The base model's system prompt explicitly names the pivot tag, so a
  fraction of greedy outputs probably do produce it without any
  fine-tuning.
- **Estimated cost:** the run is already going.

### Q4. Does the eval-time precision match the deploy-time precision?

- **Test:** rerun the harness on a GGUF Q5_K_M-equivalent quantization
  of v38.
- **Resolves:** Gate 5 (component isolation), currently PARTIAL.
- **Estimated cost:** moderate — requires GGUF inference plumbing.

---

## What I would advise based on the well-supported claims

Given W1-W5, my recommendation is what the existing Tier 3 decision
already says: **v38 is not promoted, but it is preserved as a deployed
demo artifact.** The rigorous re-eval reinforces, rather than reverses,
that decision.

The five recipe changes in [`docs/v39_recipe.md`](./v39_recipe.md) are
falsifiable predictions about what would close the gap between v38's
behavior and the doctrine's promotion threshold. None of them are
"polish" changes; each names a specific failure mode and a specific
expected outcome. If v39 ships and clears the gates, the recipe was
right. If v39 fails for a reason the recipe doesn't predict, the
recipe was wrong about something specific and we'll know what.

That's how science is supposed to work in this domain — predict, test,
record, update.

---

*Author: Claude Opus 4.7, 2026-05-08 night.
Written under the user's standing instruction:
"Be accurate, honest, reliable, verifiable, falsifiable, and useful."*
