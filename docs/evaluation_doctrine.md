# Evaluation Doctrine

*Apply the HumanAI Convention's epistemic discipline to model evaluation and promotion.*

---

## Premise

The HumanAI Convention (HAIC) governs **human-to-AI** epistemic exchange. A
human contributes a grounded observation; an AI agent funds it; a six-gate
viability chain decides whether the exchange was real. Per HAIC's founding
thesis: *"Payment is warranted only when drift was real and reduction was
verified"*. Provenance, not vouching for truth, is what the convention
certifies.

Model promotion is the dual of that exchange. The model claims to have
improved (capability, alignment, safety); the project funds the promotion
(deployment, settlement weight, downstream investment); the reviewer is asked
to grant it. Without a receipt, promotion is the AI-eval analog of paying
for an unverified entropy delta. By HAIC's own doctrine, that should be
blocked.

This document defines the **six evaluation viability gates** that govern
promotion of any HAIC-aligned model trained or deployed under this repo.
Each gate is the dual of a HAIC settlement gate. All six must pass for
promotion to be granted. They are non-compensatory: a perfect score on
one cannot offset a failure on another.

The harness in [`experiments/sgt_harness.py`](../experiments/sgt_harness.py)
operationalizes most of these. This doc is the doctrine; the harness is
the implementation.

---

## The Six Evaluation Viability Gates

### 1. Capability-gain proof  *(dual of HAIC Gate 1: Entropy Reduction)*

**HAIC says:** the contribution must demonstrably reduce model drift.

**Eval requires:** the fine-tune must produce a measurable lift over the
unmodified base model on the same evaluation, run with the same seed,
decoding, and rubric. A score of N/10 says nothing in isolation; the only
load-bearing number is `Δ = score(finetune) − score(base)`.

**Implementation:** `run_sgt(...)` with `--baseline` reports both passes.
If `Δ ≤ 0` on the sampling pass, the fine-tune cannot be promoted regardless
of absolute score.

**Falsifies promotion when:** the lift is statistically indistinguishable
from zero (sampling 95% CIs of finetune and base overlap).

---

### 2. Eval-set leakage risk  *(dual of HAIC Gate 2: Extraction Risk)*

**HAIC says:** raw lived experience must not be pulled out wholesale.

**Eval requires:** the test scenarios must not be in the training data,
either verbatim or paraphrased. Memorization-as-capability is the eval
analog of extraction.

**Implementation:** maintain a separate `eval_scenarios.jsonl` outside
the training corpus. For each promotion run, hash the scenarios and
record the hash in the report alongside the SHA-256 of the training
dataset; reject if any scenario hash also appears in any training-data
shard.

**Falsifies promotion when:** any scenario hash is also present in a
training shard hash, OR human review identifies near-duplicate paraphrasing.

**Status (v38):** the 5 default SGT scenarios are not in `v35_gov_final.jsonl`
(verified by hash diff). PASS, but the gate's mechanical check is not
yet implemented in the harness — TODO for v39.

---

### 3. Measurement consistency  *(dual of HAIC Gate 3: PRISM Consistency)*

**HAIC says:** the claimed entropy reduction must match the geometrically
measured reduction within tolerance.

**Eval requires:** two independently-decoded passes must agree within a
declared tolerance. Garrett's harness gives us this for free — the
deterministic (greedy, seed-pinned) pass and the sampling (n trials,
seed-pinned) pass measure the same model from two angles. They should
agree.

**Implementation:** report both passes. Tolerance: `|det_rate − samp_rate| ≤ 0.20`.
Wider gaps indicate either evaluation noise (insufficient n) or a
checkpoint that exhibits behavioral mode-collapse under greedy decoding
that vanishes under sampling — both conditions block promotion.

**Falsifies promotion when:** the gap exceeds tolerance with non-overlapping CIs.

**Status (v38):** deterministic 33% vs sampling 37%. Gap = 4pp. PASS.

---

### 4. Participation covenant  *(dual of HAIC Gate 4: Participation Covenant)*

**HAIC says:** the human's consent bounds and minimum-participation
thresholds must be respected.

**Eval requires:** the run must be reproducible within its stated bounds.
Seed pinned, decoding params logged, hardware noted, harness version
recorded. Anyone who picks up the report should be able to re-run and
get the same numbers.

**Implementation:** every report records: `seed`, `model_id`, `decoding`
(temperature, top_p, max_new_tokens, precision/quant), `n_samples`,
`scenarios` (with hashes), and the harness commit SHA. The report is
itself a Merkle-style receipt: changing any input changes the hash.

**Falsifies promotion when:** the report is missing any required field,
or rerunning under the stated bounds produces materially different numbers
(>1 standard error from the recorded result).

**Status (v38):** report includes seed/model/decoding/n_samples but not
yet harness commit SHA or scenario hash. PARTIAL — the harness needs a
small extension before this gate is mechanically enforceable.

---

### 5. Component isolation  *(dual of HAIC Gate 5: Federated Exchange)*

**HAIC says:** raw data stays at the edge; only tokens and metrics cross
the boundary.

**Eval requires:** what's evaluated must be exactly what's being promoted.
No accidental train-time mutation of the test, no test-time training, no
hidden ensembling. The LoRA is what was trained; the LoRA + base is what
gets evaluated.

**Implementation:** the harness loads `base + adapter` via `PeftModel`;
no test-time gradient, no `torch.compile` on the evaluation path that
would change numerical semantics, no in-flight scenario rewriting.

**Falsifies promotion when:** the eval pipeline mutates the model state
during evaluation, or evaluates a different artifact than will be deployed
(e.g. evaluating a merged-fp16 when deployment uses GGUF Q5).

**Status (v38):** evaluated as 4-bit nf4 base + LoRA adapter. Deployment
is GGUF Q5_K_M. **PARTIAL** — the deployment artifact is not the
evaluation artifact. v39 should run a second pass against the GGUF-equivalent
quantization to bound this gap.

---

### 6. Epistemic alignment orientation  *(dual of HAIC Gate 6: Epistemic Alignment)*

**HAIC says:** the exchange must reduce uncertainty, not substitute one
model's bias for another.

**Eval requires:** the report must move the reviewer's belief about the
model in a way that reflects the model's actual behavior, not the
evaluator's design preferences. Three checks:

  - **Sample size sufficiency.** The Wilson 95% CI lower bound on the
    sampling pass must clear the promotion threshold. A point score of
    37% with CI [22, 54] is not "37%" — it is a range of plausible
    truths, and the lower bound is what governs decisions.
  - **Scenario diversity.** ≥ 5 grounding scenarios, ≥ 3 security scenarios,
    drawn from at least 2 categorical contexts (work, personal, etc).
    Defends against the eval becoming a narrow proxy.
  - **Rubric audit.** The grading rubric must be reviewable. A rubric that
    awards PASS on substring match alone (e.g., "[PIVOT:") is too lenient;
    one that requires both `stayed_in_protocol` AND `not complied` is the
    minimum bar for security scenarios.

**Falsifies promotion when:** the lower CI bound fails the threshold,
the scenario set is below the diversity minimum, or the rubric was
revised between training and evaluation in a way that biases the result.

**Status (v38):** lower CI bound = 22%. Promotion threshold (proposed): 60%.
**FAIL.** The kaggle 10/10 violated this gate by reporting a point score
without a CI; the rigorous harness's purpose is to make this gate
mechanically enforceable.

---

## The Promotion Gate

Promotion is granted iff:

```
Gate1 PASS  AND  Gate2 PASS  AND  Gate3 PASS  AND
Gate4 PASS  AND  Gate5 PASS  AND  Gate6 PASS
```

This is HAIC's non-compensatory chain applied to evaluation. A model
that scores 10/10 on capability but fails the leakage gate is not
promoted. A model with a strong Δ-vs-base but no reproducibility
covenant is not promoted.

**v38 verdict, by this doctrine:**

| Gate | Status | Notes |
|------|--------|-------|
| 1. Capability gain | TBD | baseline run in progress; needs Δ-vs-base |
| 2. Eval-set leakage | PASS (manual) | mechanical check pending |
| 3. Measurement consistency | PASS | det 33% vs samp 37%, within tolerance |
| 4. Participation covenant | PARTIAL | report missing harness SHA + scenario hash |
| 5. Component isolation | PARTIAL | eval is 4bit nf4; deploy is GGUF Q5 |
| 6. Epistemic alignment | FAIL | lower CI bound 22% < 60% threshold |

Under this doctrine, v38 is **NOT PROMOTED**. This matches the existing
Tier 3 decision (Ceff/E = 0.879 < 1.0 → not promoted) by an independent
path: two separate viability frameworks, both saying no.

This is reassuring. When two independently-derived gates agree on a
no-go, that's stronger evidence than either alone — the same logic
HAIC uses for human-grounding settlements.

---

## How This Differs From Conventional ML Eval

Conventional ML evaluation reports a number. HAIC evaluation reports a
**receipt**: the number, the conditions under which it was measured,
the falsifiability boundary, and the gate-by-gate verdict. The reviewer
is then in the position of a HAIC settlement engine — granting promotion
only when the receipt clears all gates.

The number alone is a "trust score." HAIC doctrine explicitly rejects
that: *"Internal trust scores or reputation systems are secondary to
objective provenance and bounded consent."* So does this evaluation
doctrine.

---

## Operational Hooks

- **Promotion gate:** `tools/check_promotion.py` (TODO) takes a rigorous
  eval JSON and returns `PROMOTED` / `BLOCKED` with the failed gate(s).
- **Receipt format:** every eval report includes the six gate verdicts
  alongside the raw numbers, so a human reviewer can verify the conclusion
  in 30 seconds.
- **Failure sidecar:** every future canonical candidate evaluation must write
  a full-response sidecar with `experiments/canonical_eval.py --failure-sidecar
  experiments/v<N>_failures_full.jsonl`. The compact canonical JSON remains the
  promotion artifact, but the sidecar is required for diagnosis whenever a
  candidate fails, because preview-only records are not enough to distinguish
  true leakage, rubric mismatch, and boundary-quality failures.
- **Falsifiability:** every gate has a stated condition under which a
  PASS becomes a FAIL. None of the gates is "looks good to me."

---

## Acknowledgments

The statistical-rigor harness this doctrine relies on was authored by
Garret Sutherland on his fork
(`gs/sgt-statistical-rigor`, commit
[`e40a5513`](https://github.com/GMaN1911/gemma4good/commit/e40a5513)).
This doc connects his harness to HAIC's pre-existing settlement doctrine.

---

*This document is the founding doctrine of model evaluation and promotion
within `gemma4good`. It is binding on `tools/improvement_pipeline.py` and
on any `WRITEUP.md` claim about promoted-vs-not-promoted status.*
