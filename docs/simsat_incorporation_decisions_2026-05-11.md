# SimSat → Gemma4Good Incorporation Decisions

**Date:** 2026-05-11
**Status:** Final inventory after the overnight integration pass

This document records what was incorporated from the HumanAI Convention SimSat
project into the Gemma4Good submission, what was deliberately left out, and
why. Useful for the next person who looks at SimSat and wonders why some
pieces show up here and others don't.

---

## Incorporated tonight

| SimSat artifact | Gemma4Good landing |
|---|---|
| `evaluate_ttt_viability` (3 gates) | `viability/ttt_gates.py` + `tools/edge_ttt_adapter.py` |
| `evaluate_viability` (6 convention gates) | `viability/session_gates.py` |
| DiLoCo lab orchestration patterns | `tools/diloco_fragment_verifier.py` + `viability/distributed_viability.py` |
| `viability_gates_exercise.py` pattern | `experiments/runtime_loop_stress_test.py` |
| `ObservationEvidence` (8 keys) + 4 actions | `tools/enforcement_evidence_contract.py` |
| SimSat null-training + v11 partial-save failure modes | shape-coverage checks in `verify_fragment()` |
| Three-stream gate-fire-rate methodology | seven-stream stress test in `runtime_loop_stress_test.py` |

Total: 4 new modules + 1 stress-test driver + 4 test suites + 3 docs + 1
notebook scenario + 3 WRITEUP subsections.

---

## Deliberately NOT incorporated

### MuZero head + Stage 2 BC policy

SimSat's MuZero head provides multi-step planning (action sequences) over
ranked encounter windows. The Gemma4Good scenarios (clinic, classroom,
deforestation) are *single-step* decisions — each session, observation, or
feedback event is a leaf in the governance trace; there is no
"plan-then-execute" structure that would benefit from MuZero.

If a future Gemma4Good scenario involved sequential enforcement actions
(e.g., "first dispatch a drone for confirmation, then file a report, then
notify authority"), MuZero-style planning would become relevant. For the
current submission, it is out of scope.

### LFM2.5-VL backend track

SimSat's Liquid Track uses Liquid AI's LFM2.5-VL-450M (a different model
family). Gemma4Good is specifically a Gemma 4 entry. The TTT mechanism
SimSat developed for LFM2.5-VL was ported here in adapter-agnostic form
(the `step_fn` in `EdgeTTTAdapter` is an injected callback, so a Gemma 4
peft loop can be plugged in directly), but the LFM-specific bits (LoRA
shape, processor wrapping) are not.

### Spectral-register tile expansion

SimSat operates across two observational registers: geometric/structural
(maritime, disaster, urban) and spectral-biochemical (pedospheric integrity
— NDVI, SWIR, vegetation indices). The Gemma4Good deforestation scenario
is more narrowly framed (Sentinel-2 NDVI-style change detection); the
spectral-biochemical extension would be a natural follow-on but doesn't
land tonight.

### REFINE_BOOST training-data weighting

SimSat's training data uses a per-row REFINE_BOOST weight that biases the
optimizer toward refine-class examples. The v45 training notebook in
Gemma4Good has a similar "synthetic_examples" appendage pattern but does
not currently use a per-row boost weight. Adding this would be a v46
training-recipe experiment, separate from the architecture work.

### Selective downlink (per-pass τ_downlink)

SimSat's "selective downlink" pattern bandlimits which observations are
transmitted to ground. The analog for Gemma4Good would be limiting which
fragments are pushed to the central syncer in a bandwidth-constrained
deployment (Indonesian classroom scenario). The current
`diloco_fragment_verifier` accepts all fragments that pass verification;
adding a τ_downlink filter on confidence × value would be a
post-competition refinement.

### Distribution-shift validation receipt

SimSat has a published distribution-shift demo: coastal-optimized weights
adapt to a polar corpus in one TTT cycle (94% of polar-regime improvement
in a single pass). The Gemma4Good integration would benefit from an
analogous demo (e.g., clinic-trained adapter adapts to a different
clinic's case mix in one DiLoCo round), but this requires multi-domain
training data and a held-out evaluation set we don't yet have.

### Garrett Sutherland's MUZERO_SEED_SWEEP methodology

The seed-variance methodology that SimSat uses (10 seeds, multi-seed
sweep with documented variance) is the right pattern for any future v46+
experiment. The current Gemma4Good v39/v42/v44/v45 evaluation uses
single-seed eval at n=20 per scenario plus n=100 focused. Adopting a
seed sweep methodology for v46 would be a worthwhile follow-up.

---

## Architectural reflection: two viability registers

SimSat and Gemma4Good operate the same governance architecture in two
structurally different settings, and the difference is worth naming:

**With human feedback (Gemma4Good clinic, classroom, monitoring station
with operator):** Ceff(t) = verified human corrections per unit time. The
system is viable when corrections outpace internally-generated error. The
gates are filters that prevent invalid corrections (denied consent, bias-
reinforcement, fragment poisoning) from entering Ceff. This is the
"verified human feedback" register.

**Without human feedback (SimSat on-orbit, autonomous edge devices):** No
human is in the loop during operation. Ceff(t) is reduced to filtered
self-observation — observations that pass the six gates BEFORE generating
an adaptation signal, plus adaptations that pass the three TTT gates before
persisting. The gates substitute for human ground truth. This is the
"governed self-correction" register.

Both registers use the same gate vocabulary, the same Merkle receipts, and
the same Viability Condition. What differs is the C(t) source: human-supplied
or self-observed-and-gated.

The Gemma4Good submission demonstrates the first register; SimSat
demonstrates the second. The two projects together demonstrate that the
Viability Condition framework is general over the source of corrective
signal, which was the headline claim in the original paper
(DOI: 10.5281/zenodo.18144681).

---

## What to look at next

If you're picking up this work after tonight:

1. **`docs/runtime_grounding_loop_2026-05-11.md`** — the four-layer
   architecture, with concrete clinic-week walkthrough
2. **`docs/diloco_integration_2026-05-11.md`** — DiLoCo theory and the three
   scenario walkthroughs
3. **`experiments/runtime_loop_stress_test.py`** — run it; produces a fresh
   stress-test receipt
4. **`docs/autonomous_session_2026-05-11.md`** — the operator morning brief
   (commit list, test counts, open items)
5. The v45 verdict at **`docs/v45_verdict_2026-05-10.md`** is the
   most recent model-training result and frames why v42 remains the
   recommended submission model

For the runtime live-stack work (bringing the semantic grounding module back
online with v42), see `D:/humanai-convention/docs/haic_dispatch_setup_2026-05-11.md`.
