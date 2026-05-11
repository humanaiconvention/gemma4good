# The Runtime Grounding Loop — TTT × DiLoCo × Viability Condition

**Date:** 2026-05-11
**Status:** Architecture reference (implementations in `viability/`, `tools/`, tests passing)
**Companions:** `docs/diloco_integration_2026-05-11.md`, `docs/viability_condition.md`

---

## The three-layer stack

The HAIC governance pipeline now spans three composable layers of runtime
control, each with a different time scale and a different unit of accountability:

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 3: SYSTEM     Viability Condition Ceff(t) > E(t)               │
│         (federation) viability/distributed_viability.py              │
│                                                                      │
│  Time scale: per sync round (hours to days)                          │
│  Unit:       the whole federation                                    │
│  Decision:   commit · rollback · alert_operator                      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ accepts/rejects fragments
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 2: FRAGMENT   DiLoCo Fragment Verifier                         │
│         (per learner) tools/diloco_fragment_verifier.py              │
│                                                                      │
│  Time scale: per sync round, per learner (minutes to hours)          │
│  Unit:       one learner's round-receipt + LoRA delta                │
│  Checks:     Merkle integrity · consent compliance · shape · norms   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ accepts/rejects updates
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1: STEP       TTT Gates (error_bias BLOCKING)                  │
│         (per device) viability/ttt_gates.py + tools/edge_ttt_adapter │
│                                                                      │
│  Time scale: per operator feedback (seconds)                         │
│  Unit:       one signed-error scalar from one Maestro session        │
│  Gates:      error_bias (BLOCK) · weight_drift (warn) · rate (warn)  │
└──────────────────────────────────────────────────────────────────────┘
```

Each layer presupposes the one below it: Layer 3 trusts the verified fragments
that Layer 2 admitted; Layer 2 trusts the per-session receipts that Layer 1
produced; Layer 1 trusts the operator-feedback contract enforced at the
consent gate.

## What flows up

Layer 1 → Layer 2:
- Per-session receipt (SHA3-256 over the session's governance trace)
- TTT trace (which steps applied, which were blocked, weight drift summary)
- These become Merkle leaves in the per-learner round receipt

Layer 2 → Layer 3:
- Verified LoRA fragment + round receipt
- These get summed/averaged at the syncer to update `θ_global`
- Rejected fragments are logged but excluded from `Ceff_global`

## What flows down

Layer 3 → Layer 2:
- The current `θ_global` after commit (broadcast back to learners)
- Federation viability state (commit · rollback · alert)

Layer 2 → Layer 1:
- The new baseline weights for the next round's TTT
- A reset signal if the federation rolled back

## Why each layer needs the others

**TTT without DiLoCo:** Edge devices drift to their local distributions
indefinitely. No mechanism for cross-site knowledge transfer; no recovery
from local hardware failure or capture. The classroom in Sulawesi never
learns from the classroom in Bali.

**DiLoCo without TTT:** Edge devices don't adapt to local conditions
between sync rounds. The 2-hour daily satellite window arrives weekly with
a single global update — fine for batch fine-tuning, terrible for the
"continually refined by operator feedback" pitch in the WRITEUP.

**DiLoCo + TTT without the Viability Condition:** The system might train
fine, but you have no system-level invariant to point at. No way to say
"this federation is in a viable state; commit"; no way to roll back when
the corrective bandwidth is being outpaced by the error rate.

**All three together:** A real-time, per-device adaptation loop that
accumulates into round receipts, gets verified before merging globally,
and is checked against the federation-scale viability invariant before
each commit. Every gradient signal traceable from operator click → step
gate → round receipt → federation viability decision.

## Concrete walkthrough: one clinic, one week

Monday morning at the Bolivian clinic:
- Patient consents to all five layers via the standard HAIC ConsentGate.
- Maestro session runs; clinician corrects two model predictions during the
  session (these are the operator feedback signals).
- For each correction, `EdgeTTTAdapter.step(feedback)` is called.
- TTT gates evaluated PRE-step. If the morning has been characterised by
  systematic over-prediction (all positive errors), error_bias blocks the
  step — the local model stops reinforcing its over-confident behaviour
  and waits for the distribution to diversify.
- Steps that pass the gate apply the LoRA delta locally.
- Session receipt produced (Merkle root over governance trace).

This repeats through the week. By Friday night:
- Clinic accumulates ~80 session receipts.
- TTT adapter has applied maybe 60 of the 100 attempted updates; the other
  40 were blocked by error_bias (which is healthy — the gate is doing its
  job).
- `EdgeTTTAdapter.export_receipt()` produces the week's TTT trace.
- The cumulative LoRA delta + the week's receipts are packaged as a
  fragment for the DiLoCo syncer.

Saturday morning at the syncer:
- Fragments arrive from 5 clinics.
- `verify_fragment()` runs against each. Bolivia, Peru, Ecuador all pass.
  Colombia's receipt has a Merkle mismatch — flagged, fragment rejected.
  Chile's fragment has a tensor norm anomaly — flagged, fragment rejected.
- `assess_federated()` runs on the 5 contributions (3 verified, 2 rejected,
  K=3 just meets quorum).
- Ceff_global is the sum of 3 verified clinics' corrective bandwidth.
  E_global is the worst-case quantization hostility across all 5 plus
  `1/√3 ≈ 0.58` merge-error term. If Ceff_global > E_global, the round
  commits; if not, rollback.
- Because we had rejections, even a successful commit triggers
  `alert_operator` — the rejections need investigation before the next round.

Sunday: new global θ broadcasts to all 5 clinics. Bolivia's baseline weights
reset to the new global state; the TTT update_count resets to 0; weekday
adaptation resumes.

## Implementation status (2026-05-11)

All three layers implemented and unit-tested:

| Layer | Module | Tests |
|---|---|---|
| 1 (step) | `viability/ttt_gates.py`, `tools/edge_ttt_adapter.py` | 25/25 passing |
| 2 (fragment) | `tools/diloco_fragment_verifier.py` | 13/13 passing |
| 3 (federation) | `viability/distributed_viability.py` | 10/10 passing |
| Original (system) | `viability/viability_condition.py` | (pre-existing) |

48 new tests total; all green. No torch/peft dependency in any of the new
modules — the gradient step is an injected callback, so the gate logic
tests run in milliseconds without a GPU.

## What is and is not in this submission

In scope:
- The three layers above, with full tests and documentation.
- WRITEUP.md sections on federated deployment (Layer 2-3) and the runtime
  grounding loop (Layer 1).
- Citation of the original DiLoCo paper, Decoupled DiLoCo (arXiv:2604.21428),
  and the SimSat viability-gates exercise.

Out of scope for this submission (future work):
- Live federation across two physical machines (the BEAST + a Kaggle T4).
  The implementation supports this; we just haven't run an end-to-end
  multi-machine round yet.
- A torch/peft-backed concrete `step_fn` for `EdgeTTTAdapter`. The contract
  is in place; the actual fine-tuning callback that wraps a PeftModel and
  calls forward+backward+step has not been written for this repo yet
  (SimSat has the equivalent in `src/sim/observation_vla/lfm_ttt.py`).
- A notebook cell that demonstrates the runtime grounding loop end-to-end
  in the Gemma 4 function-calling style. The mechanics work; the demo
  surface hasn't been built.

---

*The runtime grounding loop is what turns "AI grounded in human lived
experience" from a slogan into an architecture. Operator feedback enters
through consented Maestro sessions; the per-step gate prevents systematic
bias from compounding; the per-fragment gate prevents compromised learners
from poisoning the federation; the per-federation invariant catches the
case where the whole system has drifted out of viable territory. Every
gradient step is traceable from operator click to global commit.*
