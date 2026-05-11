# DiLoCo Integration with the Viability Condition

**Date:** 2026-05-11
**Status:** Architecture proposal + initial implementation
**Source materials:**
- DeepMind blog: [Decoupled DiLoCo](https://deepmind.google/blog/decoupled-diloco/) (2026)
- Paper: *Decoupled DiLoCo* (arXiv:2604.21428v1)
- Original DiLoCo: Douillard et al., 2023 — "DiLoCo: Distributed Low-Communication Training of Language Models"
- SimSat DiLoCo lab: `D:\diloco_lab\` (round-1 adapter, Gemma-4-E2B, eval N=37 exact=0.86)

---

## TL;DR

The Viability Condition `Ceff(t) > E(t)` was originally stated for single-node deployments.
This document extends it to **federated DiLoCo deployments** and shows that the same
governance pipeline (wellbeing → consent → interpretability → receipt) that validates a
single Maestro session naturally extends to fragment verification at the syncer. Compromised
or unverified learners get their fragments rejected before they can enter `Ceff_global`.

The three Gemma4Good deployment scenarios (rural health clinic, low-connectivity classroom,
deforestation monitoring) all benefit:

| Scenario | Why DiLoCo matters |
|---|---|
| Rural health clinic (k=5 sites) | Patient data stays local; only fragment deltas leave the clinic. Weekly LTE/satellite sync. Each clinic's Merkle receipt verified at syncer before fragment is accepted. |
| Low-connectivity classroom (Indonesia, 12 schools × 35 students) | 2hr/day satellite window per school. DiLoCo's ~235x bandwidth reduction makes this feasible: ~1 GB fragment payload instead of full gradient sync. |
| Deforestation monitoring (20 Amazon stations) | Stations go dark during cloud cover / hardware failure. Decoupled DiLoCo maintains 88% goodput under aggressive failure; no station can stall training. Fragment provenance verification prevents a compromised station from injecting false negatives. |

DeepMind has already validated Decoupled DiLoCo *on Gemma 4 specifically* at 12B scale across
4 US regions, achieving 198 Gbps → 0.84 Gbps bandwidth reduction. SimSat validated it on
Gemma-4-E2B at small scale. This work brings the same architecture into the Gemma4Good
governance loop.

---

## Background: Decoupled DiLoCo in 1 minute

Original DiLoCo (Douillard et al., 2023):
- Inner optimizer (each learner): standard SGD/AdamW on local data, K inner steps
- Outer optimizer (syncer): treats `(θ_local - θ_global)` as a pseudo-gradient, applies it
  to `θ_global`, broadcasts back
- Communication: every K inner steps (typically K=500), only one round-trip needed
- Result: ~500x less communication vs synchronous DDP

Decoupled DiLoCo (2026):
- Removes the lock-step synchronization barrier between rounds
- Each learner runs asynchronously; a central syncer aggregates whenever a quorum K ≤ M
  reports
- "Radial-Directional Averaging" — normalize fragments by gradient norm before averaging
  direction; aggregate norm separately. Robust to variable-speed learners.
- "Adaptive grace windows" — wait briefly for additional learners beyond the quorum to
  trade idle network capacity for sample efficiency
- Result: 88% goodput under aggressive failure (vs 58% for elastic data-parallel),
  100% system uptime with sufficient learners

The fragment is the unit of communication: a learner's contribution between syncs. In
LoRA-based DiLoCo (SimSat pattern), the fragment is the LoRA delta `ΔA, ΔB` since the
last global state. In the SimSat implementation, fragments arrive as files in
`diloco_lab/inbox/<experiment>/round-NNNNNN/<learner>/`.

---

## Federated Viability Condition

### Single-node form (current)

```
Ceff(t)  = sessions_per_day * avg_turns * consent_grant_rate     [turns/day]
E(t)     = quantization_hostility * deployment_scale_factor
Viable   ⟺ Ceff(t) > E(t)
```

See `viability/viability_condition.py`.

### Federated form (new)

For a deployment with M learners and minimum quorum K, where each learner i contributes
fragment `f_i` to a round-r sync:

```
Ceff_global(r)  = Σ over verified, accepted fragments at round r:
                   ceff_i = sessions_local_i * turns_i * consent_grant_rate_i
                   * is_verified(f_i, receipt_i)
                   * is_consent_compliant(f_i, receipt_i)

E_global(r)     = max over learners of E_i, PLUS
                   merge_error_estimate(K, quorum_grace)

Viable_global(r) ⟺ Ceff_global(r) > E_global(r)
```

Two new ingredients:

1. **`is_verified(f_i, receipt_i)`** — does the fragment's Merkle receipt verify against
   the consent/governance trace the learner claims to have executed locally? This is the
   point where a compromised or malicious learner is excluded from `Ceff_global`. Implemented
   in `tools/diloco_fragment_verifier.py`.

2. **`merge_error_estimate(K, quorum_grace)`** — Radial-Directional Averaging on K learners
   introduces some noise into the global state, especially when K is small. This is an
   additive contribution to `E_global` that scales as `1/√K` (standard error-of-mean for
   the directional component). At K=20 (the deforestation scenario), this is ~22% of
   single-learner error. At K=5 (clinic scenario), it's ~45% — but for clinics where
   weekly sync is the cadence, this is still well within the Ceff budget.

### Why this matters

The governance pipeline already produces a Merkle receipt for every Maestro session. A
DiLoCo learner running locally simply accumulates receipts over the inner-optimizer
window and produces a **round receipt** — the Merkle root over all per-session receipts.
The syncer's job is then to:

1. Verify the round receipt (Merkle integrity)
2. Verify the consent gate was respected (every per-session receipt has consent
   layers verified)
3. Verify the fragment shape matches the expected LoRA layout (no surprise injections)
4. If all three pass, the fragment is accepted into the merge
5. If any fail, the fragment is **rejected**, the learner is flagged, and the round
   proceeds with the remaining quorum

This is the federated extension of the single-node Viability Condition. The same five-layer
consent model (transcript / felt_state / gfs_activations / training_signal / retention)
that gates a single session also gates federated training contributions.

---

## Scenario walkthroughs

### Scenario A: Rural health clinic federation (k=5)

Five clinics in Bolivia, each with one Gemma-4-E2B-it instance running on local hardware
(the BEAST tier: 8 GB VRAM is enough). Patients consent to specific governance layers via
the standard HAIC ConsentGate flow.

```
Round-r workflow (weekly):
  Each clinic i (Mon-Fri):
    - Runs ~20 Maestro sessions/day
    - For each session: full governance trace, per-session Merkle receipt
    - Accumulates LoRA delta from local fine-tuning on consented training_signal
  Each clinic i (Fri night):
    - Computes round receipt = Merkle root over the week's per-session receipts
    - Packages fragment: (LoRA delta, round receipt, dataset_id label, learner_id)
    - Uploads to syncer via LTE during off-hours (~50 MB payload)
  Syncer (Sat morning):
    - Receives fragments from 5 clinics
    - Verifies each round receipt (Merkle integrity, consent compliance, shape check)
    - Computes Ceff_global(r) = Σ over verified fragments
    - Computes E_global(r) = max(E_i) + merge_error(K=5)
    - If Ceff_global > E_global: accept all, run Radial-Directional Averaging,
      broadcast new global θ to all 5 clinics
    - If any fragment rejected: log to detection log, alert operator,
      continue with remaining 4 (still above K=3 minimum quorum)
```

This gives clinics a model that improves week-over-week from their own
de-identified patient data, with zero patient data leaving the clinic — only verified
LoRA deltas. The Viability Condition is checked at the syncer; if it's violated, the
round is rolled back and the operator is alerted before any global weight update.

### Scenario B: Low-connectivity classroom (Indonesia, 12 schools × 35 students)

12 schools across 4 islands, 2-hour satellite uplink window each evening. Total ~420
students. Each school's Gemma 4 runs locally on a refurbished laptop (Q4_K_M GGUF,
~3 GB VRAM target — fits on the 4 GB integrated GPUs typical in education-grant
hardware).

DiLoCo bandwidth math (per Decoupled DiLoCo paper, scaled to Gemma-4-E2B at rank-16
LoRA):

```
Full sync (synchronous DDP, hypothetical):    ~120 MB/round per school × 12 schools
                                              = 1.4 GB through the satellite per round
                                              At 12-school WAN of ~10 Mbps:
                                              ~19 minutes per round, no margin for
                                              jitter — infeasible.

DiLoCo with K=500 inner steps:                ~0.5 MB/round per school × 12 schools
                                              = 6 MB through the satellite per round
                                              ~5 seconds per round.
                                              ~240 rounds/2hr window.
```

Without DiLoCo, this scenario is technical fiction — the satellite link can't support
synchronous training. With DiLoCo, it's straightforward.

Each school's per-session receipt covers per-student consent (under-18 layer: parental
plus student affirmative consent). The syncer verifies these before accepting the school's
fragment. A school whose receipt fails (e.g., the consent gate was bypassed due to a local
bug) is excluded from the round without disrupting the other 11.

### Scenario C: Deforestation monitoring (20 Amazon stations)

20 Sentinel-2 monitoring stations across the Amazon. Each runs Gemma 4 locally to
classify NDVI patches and produce enforcement-track recommendations (flag for human
review, escalate, no-action). The C(t) signal at each station is the satellite-derived
ground-truth (verified land cover from Sentinel-2 + station operator confirmations).

Failure modes the DiLoCo decoupling handles:
- **Cloud cover blackout** — a station with no usable Sentinel-2 imagery for a week
  has no C(t) update; its fragment is empty. Syncer's quorum mechanism handles this
  without stalling.
- **Hardware failure** — a station's edge compute dies. Decoupled DiLoCo's K-quorum
  aggregation continues with the remaining 19. Goodput target: ≥88% per the paper's
  high-failure simulations.
- **Compromise / sensor spoofing** — an attacker feeds false negatives ("no
  deforestation here, move along") to a station's local C(t). The station's per-round
  receipt either fails verification (if the attack also forged receipts — caught by
  Merkle integrity), or the receipt passes but the fragment is an outlier in the
  Radial-Directional Averaging. The latter is a soft signal, logged for review,
  not a hard rejection.

The crucial property is that no single station can stall global training (no synchronous
dependency) and no single station can poison the global model (fragment verification +
robust averaging).

---

## Implementation roadmap

Phase 1 (this document, today):
- ✅ `docs/diloco_integration_2026-05-11.md` (this file)
- ✅ `viability/distributed_viability.py` — federated `assess_federated()` and `MergeQuorumPolicy`
- ✅ `tools/diloco_fragment_verifier.py` — `verify_fragment()` with receipt + shape + consent checks
- ✅ `tests/test_distributed_viability.py` — unit tests for federated assessment
- ✅ `tests/test_diloco_fragment_verifier.py` — unit tests for the verifier
- ✅ WRITEUP.md addendum section

Phase 2 (next pass, if time):
- Notebook integration: add a 4th demo scenario (federated clinic) executing the new
  tool in a Gemma 4 function-calling loop
- Sample fragment fixtures (small enough to ship in-repo) for the demo
- End-to-end smoke test that runs the notebook against the local llama-server

Phase 3 (separate effort, not part of competition submission):
- Wire the new tools into the actual SimSat diloco_lab orchestration script
- Bring HAIC Maestro sessions into a real federated training round across the local
  BEAST + a Kaggle T4 (proof of the federated grounding loop, not just simulation)

---

## Citations

```bibtex
@article{douillard2023diloco,
  title  = {DiLoCo: Distributed Low-Communication Training of Language Models},
  author = {Douillard, Arthur and Feng, Qixuan and Rusu, Andrei A. and Chhaparia, Rachita and Donchev, Yani and Kuncoro, Adhiguna and Ranzato, Marc'Aurelio and Szlam, Arthur and Shen, Jiajun},
  journal= {arXiv preprint arXiv:2311.08105},
  year   = {2023}
}

@article{decoupleddiloco2026,
  title  = {Decoupled DiLoCo: Resilient Distributed Pre-training of Large Language Models},
  journal= {arXiv preprint arXiv:2604.21428},
  year   = {2026},
  note   = {Validated on Gemma 4 12B across four U.S. regions; 198 Gbps to 0.84 Gbps; 88\% goodput under high-failure simulation.}
}
```

---

*This work bridges the SimSat DiLoCo lab and the Gemma4Good governance pipeline. Both
projects were built around Gemma 4, both speak fluently to fragments and receipts, and
both have the same Viability Condition at their theoretical core. Connecting them is
straightforward and overdue.*
