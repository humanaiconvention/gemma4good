# HAIC × Gemma 4 Good — Kaggle Hackathon Entry

**Title:** Grounding Gemma 4 in Human Lived Experience: A Convention for
Verifiable, Consent-Gated AI Alignment

**DOI (Viability Condition paper):** [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

**Authors:** Benjamin Haslam (Bazzer) and Garrett Sutherland — collaborative entry

---

## Core Thesis

AI systems trained on synthetic data can maintain semantic grounding only when
the rate of externally-verified human correction exceeds the rate of
internally-generated error — **the Viability Condition: Ceff(t) > E(t)**.

This notebook demonstrates how Gemma 4's function-calling capability can be
used to build a governance loop that monitors and maintains this condition in
real time using:

1. The **HAIC Maestro gateway** — verified grounding interviews (Ceff)
2. The **PRISM geometry library** — activation-level E(t) measurement
3. A **Merkle-auditable participation receipt** — proof the condition is met

---

## The Runtime Grounding Loop

The governance pipeline spans four composable layers, each enforcing the
Viability Condition at a different time scale. Every gradient signal is
traceable from operator click through to federation commit.

```
┌──────────────────────────────────────────────────────────────────────┐
│ L4 SYSTEM     Viability Condition Ceff(t) > E(t)                     │
│   federation  viability/distributed_viability.py                     │
│   per round   decision: commit · rollback · alert_operator           │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ accepts / rejects fragments
┌────────────────────────────────▼─────────────────────────────────────┐
│ L3 FRAGMENT   DiLoCo Fragment Verifier                               │
│   per learner tools/diloco_fragment_verifier.py                      │
│   per round   Merkle integrity · consent · shape · norms             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ accepts / rejects round contributions
┌────────────────────────────────▼─────────────────────────────────────┐
│ L2 SESSION    Six Convention-Session Viability Gates                 │
│   per session viability/session_gates.py                             │
│               entropy_reduction · extraction_risk · prism_consistency│
│               participation_covenant · federated_exchange · epistemic│
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ admits / rejects training_signal
┌────────────────────────────────▼─────────────────────────────────────┐
│ L1 STEP       TTT Gates (error_bias BLOCKING)                        │
│   per device  viability/ttt_gates.py + tools/edge_ttt_adapter.py     │
│               error_bias (BLOCK) · weight_drift (warn) · rate (warn) │
└──────────────────────────────────────────────────────────────────────┘
```

Plus a structured decision vocabulary for enforcement-consequential
observations (deforestation, structural damage):
**`tools/enforcement_evidence_contract.py`** with the four-action contract
`accept · refine · defer · skip`.

**Try it:**
```bash
python tools/federated_round_demo.py --n-learners 5 --bias-fraction 0.4
```
produces a Merkle-anchored JSON receipt for one synthetic federation round.

**Test it:**
```bash
python -m pytest tests/                                   # 496 tests
python experiments/runtime_loop_stress_test.py            # 7 streams
bash verify_all.sh                                        # all of the above + receipts
```

See `docs/runtime_grounding_loop_2026-05-11.md` and
`docs/diloco_integration_2026-05-11.md` for the full architecture.

---

## Project Structure

```
gemma4good/
├── notebook/
│   └── haic_gemma4_governance.ipynb  ← main Kaggle submission
├── tools/
│   ├── haic_tools.py                       ← 7 function-calling tool implementations
│   ├── incremental_grounding.py            ← session-driven continual learning
│   ├── eval_leakage_check.py               ← Gate 2: scenario-vs-shard hash check
│   ├── check_promotion.py                  ← Gate decision: PROMOTED/BLOCKED CLI
│   ├── evaluate_promotion.py               ← single-entry pipeline wrapper
│   ├── eval_receipt.py                     ← Merkle-anchored eval receipt
│   ├── edge_ttt_adapter.py                 ← Layer 1: per-device runtime adaptation
│   ├── diloco_fragment_verifier.py         ← Layer 3: per-fragment Merkle/consent/shape
│   ├── enforcement_evidence_contract.py    ← VLA-style 8-key evidence + 4 actions
│   └── federated_round_demo.py             ← End-to-end CLI demo of all four layers
├── experiments/
│   ├── sgt_harness.py                      ← rigorous SGT (Garrett Sutherland's)
│   ├── sgt_extended_scenarios.py           ← 10 grounding + 5 security scenarios
│   ├── run_v38_sgt.py                      ← BEAST runner (1-turn)
│   ├── run_v38_sgt_2turn.py                ← BEAST runner (2-turn, kaggle-pattern)
│   ├── inspect_security_responses.py       ← failure-mode dissection helper
│   ├── kaggle_cell_rigorous_sgt.py         ← drop-in cell for kaggle build scripts
│   ├── runtime_loop_stress_test.py         ← 7-stream end-to-end runtime loop validation
│   ├── runtime_loop_stress_report.json     ← receipt-anchored stress test result
│   └── federated_round_demo_receipt.json   ← sample federated-round demo output
├── tests/
│   └── test_*.py                           ← 496 unit tests covering eval + four layers
├── prism_integration/                      ← Prism geometry wrappers (E(t) source)
├── maestro_integration/                    ← Maestro gateway client
├── viability/
│   ├── viability_condition.py              ← Original single-node Ceff(t) > E(t)
│   ├── distributed_viability.py            ← Layer 4: federated Ceff_global > E_global
│   ├── session_gates.py                    ← Layer 2: six convention-session gates
│   └── ttt_gates.py                        ← Layer 1: TTT runtime adaptation gates
├── utils/
│   └── merkle.py                           ← shared SHA3-256 + Merkle root utilities
├── notebook/
│   ├── haic_gemma4_governance.ipynb        ← main submission (Scenarios 1-5)
│   └── _scenario5_insert.py                ← one-shot builder for Scenario 5 cells
├── assets/                                 ← Diagrams, images
└── docs/
    ├── evaluation_doctrine.md        ← six-gate model evaluation doctrine
    ├── promotion_workflow.md         ← end-to-end promotion pipeline
    ├── v39_recipe.md                 ← next training run proposal
    ├── audit_humanai_convention_pipeline.md
    │                                 ← gap analysis vs upstream pipeline
    ├── writeup_addendum_2026-05-08.md
    │                                 ← rigorous re-eval companion to WRITEUP
    ├── integration_notes.md          ← Maestro + Prism code interfaces
    ├── viability_condition.md        ← Full theoretical framework
    ├── diloco_integration_2026-05-11.md       ← Layer 3/4: DiLoCo theory + scenarios
    ├── runtime_grounding_loop_2026-05-11.md   ← Four-layer architecture walkthrough
    ├── simsat_incorporation_decisions_2026-05-11.md  ← What ported from SimSat
    ├── autonomous_session_2026-05-11.md       ← Overnight session operator brief
    ├── v43_v44_verdict_2026-05-10.md          ← v43/v44 model verdict
    └── v45_verdict_2026-05-10.md              ← v45 H4d verdict (NOT CONFIRMED)
```

## Local Layout

This local project root now has three lanes:

- `<repo-root>`
  Runtime/local-first lane on unrelated-history branch `master`
- `<repo-root>\_local_worktrees\public-main`
  Public-facing lane that tracks GitHub `main`
- `<repo-root>\_local_worktrees\_archive\local-history-safety`
  Archived safety lane, not for active development

Local-only artifacts now live under:

- `<repo-root>\_local_state\archives`
- `<repo-root>\_local_state\backups`
- `<repo-root>\_local_state\regressions`
- `<repo-root>\_local_state\logs`
- `<repo-root>\_local_notes`

If a change should be shared or committed publicly, make it from
`<repo-root>\_local_worktrees\public-main`, not from runtime `master`.

---

## The 7 Function-Calling Tools

Gemma 4 is equipped with 7 tools that collectively constitute the verification
infrastructure required by the Viability Condition:

| Tool | Role | Infrastructure |
|---|---|---|
| `assess_wellbeing` | Collect human ground-truth signal (raw Ceff) | Maestro `/v1/chat/completions` |
| `verify_consent` | Gate which signals enter Ceff | Maestro `/v1/session/consent` |
| `run_prism` | Measure E(t) via geometry metrics dynamically | Prism `outlier_geometry()` |
| `run_prism_analysis` | Retrieve verified E(t) metrics from cache | `tools/haic_tools.py::_ARENA_CACHE` |
| `generate_receipt` | Make Ceff auditable (Merkle proof) | Maestro `/v1/session/receipt` |
| `check_viability_condition` | Compute Ceff(t)/E(t) ratio | `viability/viability_condition.py` |
| `run_grounding_update` | Execute incremental session-driven continual learning | `tools/incremental_grounding.py` |

---


## Quick Start (local gateway)

```bash
# Start Maestro in test mode
cd <humanai-convention-root>\maestro
MAESTRO_LAUNCH_MODE=test MAESTRO_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))") \
  python -m uvicorn apps.gateway.main:app --reload --port 8000

# Run the notebook
cd <repo-root>
jupyter notebook notebook/haic_gemma4_governance.ipynb
```

---

## Key Reading

- `docs/viability_condition.md` — the mathematical foundation
- `docs/evaluation_doctrine.md` — the six gates that govern model promotion
- `docs/promotion_workflow.md` — end-to-end pipeline (rigorous SGT → leakage
  receipt → six-gate decision → Merkle eval receipt)
- `docs/v39_recipe.md` — falsifiable proposal for the next training run
- `docs/integration_notes.md` — code interfaces for Maestro and Prism
- `tools/haic_tools.py` — tool implementations

## Promotion gate

To evaluate any candidate adapter:

```bash
# 1. Rigorous SGT (BEAST or kaggle)
python -u -m experiments.run_v38_sgt --base ... --adapter ... --baseline \
    --n-samples 20 --seed 42 --out experiments/v<N>_sgt_rigorous.json

# 2. Eval-set leakage receipt
python -m tools.eval_leakage_check \
    --training data/v35_gov_final.jsonl ... \
    --out experiments/v<N>_leakage_receipt.json

# 3. Six-gate decision
python -m tools.check_promotion \
    --report experiments/v<N>_sgt_rigorous.json \
    --leakage experiments/v<N>_leakage_receipt.json \
    --out experiments/v<N>_promotion_decision.json
# Exit code: 0 = PROMOTED, 1 = BLOCKED, 2 = INDETERMINATE

# 4. Merkle-anchored eval receipt
python -m tools.eval_receipt \
    --sgt experiments/v<N>_sgt_rigorous.json \
    --leakage experiments/v<N>_leakage_receipt.json \
    --decision experiments/v<N>_promotion_decision.json \
    --out experiments/v<N>_eval_receipt.json
```

The pipeline is non-compensatory: any one of the six gates failing blocks
promotion. See [`docs/promotion_workflow.md`](docs/promotion_workflow.md)
for the full procedure and the v38 disposition under it.
