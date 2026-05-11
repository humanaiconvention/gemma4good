# HAIC × Gemma 4 Good — Kaggle Hackathon Entry

**Title:** Grounding Gemma 4 in Human Lived Experience: A Convention for
Verifiable, Consent-Gated AI Alignment

**DOI (Viability Condition paper):** [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

**Authors:** Benjamin Haslam (Bazzer) and Garrett Sutherland — collaborative entry
*Garrett Sutherland's contribution: `experiments/sgt_harness.py` — the rigorous SGT evaluation harness with Wilson 95% CIs, the statistical standard used for all model evaluations in this submission.*

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

## Project Structure

```
gemma4good/
├── notebook/
│   └── haic_gemma4_governance.ipynb  ← main Kaggle submission
├── tools/
│   ├── haic_tools.py                 ← 7 function-calling tool implementations
│   ├── incremental_grounding.py      ← session-driven continual learning
│   ├── eval_leakage_check.py         ← Gate 2: scenario-vs-shard hash check
│   ├── check_promotion.py            ← Gate decision: PROMOTED/BLOCKED CLI
│   ├── evaluate_promotion.py         ← single-entry pipeline wrapper
│   └── eval_receipt.py               ← Merkle-anchored eval receipt (SHA3-256)
├── experiments/
│   ├── sgt_harness.py                ← rigorous SGT (Garrett Sutherland's)
│   ├── sgt_extended_scenarios.py     ← 10 grounding + 5 security scenarios
│   ├── run_v38_sgt.py                ← BEAST runner (1-turn)
│   ├── run_v38_sgt_2turn.py          ← BEAST runner (2-turn, kaggle-pattern)
│   ├── inspect_security_responses.py ← failure-mode dissection helper
│   └── kaggle_cell_rigorous_sgt.py   ← drop-in cell for kaggle build scripts
├── tests/
│   └── test_*.py                     ← 113 unit tests for the eval pipeline
├── prism_integration/                ← Prism geometry wrappers
├── maestro_integration/              ← Maestro gateway client
├── viability/
│   └── viability_condition.py        ← Standalone Ceff(t) > E(t) evaluator
├── utils/
│   └── merkle.py                     ← shared SHA3-256 + Merkle root utilities
├── assets/                           ← Diagrams, images
└── docs/
    ├── evaluation_doctrine.md        ← six-gate model evaluation doctrine
    ├── promotion_workflow.md         ← end-to-end promotion pipeline
    ├── v39_recipe.md                 ← next training run proposal
    ├── audit_humanai_convention_pipeline.md
    │                                 ← gap analysis vs upstream pipeline
    ├── writeup_addendum_2026-05-08.md
    │                                 ← rigorous re-eval companion to WRITEUP
    ├── integration_notes.md          ← Maestro + Prism code interfaces
    └── viability_condition.md        ← Full theoretical framework
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
