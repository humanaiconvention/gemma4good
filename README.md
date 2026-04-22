# HAIC × Gemma 4 Good — Kaggle Hackathon Entry

**Title:** Grounding Gemma 4 in Human Lived Experience: A Convention for
Verifiable, Consent-Gated AI Alignment

**DOI (Viability Condition paper):** [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

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
│   └── haic_tools.py                 ← 7 function-calling tool implementations
├── prism_integration/                ← Prism geometry wrappers
├── maestro_integration/              ← Maestro gateway client
├── viability/
│   └── viability_condition.py        ← Standalone Ceff(t) > E(t) evaluator
├── assets/                           ← Diagrams, images
└── docs/
    ├── integration_notes.md          ← Maestro + Prism code interfaces
    └── viability_condition.md        ← Full theoretical framework
```

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
cd D:\humanai-convention\maestro
MAESTRO_LAUNCH_MODE=test MAESTRO_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))") \
  python -m uvicorn apps.gateway.main:app --reload --port 8000

# Run the notebook
cd D:\gemma4good
jupyter notebook notebook/haic_gemma4_governance.ipynb
```

---

## Key Reading

- `docs/viability_condition.md` — the mathematical foundation
- `docs/integration_notes.md` — code interfaces for Maestro and Prism
- `tools/haic_tools.py` — tool implementations
