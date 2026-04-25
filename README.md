# Gemma4Good

**[Kaggle notebook](https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent)** · **[WRITEUP.md](WRITEUP.md)** · **[Viability Condition paper (DOI)](https://doi.org/10.5281/zenodo.18144681)**

A governance agent that grounds Gemma 4 in verified human lived experience. Uses Gemma 4's native function calling to run four governance tools per AI deployment scenario — wellbeing assessment, consent verification, PRISM interpretability analysis, and a Merkle-anchored alignment receipt — enforcing the Viability Condition `Ceff(t) > E(t)` at inference time.

Validated end-to-end on three deployment scenarios (rural health clinic, low-connectivity classroom, deforestation monitoring) on both Kaggle T4×2 GPU and P100 CPU-only fallback paths.

**Kaggle Gemma 4 Good hackathon submission** by [HumanAI Convention](https://humanaiconvention.com).

Public release: `0.1`

## Current Status

- Main notebook: [notebook/haic_gemma4_governance.ipynb](notebook/haic_gemma4_governance.ipynb)
- Main writeup: [WRITEUP.md](WRITEUP.md)
- Core framework: [docs/viability_condition.md](docs/viability_condition.md)
- Core tool surface: [tools/haic_tools.py](tools/haic_tools.py)
- Tests: `143` passing (79 new governance-tool tests added post-release-0.1; full pip install and PIL compat fixes applied to Kaggle notebook)

This public repository focuses on the source, notebook, docs, and experiment logic needed to understand and reproduce the project safely. Heavyweight runtime artifacts and private local deployment state are intentionally kept out of the repo.

## Start Here

If you are new to the project:

1. Read [WRITEUP.md](WRITEUP.md)
2. Read [docs/viability_condition.md](docs/viability_condition.md)
3. Open [notebook/haic_gemma4_governance.ipynb](notebook/haic_gemma4_governance.ipynb)
4. Use [docs/integration_notes.md](docs/integration_notes.md) and [docs/beast_gemma4_loading_limitations.md](docs/beast_gemma4_loading_limitations.md) as supporting context

## What This Repo Contains

- `notebook/`
  Main Kaggle governance notebook, helper notebook scripts, and a small Gemini smoke-test helper.
- `tools/`
  Four function-calling tool implementations: wellbeing assessment, consent verification, PRISM analysis, and alignment receipt generation.
- `viability/`
  Standalone Viability Condition evaluator and incremental grounding tracker.
- `docs/`
  Theoretical framework, integration notes, deployment notes, and repo status notes.
- `experiments/`
  Curated experiment utilities, Kaggle training scaffolds, and phase-3 research tracks.
- `tests/`
  Unit coverage for the incremental grounding and grounding-tracker core.

## What This Repo Intentionally Does Not Contain

- Large model weights or GGUF artifacts
- Local runtime logs
- Secrets and local `.env` state
- One-off patch helper scripts used during notebook surgery
- Generated zip bundles that can be recreated from source

Those are intentionally kept out of the public-facing tree so git history stays reviewable and future pushes stay safer.

## Quick Local Checks

Run tests:

```bash
pytest -q
```

Open the governance notebook:

```bash
jupyter notebook notebook/haic_gemma4_governance.ipynb
```
