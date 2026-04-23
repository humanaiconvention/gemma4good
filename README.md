# Gemma4Good

Grounding Gemma 4 in human lived experience through a consent-gated, auditable governance loop.

This repository is the curated public-facing code and notebook layer for the Gemma4Good project. It packages the Kaggle submission notebook, the supporting governance tools, the viability-condition framework, and the experiment/utilities that connect the notebook story to the broader HAIC deployment work.

## Current Status

- Main notebook: [notebook/haic_gemma4_governance.ipynb](notebook/haic_gemma4_governance.ipynb)
- Main writeup: [WRITEUP.md](WRITEUP.md)
- Core framework: [docs/viability_condition.md](docs/viability_condition.md)
- Core tool surface: [tools/haic_tools.py](tools/haic_tools.py)
- Tests: `64` passing in the curated branch at the time of the local push-readiness pass

Operationally, the broader HAIC stack currently treats `haic-gemma4-v35-gov` as the live interviewer model, but the heavyweight deployment artifacts themselves live outside this repo in the HAIC runtime environment and local Kaggle archives. This repo keeps the source, notebook, docs, and experiment logic needed to understand and reproduce the project structure safely.

## What This Repo Contains

- `notebook/`
  Main Kaggle governance notebook, helper notebook scripts, and a small Gemini smoke-test helper.
- `tools/`
  The seven governance/function-calling helpers used by the notebook and related experiments.
- `viability/`
  Standalone Viability Condition evaluator and incremental grounding tracker.
- `docs/`
  Theoretical framework, integration notes, deployment notes, repo status, and maintainer audit notes.
- `experiments/`
  Curated experiment utilities: `v35_gov` operational helpers, Kaggle training scaffolds, and phase-3 research tracks.
- `tests/`
  Unit coverage for the incremental grounding and grounding-tracker core.

## What This Repo Intentionally Does Not Contain

- Large model weights or GGUF artifacts
- Local runtime logs
- Secrets and local `.env` state
- One-off patch helper scripts used during notebook surgery
- Generated zip bundles that can be recreated from source

Those are intentionally kept out of the public-facing tree so git history stays reviewable and future pushes stay safer.

## Start Here

If you are new to the project:

1. Read [docs/REPO_STATUS.md](docs/REPO_STATUS.md)
2. Read [docs/viability_condition.md](docs/viability_condition.md)
3. Open [WRITEUP.md](WRITEUP.md)
4. Run or inspect [notebook/haic_gemma4_governance.ipynb](notebook/haic_gemma4_governance.ipynb)

If you are maintaining the repo locally:

1. Read [GIT_WORKFLOW_LOCAL.md](GIT_WORKFLOW_LOCAL.md)
2. Read [docs/RUNTIME_DELTA_AUDIT.md](docs/RUNTIME_DELTA_AUDIT.md)

## Quick Local Checks

Run tests:

```powershell
cd D:\gemma4good\_local_worktrees\clean-github-aligned
pytest -q
```

Open the governance notebook:

```powershell
cd D:\gemma4good\_local_worktrees\clean-github-aligned
jupyter notebook notebook/haic_gemma4_governance.ipynb
```

## Related Local Context

The full local project currently has three lanes:

- Runtime tree: `D:\gemma4good`
- Local clean-history worktree: `D:\gemma4good\_local_worktrees\clean-local-history`
- GitHub-aligned worktree: `D:\gemma4good\_local_worktrees\clean-github-aligned`

The public GitHub branch should be prepared from the GitHub-aligned worktree, not from the runtime tree.
