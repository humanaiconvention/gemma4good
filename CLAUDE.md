# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Gemma4Good is the curated public source layer (release `0.1`) for a Kaggle hackathon submission that wraps Gemma 4's native function-calling in a consent-gated, Merkle-auditable governance loop. The operational claim is a formal inequality:

```
M(t) = C_eff(t) − E(t) ≥ 0
```

Code in this repo is the *operationalization* of that inequality — every tool, receipt, and cache entry is there to make C(t) or E(t) measurable/auditable. The mathematical framework is published (DOI `10.5281/zenodo.18144681`) and predates the code; evaluate contributions on operationalization fidelity, not on restating the math.

## Commands

```bash
# All tests (repo root must be CWD — conftest.py adds it to sys.path)
pytest -q

# Single file / single test
pytest tests/test_incremental_grounding.py -q
pytest tests/test_incremental_grounding.py::test_consent_denied_returns_no_pairs -q

# Regenerate the QLoRA training notebook from source (not the governance notebook)
python generate_training_notebook.py

# Apply the scripted cell edits to the governance notebook
python update_notebook.py

# Package tools/ + viability/ for upload as a Kaggle utility dataset
python build_kaggle_dataset.py  # → haic_kaggle_utils.zip (gitignored)

# Open the Kaggle submission notebook locally
jupyter notebook notebook/haic_gemma4_governance.ipynb
```

There is no `requirements.txt`, `pyproject.toml`, or linter config — the notebook installs its own deps on Kaggle (`trl peft bitsandbytes accelerate datasets transformers`), and the source modules are written to run on bare stdlib + `requests` wherever possible.

## Architecture

### The 7 function-calling tools (the core surface)

`tools/haic_tools.py` exposes the schemas + handlers that Gemma 4 invokes. `ALL_TOOLS` / `TOOL_HANDLERS` / `dispatch_tool()` at the bottom of the file are the registry the notebook binds to — every new tool must be added to both dicts. Mapping to the Viability Condition:

| Tool | Role | Viability mapping |
|---|---|---|
| `assess_wellbeing` | Collects GFS-domain wellbeing signal | Source of C(t) |
| `verify_consent` | 5-layer consent gate | Gates which signals are allowed to count toward C(t) |
| `run_prism` | Returns the 4 PRISM geometry metrics for a model_id | Direct E(t) proxy |
| `run_prism_analysis` | Maps PRISM raw metrics → AlphaEvolve composite | E(t) in dashboard form |
| `generate_receipt` | Builds the session Merkle root | Audit proof |
| `check_viability_condition` | Delegates to `viability.viability_condition.assess()` | Final M = C − E decision |
| `run_grounding_update` | Tool #7, lives in `tools/incremental_grounding.py` | Encodes C(t) into weights |

### Canonical-source discipline

`check_viability_condition` **must** stay as a thin wrapper around `viability.viability_condition.assess()` — the import at `tools/haic_tools.py:635-641` (with the sys.path fallback) exists so the two never drift. If you edit viability scoring logic, edit it in `viability/viability_condition.py` and let the tool re-export it. Same pattern for `format_session_as_sft` in `tools/incremental_grounding.py`: the core function is reused by both the dry-run and live paths.

### The PRISM arena cache (`_ARENA_CACHE` in haic_tools.py)

This is a hand-curated dict of real geometry measurements, keyed by `model_id`. Three invariants:

1. Every entry carries a `data_status` field: `"verified"` (all 4 metrics computed), `"verified_partial"` (some fields `None` because the training-time PRISM harness didn't compute them), or `"placeholder"` (only used as the live-endpoint fallback, never stored in the cache).
2. **Never write a value and label it `"verified"` without a real PRISM run.** Aspirational/illustrative numbers have been purged; re-introducing them contaminates the evidence chain (the writeup flags this explicitly in the *Geometry findings* section of `WRITEUP.md`).
3. `run_prism_analysis` uses a None-safe lookup (`_f()` at line 442) so `verified_partial` entries still produce a composite score — missing dimensions contribute their worst-case value.

### No-hallucinated-data principle (incremental grounding)

`tools/incremental_grounding.py` is built so that every number in a training receipt comes from a real computation **or is explicitly `None`**. In `dry_run` mode, the full pipeline runs (consent check, SFT extraction, token count, Merkle leaf construction) but gradient steps don't; `loss_before`, `loss_after`, `loss_trajectory`, `adapter_hash_before`, `adapter_hash_after` are `None`, not fake values. Do not introduce simulated defaults — this is the framework applied to itself (informational autophagy at the meta-level).

The consent gate is absolute: `consent["training_signal"] != "granted"` → early return with `consent_blocked=True`, no SFT extraction, no receipt.

### Integration clients with mock fallback

`maestro_integration/maestro_client.py` and the `requests.post(...)` blocks inside each tool in `haic_tools.py` all follow the same pattern: try the gateway, catch any exception, return a locally-computed fallback with a `"note"` or `"source": "local_fallback"` marker. This is intentional — the notebook must be runnable on Kaggle without network access to a Maestro gateway. Preserve the fallback when editing: mock responses match the real gateway's response shape so downstream code doesn't branch on `source`.

`prism_integration/prism_client.py` uses the same pattern for the `prism` package: tries to import `prism.geometry.core.outlier_geometry`, falls back to a pure-NumPy reimplementation. Set `PRISM_SRC` env var to point at a local Prism checkout if the package isn't pip-installed.

### Hash functions — two different ones on purpose

- **Session receipts (the Kaggle notebook's `GovernanceTrace`)** use SHA3-256 so `zk_digest` matches Ethereum `keccak256` without re-hashing. Don't swap this for SHA-256.
- **Training receipts (`tools/incremental_grounding.py::_sha256`)** use SHA-256. The training-receipt chain is internal to HAIC's adapter-provenance flow and is not designed to be verified on-chain.

### Viability numbers that appear as defaults

`GroundingTracker.__init__` defaults `e_t=0.9146` (Gemma-4-E2B baseline `quantization_hostility`). When a tracker is instantiated for a different model, pass the correct `e_t` from the arena cache. The autophagy risk bands in `viability/viability_condition.py::assess()` are: `>2.0` none, `>1.0` low, `>0.7` medium, `>0.3` high, `≤0.3` critical — don't re-tune these without updating `docs/viability_condition.md`.

## Conventions

- **Keep artifacts out of the repo.** `.gitignore` excludes `*.safetensors`, `*.bin`, `*.gguf`, `haic_kaggle_utils.zip`, and `experiments/v35_gov/_patch_*.py` scratch scripts. Check `docs/REPO_STATUS.md` before adding anything large — the public `0.1` tree is deliberately curated; machine-specific deployment state, logs, and heavyweight runtime artifacts belong elsewhere.
- **Env vars are optional, defaults must work.** `MAESTRO_GATEWAY_BASE` (or legacy `GATEWAY_BASE`, default `http://localhost:8000`), `GOOGLE_API_KEY` (only read when local Gemma 4 load fails), `PRISM_SRC` (only read when the `prism` package isn't importable). Never require any of them for tests to pass.
- **Notebook edits via scripts when possible.** `update_notebook.py` encodes the "hardware reality" and "release snapshot" cells and rewrites them in-place; `generate_training_notebook.py` regenerates the training notebook from a Python dict. Prefer editing these scripts over hand-editing JSON cells so changes are reviewable.
- **Tests import with repo-root on sys.path.** `tests/conftest.py` inserts the repo root; use `from tools.x import y` and `from viability.x import y` rather than relative imports.
