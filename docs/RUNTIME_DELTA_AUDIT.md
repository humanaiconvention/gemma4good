# Runtime Delta Audit

Date: 2026-04-22

This note records what still exists only in the runtime tree after the curated porting pass, and how each item should be treated.

## Key Result

The important tracked project files that were expected to differ were checked directly and are already synced byte-for-byte between:

- `D:\gemma4good`
- `D:\gemma4good\_local_worktrees\clean-github-aligned`

That includes:

- `README.md`
- `WRITEUP.md`
- `tools/haic_tools.py`
- `notebook/haic_gemma4_governance.ipynb`

So the remaining runtime-only set is now genuinely local-only material, not missed source-of-truth code.

## Runtime-Only Files and Classification

### Keep Local Only

- `.env`
  Classification: local secret / local machine configuration
  Action: do not port

- `server.log`
  Classification: local runtime log
  Action: do not port

- `experiments/v35_gov/_patch_cell6_v3.py`
- `experiments/v35_gov/_patch_v4.py`
- `experiments/v35_gov/_patch_v5.py`
- `experiments/v35_gov/_patch_v7.py`
  Classification: one-off scratch patch helpers
  Action: keep local only

- `experiments/v35_gov/seed_test.jsonl`
  Classification: scratch validation artifact
  Action: keep local only

- `haic_kaggle_utils.zip`
  Classification: generated packaging artifact
  Action: keep local only; rebuild from `build_kaggle_dataset.py` if needed

## Clean-Only Files Added Intentionally

These files exist in the curated GitHub-aligned tree even though they were not tracked in the runtime tree:

- `LICENSE`
  Classification: public-repo requirement
  Action: keep in curated tree

- `tests/conftest.py`
  Classification: portability shim for standalone pytest execution
  Action: keep in curated tree

## Guidance

When a future runtime-only file appears, classify it the same way:

- source-of-truth code or docs → port deliberately
- local runtime state or secrets → keep local
- generated artifacts → rebuild from source, do not port by default
- scratch helpers → keep local unless they graduate into maintained utilities
