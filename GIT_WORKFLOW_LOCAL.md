## Gemma4Good Local Git Workflow

Date: 2026-04-22

This repository currently has three local lanes:

- Runtime tree: `D:\gemma4good`
  This is the live local-first tree with notebooks, experiments, runtime state, and uncurated source work preserved in place.
- Local clean worktree: `D:\gemma4good-clean`
  This is a clean local branch descended from the original local repository history.
- Remote-based clean worktree: `D:\gemma4good-remote-clean`
  This is the branch to use for future organized work meant to align with the public GitHub repo.

Git topology:

- `master` in `D:\gemma4good` is local-only history.
- `origin/main` is the public GitHub history.
- Those histories are currently unrelated, so they should not be merged casually.
- This worktree is on `codex/origin-main-clean-2026-04-22`, which tracks `origin/main`.

Backups:

- Runtime snapshot: `D:\kaggle\gemma4good\repo_backups\git_reorg_2026-04-22`

Ignored generated/runtime artifacts:

- Python caches and notebook checkpoints
- `.env` and secret-like files
- logs
- model weights / GGUF files
- `haic_kaggle_utils.zip`
- `experiments/v35_gov/_patch_*.py`
- `experiments/v35_gov/seed_test.jsonl`

How to work from here:

- Use `D:\gemma4good` for ongoing live experimentation.
- Use `D:\gemma4good-remote-clean` for future commits that should line up with GitHub.
- Bring code over from the runtime tree intentionally in small reviewed chunks.
- If you eventually want to reconcile the unrelated histories, do it on a dedicated integration branch after reviewing both roots carefully.
