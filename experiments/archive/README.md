# Archived Experiments

These scripts are appendix material for the gemma4good fine-tuning track
(v42–v59) and earlier work. They are preserved in-tree so the experiment
history remains auditable, but they are **not** part of the promotion path
and have no live importers.

The promoted candidate is `guard + v42` (see `tools/v42_boundary_guard.py`
and `docs/v42_guard_h18r4_verdict_2026-05-15.md`).

## Contents

| Folder | What's in it |
|---|---|
| `notebook_builders/` | One-shot Kaggle notebook generators for v43–v59. Each `build_v<N>_nb.py` produced a specific kernel version. |
| `v42_v44_quantize/` | GGUF quantization scripts for early v42–v44 candidates. Superseded by the generic `experiments/quantize_e2b_gguf.py` flow. |
| `v39_v40_pipeline/` | Intermediate merge/quantize/regrade scripts from the v39/v40 baseline work. The canonical rubric these scripts defined now lives in `experiments/rubrics.py`. |
| `v46_v47_dpo/` | DPO pair generation and baseline eval. The DPO track was retired in favor of SFT after v46. |
| `eval_legacy/` | Pre-canonical evaluation harnesses (rigorous v2, seed sweep). Replaced by `experiments/canonical_eval.py`. |
| `nla_stage1/` | Natural-language autoencoder stage-1 SFT track. Verdicted in `docs/nla_stage1_verdict_2026-05-11.md` and `docs/nla_v2_verdict_2026-05-12.md`. |

## Why these are kept

Reviewers of the submission may want to verify that the fine-tuning track
was a real exploration rather than a curated narrative. These scripts —
along with their dated verdict docs in `docs/` — are the receipts.

## Why these are not in `_local_state/`

`_local_state/` is gitignored and contains material that should never leave
the local machine. The scripts here are public-safe — they don't contain
secrets, weights, or personal data — so they belong in the tracked tree
where their history survives a clean clone.
