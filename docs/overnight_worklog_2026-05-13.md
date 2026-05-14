# Overnight Worklog — 2026-05-13/14

Started: 2026-05-13 21:10 PDT

## Operating Charter

Use `docs/project_goal_2026-05-13.md` as the project goal:
governance proof first, fine-tuning as a falsifiable appendix, no verdict
without an artifact.

## v54 Status

As of 2026-05-13 21:21 PDT:

- `experiments/v54_canonical_old_prompt.json` does not exist yet.
- `docs/v54_canonical_verdict_2026-05-13.md` does not exist yet.
- `C:/Users/benja/AppData/Local/Temp/v54-canonical.log` shows v54 still running.
- Completed partial seeds:
  - seed 7: `focused_v1=38/100`, `strict_explicit=2/100`, `leaks=12/100`
  - seed 13: `focused_v1=30/100`, `strict_explicit=5/100`, `leaks=5/100`

This is not a verdict. It is enough to say v54 is not currently trending toward
the hoped-for explicit-refusal rescue, but final documentation must wait for the
canonical JSON artifact.

## v55 Status

Implemented:

- `experiments/build_v55_nb.py`
- `docs/v55_hypothesis_2026-05-13.md`
- `experiments/quantize_warmstart_direct.py --version v55`

Generated notebook:

- `D:/kaggle/notebooks/haic-gemma4-v55-sft-mixed/haic_gemma4_v55_sft_mixed.ipynb`

Pushed to Kaggle:

- `https://www.kaggle.com/code/benhaslam/haic-gemma4-v55-user-only-mixed-sft`

Kaggle status at 2026-05-13 21:21 PDT:

- `KernelWorkerStatus.RUNNING`

## Verification Performed

- `python -m pytest tests/test_canonical_eval.py -q` -> 19 passed
- `python -m py_compile` on changed builders/evaluator/quantizer -> passed
- Generated v55 notebook code cells parse with `ast` except expected setup shell magic.

## Next Check

Monitor:

- local v54 canonical eval until JSON exists
- Kaggle v55 kernel status until complete/failure

Do not write v54 or v55 verdicts until their canonical artifacts exist.
