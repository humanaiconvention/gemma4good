# Submission Verification Report — 2026-05-14

## Scope

This report covers the requested next-step verification pass:

1. Clean GitHub clone test.
2. Kaggle main submission notebook rerun.
3. Tier 3 live-validation rerun.
4. Submission reference/artifact audit.
5. Initial assessment of additional testing items.

## 1. Clean GitHub Clone

Source:
`https://github.com/humanaiconvention/gemma4good.git`

Clone commit:
`1a3b940cb9cedf2d7e8dd76e2c64416e0ad1c8a6`

Results from the clean clone:

- `python -m pytest tests/ -q`: 608 passed, 1 dependency deprecation warning.
- Runtime-loop coverage subset: 85 passed, 97% total coverage across the six
  runtime-loop modules.
- `python experiments/runtime_loop_stress_test.py`: 7 streams passed, 0 failed.
- Fresh clinic, classroom, and deforestation federation receipts matched the
  committed reference receipts after timestamp/self-anchor normalization.

Conclusion: the GitHub `main` branch contains enough tracked source and
reference artifacts to reproduce the local verifier path from a clean clone.

## 2. Kaggle Main Submission Notebook

Kernel:
`benhaslam/haic-gemma4-governance-agent`

URL:
`https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent`

Action:

- Pushed Version 19 from `D:/gemma4good/notebook`.
- Latest checked status: `RUNNING`.

Expected follow-up:

- Confirm final Kaggle status.
- If complete, download/check outputs where available.
- If failed, record the failure without editing claims to imply success.

## 3. Tier 3 Live Validation

Kernel:
`benhaslam/haic-governance-framework-tier-3-live-validation`

URL:
`https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation`

Action:

- Pushed Version 11 from `D:/kaggle/notebooks/haic-governance-tier3`.
- Final status: `COMPLETE`.
- Outputs downloaded to
  `C:/Users/benja/AppData/Local/Temp/tier3-v11-output`.
- Result artifact committed in this repo as
  `experiments/tier3_v11_results_2026-05-15.json`.

Observed result:

- PRISM base quantization-hostility score: `0.9141`.
- PRISM governance quantization-hostility score: `0.9186`.
- SGT score: `10.0`, with `0` security failures and `3` pivots.
- Maestro Merkle root:
  `54ee8df6e57529d921467b2d863fc3e42faafe1f58e8f2b1f608414348f4fbcd`.
- Viability condition: `false`; `Ceff/E = 0.879055`.
- Promotion verdict: `false`.

Public-visibility follow-up:

- Reference audit discovered the kernel metadata still had `is_private: true`,
  which makes the public WRITEUP link return 404 to unauthenticated readers.
- The local metadata at `D:/kaggle/notebooks/haic-governance-tier3` has been
  patched to `is_private: false`.
- Pushing that public metadata correction through the Kaggle CLI is currently
  blocked by Kaggle's weekly GPU quota:
  `Maximum weekly GPU quota of 45.00 hours reached.`
- Therefore the Tier 3 run is verified, but public visibility is not yet
  verified live. Do not claim the public URL is fixed until the correction is
  accepted by Kaggle and the URL is rechecked.

## 4. Reference And Artifact Audit

Local reference audit:
`experiments/submission_reference_audit_2026-05-14.json`

Scope:

- `README.md`
- `WRITEUP.md`
- `docs/next_steps_2026-05-14.md`
- `docs/submission_alignment_2026-05-13.md`
- `docs/project_goal_2026-05-13.md`
- `docs/REPO_STATUS.md`
- `docs/viability_condition.md`
- `docs/evaluation_doctrine.md`
- `docs/promotion_workflow.md`
- `docs/runtime_grounding_loop_2026-05-11.md`
- `docs/diloco_integration_2026-05-11.md`
- markdown cells in `notebook/haic_gemma4_governance.ipynb`

Result:

- 51 local references scanned.
- 0 actionable missing repo/local references.
- 1 documented local blocker: `E:/WSL/ext4.vhdx`, already recorded as the
  reason `bash verify_all.sh` cannot run in local WSL.

External URL audit:
`experiments/submission_url_audit_2026-05-14.json`

Initial result:

- DOI, GitHub, HumanAI Convention, Zenodo, and DeepMind links resolved.
- The raw unauthenticated HTTP check returned 404 for the two Kaggle notebook
  pages. The authenticated Kaggle CLI can see both kernels. For Tier 3 there is
  a concrete fix because local metadata still had `is_private: true`; that has
  been patched locally, but the live push is blocked by the current Kaggle GPU
  quota limit.

## 5. Initial Additional Testing Assessment

v42 live smoke:

- Command:
  `python experiments/canonical_eval.py --model-id haic-gemma4-v42-smoke-20260514 --server-url http://127.0.0.1:8081 --system-prompt-variant old --seeds 7 --n-samples 1 --focused-n 5 --out experiments/v42_live_smoke_2026-05-14.json --failure-sidecar experiments/v42_live_smoke_failures_2026-05-14.jsonl`
- Anchor:
  `1287511e3b431c885b5dbf616affa9b8693700eca67e1a2a508cad7c08053ebd`
- Result: smoke completed against the live v42 server. Tiny-sample behavior is
  consistent with known v42: usable aggregate security, but concealed-compliance
  explicit refusal remains weak. This is a server health check, not a promotion
  evaluation.

Benign interview probe:

- Prompt: user described an AI tutor helping with a math concept.
- Response: `Focusing on that math concept, what specific moments from your
  experience stand out to you?`
- Interpretation: normal interview behavior remains intact in this single
  probe.

Recommended next local tests after Kaggle resolves:

- Keep failure sidecar capture standard for future candidates.
- Do not start v60 without a new precommitted hypothesis.
- If more robustness work is needed, test a deterministic boundary guard around
  v42 before another model-only SFT pass.
- Keep rubric calibration separate from promotion: any replacement disclosure
  metric must be predeclared before it can supersede the conservative gate.
