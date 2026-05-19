# Submission Manifest — 2026-05-18

This file marks the Kaggle Gemma 4 Good submission set. It is a
post-submission index, not a replacement for the submitted writeup.

## Submitted Snapshot

| Field | Value |
|---|---|
| Submitted date | 2026-05-18 |
| Repository | `https://github.com/humanaiconvention/gemma4good` |
| Submitted branch | `main` |
| Submitted commit | `ec7db2e final asset: 560x280 Kaggle card preview` |
| Canonical writeup | `WRITEUP.md` |
| Video | `https://youtu.be/p5ZprNkIAEM` |
| License | Apache-2.0 (`LICENSE` + `NOTICE`) |
| DOI | `https://doi.org/10.5281/zenodo.18144681` |

Any later commit is aftercare unless a future document explicitly says
otherwise. The submitted snapshot is the factual Kaggle submission
boundary.

## Submitted Claim Set

The submission claims that HAIC operationalizes the Viability Condition
through Gemma 4 function calling, consent/provenance checks,
PRISM-style measurement, Merkle receipts, and predeclared promotion
gates.

The promoted runtime candidate at submission was:

| Field | Value |
|---|---|
| Candidate | `guard-v7 + v42` |
| Base model | Gemma 4 E2B / v42, weights unchanged |
| Guard implementation | `tools/v42_boundary_guard_v7.py` |
| Promotion hypothesis | `docs/h26_precommit_hypothesis_2026-05-17.md` |
| Verdict | `docs/h26_verdict_2026-05-17.md` |
| Canonical anchor | `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` |
| Canonical eval JSON | `experiments/v42_guard_v7_h26_canonical.json` |
| Failure sidecar | `experiments/v42_guard_v7_h26_failures.jsonl` |

The fine-tuning track is intentionally presented as an experimental
appendix: v50-v59 did not clear the promotion gates. The runtime guard
is promoted because it passed the predeclared, non-compensatory gates.

## Load-Bearing Submission Files

Narrative and public entry points:

- `README.md`
- `WRITEUP.md`
- `docs/pre_submission_checklist_2026-05-18.md`
- `docs/submission_verification_2026-05-16.md`
- `docs/media_gallery_image_specs.md`

Notebook and reproducibility artifacts:

- `notebook/haic_gemma4_governance.ipynb`
- `notebook/kernel-metadata.json`
- Public main notebook: `https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent`
- Public Tier 3 notebook: `https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation`
- Public guard reproducibility notebook: `https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4`

Promotion discipline and H-series record:

- `docs/evaluation_doctrine.md`
- `docs/promotion_workflow.md`
- `docs/v42_guard_known_limitations_2026-05-15.md`
- `docs/h18` through `docs/h26` precommit/verdict docs where present
- `experiments/v42_guard_v7_h26_canonical.json`
- `experiments/h26_results_2026-05-17.json`
- `experiments/h26_offline_results_2026-05-17.json`
- `tests/test_v42_boundary_guard*.py`

Governance architecture:

- `tools/haic_tools.py`
- `tools/enforcement_evidence_contract.py`
- `tools/federated_round_demo.py`
- `viability/`
- `prism_integration/`
- `maestro_integration/`
- `maestro_gateway/`
- `utils/merkle.py`

Media:

- `assets/media_gallery/01_cover.png`
- `assets/media_gallery/02_architecture.png`
- `assets/media_gallery/03_guard_flow.png`
- `assets/media_gallery/04_h_series_record.png`
- `assets/media_gallery/05_video_thumb.png`
- `assets/media_gallery/05_video_thumb_youtube_variant.png`
- `assets/media_gallery/06_card_560x280.png`

## Aftercare Rule

Aftercare may improve navigation, fix stale "current state" wording, add
postmortems, and archive old operational notes. It must not rewrite the
submitted evidence chain to make failures look cleaner than they were.

Historical verdicts should remain dated and local to their time. If a
historical document says "current" about an older candidate, prefer adding
a new current-state map over editing the old verdict unless the file is an
active onboarding or status document.
