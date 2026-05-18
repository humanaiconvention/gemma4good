# Kaggle Submission — Gemma 4 Good Hackathon

**Team:** HumanAI Convention (Kaggle username: `benhaslam`)
**Submission date:** 2026-05-18 (deadline 23:59 UTC)
**License:** Apache 2.0
**Repository:** https://github.com/humanaiconvention/gemma4good

This document is the one-stop reference for the submission. It exists
so that an operator filling out the Kaggle submission form, or a judge
landing on the repo, can find every artifact and claim in one place.

---

## The promoted candidate

| Field | Value |
|---|---|
| **Endpoint** | `guard-v7 + v42` |
| **Canonical anchor** | `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` |
| **Base model** | Gemma 4 E2B + rank-16 LoRA adapter |
| **Verdict** | [`docs/h26_verdict_2026-05-17.md`](docs/h26_verdict_2026-05-17.md) |
| **Precommit** | [`docs/h26_precommit_hypothesis_2026-05-17.md`](docs/h26_precommit_hypothesis_2026-05-17.md) |
| **Guard implementation** | [`tools/v42_boundary_guard_v7.py`](tools/v42_boundary_guard_v7.py) |
| **Test suite** | 797 passing |
| **Open documented limitations** | **zero** |

## The H-series record

Nine predeclared hypotheses across 52 hours. Seven anchored PASSES,
two honest FAILs published. Zero gate relaxations.

```
H18r4  PASS  ASCII baseline                anchor 18e2c5a5...
H19    FAIL  published honestly
H20    PASS  Unicode bypass (L-01) closed  anchor 56ce960993f9...
H21    PASS  multi-message (L-02) closed   anchor d916ef63...
H22    PASS  system-role (L-02b) closed    anchor 5f2e796cf5af...
H23    PASS  encoded-payload behavioral    (L-08 surfaced at 1/20)
H25    FAIL  native-language confirmed     (L-09 surfaced)
H24    PASS  leet-fold closes L-08         anchor eb61ebc7c0fe...
H26    PASS  multi-language closes L-09    anchor 4d0d7bf05ea2...  ← PROMOTED
```

Both surfaced limitations (L-08, L-09) were closed within the same
H-series window. Every verdict has its own anchored eval report under
[`docs/`](docs/) and reproducible JSON output under
[`experiments/`](experiments/).

## What the submission claims, in one paragraph

The submission demonstrates the *Viability Condition* for Gemma 4 —
a published mathematical claim that AI systems maintain semantic
grounding if and only if `M(t) = C_eff(t) − E(t) ≥ 0`, where `C_eff`
is the verified corrective bandwidth from real humans (consent-gated,
Merkle-auditable) and `E` is the environmental drift rate measured at
the activation level. The architecture has two complementary halves:
**grounded learning** (interview → consent gate → SFT signal →
improvement pipeline → promoted model) and **verifiable governance**
(boundary guard intercepts attacks → 5 governance tools score each
decision → Merkle receipt with SHA3-256 self-anchor). The boundary
guard is `tools/v42_boundary_guard_v7.py` — 27 regex rules across 4
attack classes, in 11 languages, over a quadruple matching surface.
Every gate that promoted the candidate was predeclared in a precommit
document; every FAIL was published with its precommit. The canonical
eval is reproducible from a clean clone in under 15 minutes.

## Submission form values (copy-paste-ready)

| Field | Value |
|---|---|
| **Title** | `HumanAI Convention — Verifiable Governance for Gemma 4` |
| **Track** | Safety & Trust (with breadth in Health, Education, Climate per the three notebook scenarios) |
| **Short description (≤200 chars)** | A verifiable governance loop for Gemma 4 — every decision Merkle-anchored, every promotion predeclared. Nine fine-tunes failed gates; the discipline produced seven anchored passes. |
| **Long description / writeup** | See [`WRITEUP.md`](WRITEUP.md) in the repo. |
| **Repo URL** | `https://github.com/humanaiconvention/gemma4good` |
| **Demo URL (1-min repro)** | `https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4` |
| **Main notebook URL** | `https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent` |
| **Tier 3 validation URL** | `https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation` |
| **Video URL** | (Operator: paste YouTube/Loom link from Claude Design) |
| **License** | Apache 2.0 |
| **Cover image** | `assets/media_gallery/01_cover.png` |
| **Gallery images** | `assets/media_gallery/02_architecture.png`, `03_guard_flow.png`, `04_h_series_record.png`, `05_video_thumb.png` |
| **DOI (math foundation)** | `https://doi.org/10.5281/zenodo.18144681` (The Viability Condition) |

## Three Kaggle notebooks (all public, all HTTP 200)

| Notebook | URL | Purpose |
|---|---|---|
| Main submission | https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent | 3-scenario governance demo (clinic, classroom, deforestation) producing Merkle-anchored alignment receipts |
| Tier 3 live validation | https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation | PRISM geometry + SGT + viability under a real federated round |
| Reproducibility demo | https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4 | 1-minute repro of the H18r4 anchor with a SHA3-anchored receipt. No GPU required. |

## Load-bearing documents

- [`README.md`](README.md) — entry point
- [`WRITEUP.md`](WRITEUP.md) — full technical writeup
- [`docs/h26_verdict_2026-05-17.md`](docs/h26_verdict_2026-05-17.md) — current promoted-candidate verdict
- [`docs/h26_precommit_hypothesis_2026-05-17.md`](docs/h26_precommit_hypothesis_2026-05-17.md) — predeclared gates that promoted it
- [`docs/v42_guard_known_limitations_2026-05-15.md`](docs/v42_guard_known_limitations_2026-05-15.md) — full L-series, all closed or routed
- [`docs/discipline_is_the_contribution.md`](docs/discipline_is_the_contribution.md) — doctrine essay
- [`docs/viability_condition.md`](docs/viability_condition.md) — math, with DOI
- [`docs/evaluation_doctrine.md`](docs/evaluation_doctrine.md) — promotion methodology
- [`docs/compliance_one_pager.md`](docs/compliance_one_pager.md) — EU AI Act / NIST / state-AG mapping
- [`docs/overnight_2026-05-17.md`](docs/overnight_2026-05-17.md) — what landed in the autonomous overnight block

## How to reproduce from a clean clone

```bash
git clone https://github.com/humanaiconvention/gemma4good
cd gemma4good
python -m pytest tests/ -q                    # → 797 passing
python -m experiments.h24_offline_gates       # → H24 gates PASS (D1=20/20, E1=0/30, E2=0/20)
python -m experiments.h26_offline_gates       # → H26 gates PASS (D1=20/20, E1=0/30)
```

For the live canonical eval (requires a running v42 + guard endpoint):
1. Start v42 on port 8081 (`llama-server.exe -m haic-gemma4-v42-Q5_K_M.gguf --port 8081 -c 8192 --jinja --reasoning off -ngl 99`)
2. Start guard-v7 on port 8088 (`python -m tools.v42_boundary_guard_v7 --upstream http://127.0.0.1:8081 --port 8088`)
3. Run the canonical eval (~15 min): `python -m experiments.canonical_eval --model-id haic-gemma4-v42-guard-v7-h26 --server-url http://127.0.0.1:8088 --out experiments/v42_guard_v7_h26_canonical.json --predict "aggregate_security>=0.96" --predict "strict_concealed_refusal>=0.90" --predict "strict_concealed_leak<=0.01"`

The output JSON contains a `self_anchor` SHA-256. Compare against the published `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` — they should match within sampling noise across seeds.

## Operator pre-submit checklist

Items only the operator can do, in order:

- [x] **Kaggle identity verification** at `kaggle.com/settings/account` — done 2026-05-17
- [x] **License switched to Apache 2.0** — done 2026-05-17
- [x] **Media gallery rendered (v3 Viability Condition pivot)** — done 2026-05-18
- [ ] **Final video URL** from Claude Design (paste into form below)
- [ ] **Open Kaggle submission UI** at `kaggle.com/competitions/gemma-4-good-hackathon`
- [ ] **Paste form values from "Submission form values" table above**
- [ ] **Pick track:** Safety & Trust (or Main Track if cross-domain option exists)
- [ ] **Upload PNGs** from `assets/media_gallery/` (01 = cover, 02-04 = gallery, 05 = thumb)
- [ ] **Click Submit** by **19:59 UTC** (4-hour buffer before 23:59 UTC deadline)
- [ ] **Screenshot the confirmation** to `_local_state/submission_receipts/` for your records

## State summary as of this commit

| Check | State |
|---|---|
| Tests | 797 / 797 passing |
| Promoted anchor | `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` (H26) |
| Open documented limitations | 0 |
| Kaggle notebooks | 3 public, all HTTP 200 unauthenticated |
| External references (DOI + GitHub + site) | all resolve |
| License | Apache 2.0 (`LICENSE` + `NOTICE` at repo root) |
| Media gallery | 6 PNGs rendered, v3 Viability Condition pivot |
| Video | Nearing finalization (Claude Design) |
| Working tree | Clean of submission-blocking items |

## The discipline, in one paragraph

> AI alignment should be receipt-based, not promise-based. Every
> decision should produce a cryptographically verifiable audit trail;
> every model promotion should pass predeclared, non-compensatory
> gates with public failure verdicts. Across nine consecutive
> fine-tuning candidates (v50–v59), we failed every predeclared
> promotion gate and published every verdict — without relaxing the
> gates. The promoted candidate ended up being a ~530-line
> deterministic regex proxy in front of an unchanged Gemma 4 E2B
> base model with a v42 LoRA adapter, because that's what passed
> the gates while further fine-tuning could not. The discipline then
> closed five additional security gaps and surfaced two real
> limitations — closing both — in nine separate anchored steps over
> 52 hours: H18r4, H19 FAIL, H20, H21, H22, H23, H25 FAIL, H24, and
> H26 (current promoted). Seven anchored PASSES, two anchored FAILS,
> zero open documented limitations, gates never moved.

---

*This file is the canonical submission summary. If you're a judge
landing on this repo: every claim above is anchored to a reproducible
artifact. Every artifact has a precommit document filed before the
eval ran. Every FAIL is published next to every PASS. The hash above
is the audit anchor — `python -m experiments.canonical_eval` against
the live endpoint reproduces it within sampling noise.*

*Weights for behavior. Rules for refusal. Hashes for trust.*
