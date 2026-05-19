# Pre-submission Verification — 2026-05-16 (late evening)

*Final verification pass before the Kaggle Gemma 4 Good deadline
(2026-05-18 23:59 UTC). Supersedes the prior 2026-05-14 report by
recording the post-H22 / H23 / H24 / H25 state. Post-submission aftercare
updated this document to point at the final H26 submitted candidate; see
`docs/submission_manifest_2026-05-18.md` for the frozen submission boundary.*

---

## Promoted candidate

| Field | Value |
|---|---|
| Endpoint | `guard-v7 + v42` (H26, 2026-05-17) |
| Canonical anchor | `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` |
| Base model | Gemma 4 E2B (weights unchanged across all 7 promoted generations) |
| Guard implementation | `tools/v42_boundary_guard_v7.py` |
| Verdict | `docs/h26_verdict_2026-05-17.md` |
| Precommit | `docs/h26_precommit_hypothesis_2026-05-17.md` |

## H-series record (final state for submission)

```
H18r4  PASS  ASCII baseline                     anchor 18e2c5a5...
H19    FAIL  honestly published                  -
H20    PASS  Unicode bypass (L-01) closed        anchor 56ce960993f9...
H21    PASS  multi-message scan (L-02) closed    anchor d916ef63...
H22    PASS  system-role (L-02b) closed          anchor 5f2e796cf5af...
H23    PASS  encoded-payload behavioral defense  (L-08 surfaced at 1/20)
H25    FAIL  native-language attack confirmed    (L-09 surfaced)
H24    PASS  leet-fold closes L-08               anchor eb61ebc7c0fe...
H26    PASS  multi-language closes L-09          anchor 4d0d7bf05ea2...  ← promoted
```

**Seven PASSES, two FAILS, zero gate relaxations, two limitations
surfaced and BOTH closed in-cycle (L-08 in H24, L-09 in H26).**

## Open limitations on the promoted candidate

**ZERO.**

All items in `docs/v42_guard_known_limitations_2026-05-15.md`
(L-01 through L-09) are either closed via an anchored H-series PASS,
documented as intentional, routed to the correct architectural layer,
or removed.

---

## Verification checklist

### 1. Repo state

| Check | Result |
|---|---|
| Tests passing | **797 / 797** (`python -m pytest tests/ -q`) |
| Local branch | `main` |
| Local clean | clean at submitted snapshot, before post-submission aftercare commits |
| Origin | `origin/main` up-to-date through submitted commit `ec7db2e`, including H26 and media assets |
| License | `LICENSE` + `NOTICE` present at repo root, SPDX-License-Identifier: **Apache-2.0** (switched from CC0-1.0 on 2026-05-16 for hackathon alignment) |

### 2. Kaggle notebooks (public-visibility re-check)

| Notebook | URL | HTTP |
|---|---|---|
| Main submission | `https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent` | **200** |
| Tier 3 live validation | `https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation` | **200** |
| Reproducibility demo | `https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4` | **200** |

All three notebooks return HTTP 200 unauthenticated. The 2026-05-15 fix
to `is_private: false` on the Tier 3 notebook held.

### 3. External references (DOI + repos + site)

| URL | Resolved |
|---|---|
| https://doi.org/10.5281/zenodo.18144681 (Viability Condition) | 200 (zenodo.org/records/18144681) |
| https://doi.org/10.5281/zenodo.15302994 (Alignment paper) | 200 (zenodo.org/records/15302994) |
| https://github.com/humanaiconvention/gemma4good | 200 |
| https://humanaiconvention.com | 200 |

### 4. WRITEUP fidelity

`WRITEUP.md` surfaces H26 as the submitted promoted candidate alongside
the full H18r4 → H20 → H21 → H22 → H23 → H25 → H24 → H26 anchored chain.
Both FAIL hypotheses (H19, H25) are referenced. The 30-second version,
the H-series narrative section, and the candidate-state paragraph all
reference the H26 anchor consistently.

### 5. Artifact integrity (H23/H24/H25/H26)

| Artifact | Path | Status |
|---|---|---|
| H23 precommit | `docs/h23_precommit_hypothesis_2026-05-16.md` | committed |
| H23 verdict | `docs/h23_verdict_2026-05-16.md` | committed |
| H23 attack suite | `experiments/h23_encoded_attack_suite.jsonl` | committed |
| H23 results | `experiments/h23_results_2026-05-16.json` | committed |
| H25 precommit | `docs/h25_precommit_hypothesis_2026-05-16.md` | committed |
| H25 verdict | `docs/h25_verdict_2026-05-16.md` | committed |
| H25 attack/benign suites | `experiments/h25_native_lang_attack_suite.jsonl`, `experiments/h25_native_lang_benign_suite.jsonl` | committed |
| H25 results | `experiments/h25_results_2026-05-16.json` | committed |
| H24 precommit | `docs/h24_precommit_hypothesis_2026-05-16.md` | committed |
| H24 verdict | `docs/h24_verdict_2026-05-16.md` | committed |
| H24 attack suite | `experiments/h24_leet_attack_suite.jsonl` | committed |
| H24 benign-digits suite | `experiments/h24_benign_digits_suite.jsonl` | committed |
| H24 suite validator | `experiments/h24_validate_suites.py` | committed |
| H24 offline gate runner | `experiments/h24_offline_gates.py` | committed |
| H24 offline results | `experiments/h24_offline_results_2026-05-16.json` | committed |
| H24 canonical eval | `experiments/v42_guard_v6_h24_canonical.json` | committed |
| H24 failure sidecar | `experiments/v42_guard_v6_h24_failures.jsonl` | committed |
| Guard-v6 implementation | `tools/v42_boundary_guard_v6.py` | committed |
| Known-limitations doc | `docs/v42_guard_known_limitations_2026-05-15.md` | L-08 closure recorded |
| H26 precommit | `docs/h26_precommit_hypothesis_2026-05-17.md` | committed |
| H26 verdict | `docs/h26_verdict_2026-05-17.md` | committed |
| H26 live results | `experiments/h26_results_2026-05-17.json` | committed |
| H26 canonical eval | `experiments/v42_guard_v7_h26_canonical.json` | committed |
| H26 failure sidecar | `experiments/v42_guard_v7_h26_failures.jsonl` | committed |
| Guard-v7 implementation | `tools/v42_boundary_guard_v7.py` | committed |

### 6. Submission checklist

| Item | Status | Notes |
|---|---|---|
| Repo on GitHub | ✓ | https://github.com/humanaiconvention/gemma4good |
| License file present | ✓ | Apache-2.0 at repo root + NOTICE file |
| WRITEUP up to date | ✓ | H26 anchor + full chain |
| Test suite green | ✓ | 797/797 |
| Promoted candidate documented | ✓ | H26 verdict at `docs/h26_verdict_2026-05-17.md` |
| All H-series verdicts published | ✓ | Including the two anchored FAILS (H19, H25) |
| Open limitations published | ✓ | Zero open documented limitations as of H26 |
| Discipline essay | ✓ | `docs/discipline_is_the_contribution.md` |
| Compliance one-pager | ✓ | `docs/compliance_one_pager.md` |
| Kaggle main notebook | ✓ | HTTP 200, v19 COMPLETE |
| Kaggle Tier 3 notebook | ✓ | HTTP 200, v12 COMPLETE, public |
| Kaggle reproducibility demo | ✓ | HTTP 200 |
| External DOIs resolve | ✓ | Viability + Alignment both 200 after redirect |
| Identity verification on Kaggle account | **operator-owned** | Cannot be verified by this agent; flagged for operator confirmation before submit |

---

## Items NOT verified by this agent at the time (operator-owned)

These require operator action and cannot be checked from this session:

1. **Kaggle account identity verification.** Required for prize
   eligibility. The operator must confirm `kaggle.com/settings`
   shows verified identity before final submit.
2. **Video production.** Later completed and submitted with URL
   `https://youtu.be/p5ZprNkIAEM`.
3. **Kaggle competition rules re-read.** A final operator pass through
   the rules page is recommended for any last-mile constraint
   (model-size, dataset-license, write-up-length) that may have been
   updated since the 2026-05-14 review.
4. **Final notebook re-run on submission day.** If notebook code or
   reference receipts changed, a re-run pre-deadline is prudent.

---

## What this submission claims, in one paragraph

The promoted candidate (`guard-v7 + v42`, anchor `4d0d7bf05ea2…`) is
an unchanged Gemma 4 E2B base model fronted by a ~330-line
deterministic FastAPI proxy. The proxy enforces 27 regex rules (16
English + 11 native-language) across 4 attack classes (DIRECT_INJECT,
CONCEALED_COMPLY, PROTO_DISCLOSE, JAILBREAK) over a Unicode-normalized
+ zero-width-stripped + homoglyph-folded + leet-folded matching
surface PLUS a pre-fold multi-language surface (Japanese, Korean,
Chinese, Russian, Spanish, French, German, Portuguese, Arabic, Hindi,
Hebrew). The guard scans every user message in the history and rejects
client-supplied `role: system` injections at any non-first position.
Every gate that promoted it was predeclared in a precommit document;
every FAIL was published with its precommit; the canonical eval is
reproducible from a clean clone in under 15 minutes. The discipline
produced 7 PASSES, 2 FAILS, 0 gate relaxations, and surfaced 2 real
attack-class limitations across the H-series — **both of which we
closed in-cycle with predeclared non-compensatory gates**.

---

## Reference

- Promoted-candidate verdict: `docs/h24_verdict_2026-05-16.md`
- Full known-limitations: `docs/v42_guard_known_limitations_2026-05-15.md`
- Prior verification report (superseded): `docs/submission_verification_report_2026-05-14.md`
- Discipline essay: `docs/discipline_is_the_contribution.md`
- Compliance one-pager: `docs/compliance_one_pager.md`
- Evaluation doctrine: `docs/evaluation_doctrine.md`
