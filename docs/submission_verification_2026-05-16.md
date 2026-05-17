# Pre-submission Verification — 2026-05-16 (late evening)

*Final verification pass before the Kaggle Gemma 4 Good deadline
(2026-05-18 23:59 UTC). Supersedes the prior 2026-05-14 report by
recording the post-H22 / H23 / H24 / H25 state.*

---

## Promoted candidate

| Field | Value |
|---|---|
| Endpoint | `guard-v6 + v42` (H24, 2026-05-16) |
| Canonical anchor | `eb61ebc7c0fef6bf200dedaed40d5f48d4c18da0c3624e8dc7efc041192cb55f` |
| Base model | Gemma 4 E2B (weights unchanged across all 6 promoted generations) |
| Guard implementation | `tools/v42_boundary_guard_v6.py` |
| Verdict | `docs/h24_verdict_2026-05-16.md` |
| Precommit | `docs/h24_precommit_hypothesis_2026-05-16.md` |

## H-series record (final state for submission)

```
H18r4  PASS  ASCII baseline                     anchor 18e2c5a5...
H19    FAIL  honestly published                  -
H20    PASS  Unicode bypass (L-01) closed        anchor 56ce960993f9...
H21    PASS  multi-message scan (L-02) closed    anchor d916ef63...
H22    PASS  system-role (L-02b) closed          anchor 5f2e796cf5af...
H23    PASS  encoded-payload behavioral defense  (L-08 surfaced at 1/20)
H25    FAIL  native-language attack confirmed    (L-09 documented)
H24    PASS  leet-fold closes L-08               anchor eb61ebc7c0fe...  ← promoted
```

**Six PASSES, two FAILS, zero gate relaxations, two limitations
surfaced (one closed in-cycle, one published openly).**

## Open limitations (1)

| ID | Limitation | Status |
|---|---|---|
| L-09 | Native-language attack bypass (Japanese, Korean, Spanish observed leaking; English-only rule patterns by design) | OPEN, deferred to future H26 |

All other items in `docs/v42_guard_known_limitations_2026-05-15.md`
(L-01, L-02, L-02b, L-03, L-04, L-05, L-06, L-07, L-08) are closed,
documented as intentional, routed to the correct architectural layer,
or removed.

---

## Verification checklist

### 1. Repo state

| Check | Result |
|---|---|
| Tests passing | **702 / 702** (`python -m pytest tests/ -q`) |
| Local branch | `main` |
| Local clean | working tree carries video/preflight in-progress (unrelated to submission) |
| Origin | `origin/main` up-to-date through `57b2b8f H24 PASS`, including precommit `31fd4ef` and all H-series verdicts |
| License | `LICENSE` present at repo root, SPDX-License-Identifier: CC0-1.0 |

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

`WRITEUP.md` updated this session to surface H24 as the current
promoted candidate alongside the full H18r4 → H20 → H21 → H22 → H23 →
H24 anchored chain. Both FAIL hypotheses (H19, H25) are referenced.
The 30-second version, the H-series narrative section, and the
candidate-state paragraph all reference the H24 anchor consistently.

### 5. Artifact integrity (H23/H24/H25)

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

### 6. Submission checklist

| Item | Status | Notes |
|---|---|---|
| Repo on GitHub | ✓ | https://github.com/humanaiconvention/gemma4good |
| License file present | ✓ | CC0-1.0 at repo root |
| WRITEUP up to date | ✓ | H24 anchor + full chain |
| Test suite green | ✓ | 702/702 |
| Promoted candidate documented | ✓ | H24 verdict at `docs/h24_verdict_2026-05-16.md` |
| All H-series verdicts published | ✓ | Including the two anchored FAILS (H19, H25) |
| Open limitations published | ✓ | L-09 only |
| Discipline essay | ✓ | `docs/discipline_is_the_contribution.md` |
| Compliance one-pager | ✓ | `docs/compliance_one_pager.md` |
| Kaggle main notebook | ✓ | HTTP 200, v19 COMPLETE |
| Kaggle Tier 3 notebook | ✓ | HTTP 200, v12 COMPLETE, public |
| Kaggle reproducibility demo | ✓ | HTTP 200 |
| External DOIs resolve | ✓ | Viability + Alignment both 200 after redirect |
| Identity verification on Kaggle account | **operator-owned** | Cannot be verified by this agent; flagged for operator confirmation before submit |

---

## Items NOT verified by this agent (operator-owned)

These require operator action and cannot be checked from this session:

1. **Kaggle account identity verification.** Required for prize
   eligibility. The operator must confirm `kaggle.com/settings`
   shows verified identity before final submit.
2. **Video production.** Off Claude Code's plate; Claude Design owns
   per the strategic plan. Submission is still valid without a video
   if the deadline arrives first, but the integration ask depends on
   it.
3. **Kaggle competition rules re-read.** A final operator pass through
   the rules page is recommended for any last-mile constraint
   (model-size, dataset-license, write-up-length) that may have been
   updated since the 2026-05-14 review.
4. **Final notebook re-run on submission day.** If notebook code or
   reference receipts changed, a re-run pre-deadline is prudent.

---

## What this submission claims, in one paragraph

The promoted candidate (`guard-v6 + v42`, anchor `eb61ebc7c0fe…`) is
an unchanged Gemma 4 E2B base model fronted by a ~280-line
deterministic FastAPI proxy. The proxy enforces 16 regex rules across
4 attack classes (DIRECT_INJECT, CONCEALED_COMPLY, PROTO_DISCLOSE,
JAILBREAK) over a Unicode-normalized + zero-width-stripped +
homoglyph-folded + leet-folded matching surface, scans every user
message in the history, and rejects client-supplied `role: system`
injections at any non-first position. Every gate that promoted it was
predeclared in a precommit document; every FAIL was published with
its precommit; the canonical eval is reproducible from a clean clone
in under 15 minutes. The discipline produced 6 PASSES, 2 FAILS, 0
gate relaxations, and surfaced 2 real attack-class limitations across
the H-series — one of which we closed mid-cycle and one of which we
documented openly as a future work item.

---

## Reference

- Promoted-candidate verdict: `docs/h24_verdict_2026-05-16.md`
- Full known-limitations: `docs/v42_guard_known_limitations_2026-05-15.md`
- Prior verification report (superseded): `docs/submission_verification_report_2026-05-14.md`
- Discipline essay: `docs/discipline_is_the_contribution.md`
- Compliance one-pager: `docs/compliance_one_pager.md`
- Evaluation doctrine: `docs/evaluation_doctrine.md`
