# Pre-submission checklist — Gemma 4 Good Hackathon

**Deadline:** 2026-05-18 23:59 UTC
**Target submission window:** at least 4 hours before deadline (i.e. by 19:59 UTC on May 18) to allow buffer for upload retries, mis-fields, last-minute fixes.

Run this list top-to-bottom on the day of submission. Each item has a
verification command or link. Don't skip — most submission failures
are from one missed checkbox, not from substance.

---

## Decisions to lock in BEFORE opening the Kaggle submission UI

### 1. Primary track selection

The hackathon has five tracks: **Education · Health · Digital Equity · Global Resilience · Safety**.

Our submission spans Health (clinic), Education (classroom), Climate /
Global Resilience (deforestation), and Safety (the boundary guard).
Cannot pick all four.

**Recommendation: Safety as primary track.**

Why: the H18r4 guard + nine published failures is the strongest
evidence in the submission, and the Safety track is where the
discipline argument lands best. The three scenarios remain as breadth
evidence in the writeup.

**Alternative:** "Main Track" if there's an explicit cross-domain
option in the submission UI.

**Final decision:** _______________________________________

### 2. License confirmation — DONE (2026-05-16)

Current repo license: **Apache-2.0** (`LICENSE` + `NOTICE` files at
repo root, SPDX-License-Identifier: Apache-2.0).

Switched from CC0-1.0 on 2026-05-16 after the rules re-read confirmed
multiple secondary sources describe submissions as released under
Apache 2.0 (matching Gemma 4's own license). CC0 was strictly more
permissive than Apache 2.0, so the transition narrows the license but
adds patent grants and attribution — both desirable.

- [x] LICENSE replaced with canonical Apache 2.0 text
- [x] NOTICE file added with copyright + transition note
- [x] SPDX header updated to Apache-2.0
- [x] WRITEUP / submission-verification docs updated to reference Apache 2.0

### 3. Video upload + URL ready — DONE (2026-05-18)

- [x] Video rendered (Claude Design)
- [x] Uploaded to YouTube (HTTP 200 unauthenticated, confirmed 2026-05-18)
- [x] **YouTube URL: https://youtu.be/p5ZprNkIAEM**
      (canonical: https://www.youtube.com/watch?v=p5ZprNkIAEM)

### 4. Cover image + media gallery assets ready

Per `docs/media_gallery_image_specs.md`:

- [ ] Cover image (1200×630 px)
- [ ] Architecture diagram
- [ ] Guard decision flow visual
- [ ] H18r4 verdict screenshot

### 5. Identity verification on Kaggle account

- [ ] Confirm `benhaslam` Kaggle profile has identity verification badge
- [ ] If not done, complete via https://www.kaggle.com/settings/account

---

## Final repo state pre-submission

Run these commands. All should succeed.

```bash
cd D:/gemma4good

# 1. Working tree clean
git status                                        # → nothing to commit

# 2. On main, up to date with origin
git log --oneline -1 origin/main..HEAD            # → empty (no unpushed commits)

# 3. Tests pass
python -m pytest tests/ -q                        # → 679+ passed

# 4. Submission notebook still runs (smoke check)
python -c "import json; nb = json.load(open('notebook/haic_gemma4_governance.ipynb')); print(f'cells: {len(nb[\"cells\"])}')"

# 5. License file present
head -1 LICENSE

# 6. WRITEUP.md present, has 30-second-version section
head -25 WRITEUP.md
```

- [ ] Working tree clean
- [ ] No unpushed commits
- [ ] Tests pass (679+)
- [ ] Notebook still parses
- [ ] LICENSE file present
- [ ] WRITEUP has 30-second-version above the fold

---

## Verify the three public Kaggle notebooks are accessible

Open in an incognito window (no Kaggle login):

- [ ] **Main notebook:** https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent — should return HTTP 200, status COMPLETE, public.
- [ ] **Tier 3 live validation:** https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation — should be public.
- [ ] **Reproducibility demo:** https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4 — should be public.

If any return 404 in incognito, the kernel is private. Fix on Kaggle
before submitting.

---

## Public repo state

- [ ] https://github.com/humanaiconvention/gemma4good is public (not 404)
- [ ] README.md displays correctly on GitHub
- [ ] WRITEUP.md displays correctly on GitHub
- [ ] LICENSE file present in repo root
- [ ] No `.env` or secret in the repo (re-check: `git ls-files | grep -E '\.env$|credentials|secret'` returns nothing tracked)

---

## Submission form (when you open the Kaggle UI)

The Kaggle hackathon submission flow likely has these fields. Have answers ready:

| Field | Value (paste from below) |
|---|---|
| Title | `HumanAI Convention — Verifiable Governance for Gemma 4` |
| Track | (your decision from item 1) |
| Short description | (200 chars) `A cryptographically auditable governance loop for Gemma 4 — every decision produces a Merkle-anchored receipt; nine consecutive fine-tunes failed predeclared gates and we published every verdict.` |
| Long description / writeup | Use the WRITEUP.md content directly, or link to the repo's WRITEUP.md if the form allows links |
| Repo URL | https://github.com/humanaiconvention/gemma4good |
| Video URL | (from item 3) |
| Live demo / notebook URL | https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4 (1-minute repro) |
| Cover image | (from item 4) |
| Media gallery | (from item 4 — 3 additional images) |
| License | (from item 2) |

---

## Final pre-submit sanity check (5 min)

Right before hitting Submit:

- [ ] Watch the video one more time, all the way through, with sound. Catch typos in slides, weird voiceover pauses, broken cuts.
- [ ] Open the Kaggle submission preview if available. Read every field as a judge would.
- [ ] Click every link in the form. If any 404, fix before submitting.
- [ ] Confirm the deadline timezone: **23:59 UTC**, not local time. UTC is currently the relevant reference. Don't get caught by 23:59 PT thinking it's the deadline.

---

## After submission

- [ ] Take a screenshot of the Kaggle "Submission received" confirmation. Save to `_local_state/submission_receipts/2026-05-18_kaggle_submission.png`.
- [ ] Tweet / LinkedIn post mentioning the submission, linking to the public repo and the video. Use the 30-second-version paragraph as draft text.
- [ ] Send the outreach emails from `_local_state/outreach/` over the next 14 days, ~1 per day, personalized.

---

## If something goes wrong at the wire

- **Video upload fails 30 min before deadline:** Use the Loom backup URL.
- **Kaggle submission form rejects something:** Submit with the minimal required fields filled, then edit after if Kaggle allows. A submission with TBD optional fields is better than no submission.
- **License issue surfaces last-minute:** Switch to Apache 2.0 in one commit (command in item 2). Push. Done in 2 minutes.
- **Notebook becomes inaccessible:** Push a fresh version of the kernel from `D:/kaggle/notebooks/haic-guard-reproducibility/` — `kaggle kernels push` from that dir.

---

**Total budget for this checklist on submission day: 90 minutes maximum.** Most items are 30-second confirmations. The video upload + media gallery are the only items that take real time.

Hit submit by 19:59 UTC. Use the remaining 4 hours for unexpected issues.
