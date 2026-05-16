# Preflight — verify before recording starts (Day 1) and before upload (Day 2)

Every load-bearing claim in the video must be true on the day of upload. This is the script you walk through twice — once before recording, once before the upload goes Public.

## Day 1 — 09:00 — Before recording

### Numbers and strings
- [ ] **Canonical anchor matches the repo exactly.**
      Open `D:\gemma4good\docs\v42_guard_h18r4_verdict_2026-05-15.md`. Confirm the anchor is `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`. Compare character-by-character to:
      - `graphics\shot_09_split_proof.svg`
      - `graphics\thumbnail_1280x720.svg`
      - `record.md` (no anchor; just the line that mentions it)
      - any other surface
- [ ] **Gate count = 13.** Verdict doc shows H18a through H18m. Confirm.
- [ ] **Failed candidate count = 10.** Verdict files v50…v59 all on disk in `docs\`. Confirm.
- [ ] **Guard line count ≈ 200.** Open `D:\gemma4good\tools\v42_boundary_guard.py`. If it has drifted significantly, update the script wording or the code.
- [ ] **Guard rule count = 16, attack class count = 4.** Verdict doc H18r4 §Evaluation Artifact line: "16 rules, 4 classes." Confirm.
- [ ] **Repo test count = 679.** Run `pytest --collect-only -q` and count. Update WRITEUP and YouTube description if different.

### URLs resolve
Open each in an Incognito window. Each must return a non-error page.
- [ ] https://github.com/humanaiconvention/gemma4good
- [ ] https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent
- [ ] https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation
- [ ] https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4
- [ ] https://github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md
- [ ] https://humanaiconvention.com
- [ ] https://doi.org/10.5281/zenodo.18144681

### Reproducibility kernel is current
- [ ] Open the reproducibility kernel. The `Run All` cells complete cleanly. The final SHA3 receipt printed matches the anchor in the verdict doc.
- [ ] If it doesn't, fix the kernel BEFORE recording the screen-capture (A4).

### Files staged
- [ ] `D:\gemma4good\video\graphics\` contains 9 SVG files plus `preview.html`.
- [ ] `D:\gemma4good\video\vo\` exists and is empty.
- [ ] `D:\gemma4good\video\teleprompter.html` opens in your browser and scrolls.
- [ ] `D:\gemma4good\video\record.md` is open on a second screen or in another window.

### Recording rig
- [ ] Webcam or phone mic positioned 8–10 inches from your mouth, slightly off-axis (45°).
- [ ] Closet / duvet / room treatment in place.
- [ ] Computer fan is off or quiet. Notifications muted. Phone on Do Not Disturb.
- [ ] 30 s test recording done — listen back, confirm no ticks, no fan whine, no clipping.

---

## Day 2 — Before upload goes Public

### The video itself
- [ ] Run-time is 2:30 ± 3 s.
- [ ] Audio at -16 LUFS integrated, peaks ≤ -1 dBTP. Check in Resolve / Audacity / Loudness Penalty.
- [ ] No audio glitch / pop / breath spike visible in the waveform.
- [ ] Watch end-to-end at 1×. Then at 1.5×. Both reads pass.
- [ ] The SHA3 anchor on screen matches the repo character-by-character (re-check now, even if you did Day 1).
- [ ] The PASS panel reads `13 / 13`. The FAIL panel reads ten chips v50–v59.
- [ ] No spelling mistake in any URL on screen.
- [ ] No spelling mistake in any name (HumanAI Convention, Gemma 4, PRISM, NLA, Merkle, SHA3, Claude, Gemini, GPT).

### YouTube upload — `@HumanAIConvention`
- [ ] Title set exactly: `HumanAI Convention — Verifiable AI Governance, Demonstrated on Gemma 4 (Gemma 4 Good)`.
- [ ] Description pasted (see `docs/video_production_brief_v1.md` §8). All URLs resolve.
- [ ] Captions (SRT) uploaded — `captions_locked_draft.srt` re-timed to the actual recording.
- [ ] Tags set.
- [ ] Category: Science & Technology. Language: English.
- [ ] Thumbnail uploaded (1280×720 PNG export of `thumbnail_1280x720.svg`).
- [ ] End-screen elements added: subscribe (`@HumanAIConvention`), repo card, WRITEUP card.
- [ ] Set Unlisted. Confirm playback works.
- [ ] At T-60 minutes: switch to Public. Or use Schedule Publish set to 2026-05-18 22:00 UTC.

### Kaggle submission
- [ ] Submission form references the YouTube URL (Public).
- [ ] Submission form references the WRITEUP.
- [ ] Final submit hit before 2026-05-18 23:59 UTC.
- [ ] Receipt of submission emailed to you.

---

## What "abort and ship the 90 s cut" looks like

Trigger the fallback if any of these hits 2026-05-18 12:00 UTC:
- [ ] Picture not locked
- [ ] VO segments missing
- [ ] Captions not started
- [ ] Audio mix not done

The 90 s cut uses Segments 1A–6A (same VO discipline, shorter takes). `assemble.py` will need shorter shot durations — open the SHOTS list and trim, then rerun.
