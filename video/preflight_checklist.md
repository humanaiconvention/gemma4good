# Preflight — submission day 2026-05-18

3-act Viability Condition cut. Run-time ~2:30. Deadline 23:59 UTC.

## Before recording (now)

### Numbers and strings — verify against repo
- [ ] **Promoted candidate is H26.** `docs/h26_verdict_2026-05-17.md` header reads "H26 PASSES all four predeclared non-compensatory gates with maximum margin" and "guard-v7 + v42 becomes the new promoted live candidate."
- [ ] **Canonical anchor.** Verdict doc line: `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8`. Compare character-by-character to:
      - `assets/media_gallery/01_cover.png`
      - `assets/media_gallery/04_h_series_record.png`
      - `assets/media_gallery/05_video_thumb.png`
- [ ] **H-series counts.** v50 → v59 verdicts NOT referenced in this cut. H18 → H26 = 9 hypotheses. PASS: H18, H20, H21, H22, H23, H24, H26 = 7. FAIL: H19, H25 = 2.
- [ ] **52-hour window.** First verdict H18 at 2026-05-15 11:25, last verdict H26 at 2026-05-17 early hours. Inside the 52-hour claim.
- [ ] **Guard rule count.** 16 deterministic + 11 multi-language = 27 rules in `tools/v42_boundary_guard_v7.py`. The script says "sixteen deterministic rules, plus eleven multi-language rules added in the last cycle." Confirm both subcounts are accurate.
- [ ] **Limitations closed.** L-08 leetspeak (H24), L-09 native-language (H26). Both must be referenced as closed.

### URLs resolve (Incognito window)
- [ ] https://github.com/humanaiconvention/gemma4good
- [ ] https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent
- [ ] https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation
- [ ] https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4 (or whatever the current reproducibility kernel URL is — confirm)
- [ ] https://humanaiconvention.com
- [ ] https://doi.org/10.5281/zenodo.18144681

### Recording rig
- [ ] Mic 8–10 inches off-axis. Closet / duvet. Computer fan off.
- [ ] 30 s test recording done. No fan whine, no clicks.
- [ ] `record.md` open. `teleprompter.html` open in adjacent window. Arrow keys to scroll.

### Files staged
- [ ] `D:\gemma4good\video\vo\` exists and is empty.
- [ ] `assets/media_gallery/` has six v3 PNGs at 1200×630, 1600×900, 1200×900, 1200×900, 1280×720, 1280×720.
- [ ] v2 archived at `assets/media_gallery/_archive/v2_2026-05-17/`.

---

## After recording, before publish

### The video
- [ ] Run-time 2:30 ± 5 s.
- [ ] VO: ~ -16 LUFS integrated, true peak ≤ -1 dBTP.
- [ ] H26 anchor on screen matches `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` character-by-character.
- [ ] H-series table reads H18 → H26, nine rows, seven green PASS, two red FAIL, H26 row highlighted PROMOTED.
- [ ] No spelling error in any URL or named term (autophagy, PRISM, NLA, Merkle, SHA3, Gemma 4, HumanAI Convention).
- [ ] Captions (SRT) match the actual VO timing.

### YouTube upload (`@HumanAIConvention`)
- [ ] Title: `HumanAI Convention — The Viability Condition for Gemma 4 (Gemma 4 Good)`
- [ ] Description pasted (see brief §8 — update H26 anchor before pasting).
- [ ] Captions SRT uploaded.
- [ ] Thumbnail uploaded: `assets/media_gallery/05_video_thumb.png` (or YouTube variant).
- [ ] End-screen: subscribe (`@HumanAIConvention`), repo card, WRITEUP card.
- [ ] Set Unlisted. Watch end-to-end at 1×.
- [ ] T-60 min: switch to Public. Or use Schedule Publish 22:00 UTC.

### Kaggle submission
- [ ] Submission form references the YouTube URL (Public).
- [ ] WRITEUP referenced. WRITEUP references the same H26 anchor.
- [ ] Final submit before 2026-05-18 23:59 UTC.

---

## Abort triggers and fallbacks

- If `assemble.py` fails: render each shot SVG to PNG manually via `cairosvg`, build a slideshow in any video editor with the chosen-take WAVs over the top.
- If a critical claim turns out wrong on review: trim the segment that contains it rather than re-recording. Better short than wrong.
- If H26 anchor on disk has changed (another verdict landed overnight): update every surface that shows it. The cover, the H-series record, the thumbnail, the YouTube description, the WRITEUP.
