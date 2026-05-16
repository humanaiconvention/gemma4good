# Video production — file index

Everything for the 2:30 HAIC Gemma 4 Good submission video. Open these in the order below.

## Read first
- [`..\docs\video_production_brief_v1.md`](../docs/video_production_brief_v1.md) — full brief, locked script, shot list, mix levels.

## Record
- [`record.md`](record.md) — recording cheat sheet for the second screen.
- [`teleprompter.html`](teleprompter.html) — open in browser; auto-scroll button; size slider.
- [`take_rubric.md`](take_rubric.md) — how to pick between two takes in under 30 s.
- Drop chosen takes in `vo\` as `seg_01.wav` … `seg_06.wav`.

## Look
- [`graphics\preview.html`](graphics/preview.html) — every shot stacked top to bottom.
- [`graphics\shot_*.svg`](graphics/) — individual shot frames at 1920×1080.
- [`graphics\thumbnail_1280x720.svg`](graphics/thumbnail_1280x720.svg) — YouTube thumbnail.

## Build
- [`assemble.py`](assemble.py) — lo-fi ffmpeg pipeline. Hard cuts, no motion. Run after VO is recorded.
  - `python assemble.py` (defaults to repo-relative paths)
  - `python assemble.py --kaggle-capture kaggle_repro.mkv` if you've recorded A4.
- [`animation_spec.json`](animation_spec.json) — frame-accurate keyframe spec for After Effects (if doing a polished cut).
- [`ae_chip_stagger.jsx`](ae_chip_stagger.jsx) — AE script that staggers the FAIL chip layers automatically.

## Ship
- [`captions_locked_draft.srt`](captions_locked_draft.srt) — 27-cue SRT against the locked script; re-time after recording.
- [`music_candidates.md`](music_candidates.md) — three places to find a music bed, plus the silent fallback.
- [`preflight_checklist.md`](preflight_checklist.md) — Day 1 + Day 2 verification.

## Deadline
2026-05-18 23:59 UTC.

## Order of operations
1. Read the brief.
2. Walk preflight (Day 1 section).
3. Open teleprompter + record.md. Record six segments, two takes each, drop in `vo\`.
4. Pick takes via take_rubric.md. Rename winners to `seg_NN.wav`.
5. (Optional) Record the Kaggle screen capture as `kaggle_repro.mkv`.
6. Pick a music bed from music_candidates.md (or skip).
7. `python assemble.py [--kaggle-capture kaggle_repro.mkv]`.
8. Watch `out\haic_gemma4_good.mp4`. Iterate as needed.
9. Walk preflight (Day 2 section).
10. Upload to `@HumanAIConvention` per brief §8.
11. Submit to Kaggle.
