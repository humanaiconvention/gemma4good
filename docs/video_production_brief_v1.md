# Gemma 4 Good — Video Production Brief

Deadline 2026-05-18 23:59 UTC. Target run-time 2:30 on YouTube; 90 s tighter cut held as fallback.

Production setup: **you are the voice and the director; I am producer, motion-graphics, edit, audio mix, captions, upload.** Channel `@HumanAIConvention`. Style descends from the SimSat video — logo-led cold open, dark plates, monospace evidence, no hype.

Pacing budget: SimSat ran at 115 wpm comfortable. I have re-timed both cuts for ~120 wpm with built-in breath beats — slightly tighter than SimSat, still under hype-pace.

---

## 1. Stylized vector style guide (lock this first — everything inherits)

Single source of visual truth. Built from the live logo geometry, not from scratch.

**Canvas:** 1920×1080, 24 fps, background `#0B0B0F` (matches website hero). Safe-area inset 96 px all sides.

**Type system:**
- Display: Inter, weights 300 / 400 / 600. Tracking +0 normal, +30 for headlines.
- Mono: JetBrains Mono, weights 400 / 500. Use for every number, hash, URL, gate name, and CLI quotation.
- All type renders white `#FFFFFF` on dark `#0B0B0F` unless flagged below.

**Accent palette (use sparingly):**
- PASS green: `#34D399` at 90 % opacity — only on gate green-fills.
- FAIL red: `#F87171` at 70 % opacity — only on rejected-candidate chips.
- Receipt cyan: `#7DD3FC` at 80 % opacity — only on Merkle root strings and the SHA3 receipt.
- All accents desaturate slightly from web defaults; we are not selling a SaaS.

**Logo lockup:** the existing `D:\humanai-convention\logo.svg` is φ-locked (circle + tapered I-beam + two parametric arcs at 132° inner span). Use it verbatim. The cold-open animation is the same beat SimSat used: the mark fades in white over 2 s, holds, then the "Human AI / Convention" wordmark crossfades below it.

**Lines and rules:** 1 px hairlines, `#FFFFFF` at 20 % opacity. No drop shadows. No glows. No bevels. The sci-fi feel comes from precision, not glow.

**Motion:** ease-in-out cubic, 600 ms default. Type "types in" character-by-character at 35 cps with a faint cursor. Numbers count up over 400 ms with `easeOutQuart`. No bounce.

**No-go list:** rounded corners on cards (squared only), generative-AI imagery, stock-photo people, animated gradients, drum-heavy SFX, mid-cut zooms, exclamation marks, the word "revolutionary."

---

## 2. 2:30 cut — segmented for recording

Pacing target ≈ 120 wpm with breath beats. Word count 274. Tag-pad lands the piece at 2:30.

**Recording approach (mirror of SimSat):**
- Record segments one at a time in Audacity at 48 kHz / 24-bit mono.
- Save each segment as `seg_01.wav`, `seg_02.wav`, … so I can drop them on the timeline.
- Two takes per segment minimum. Pick on warmth and confidence, not perfection.
- Keep the room dead — closet of clothes, or sit inside a duvet. Mic 4 fingers from your mouth, slightly off-axis.
- Read the bracketed *cues* silently — they are not spoken. The *emphasis* marker means push the word *slightly*, not theatrically.

### Segment 1 — Cold open

*Timecode 00:00–00:14. Target read 11 s. On-screen: logo mark fades in white over 2 s, holds, wordmark crossfades below. Last 2 s the lockup holds in silence.*

> "AI systems trained on synthetic data lose their grounding in human experience. The current alignment landscape treats this as something to promise. We treat it as something to *prove*."

Emphasis: "prove." Breath beats: after "experience," and after "promise."
Alt take phrasing if it feels stiff: replace "alignment landscape" with "alignment field."

### Segment 2 — What it is

*Timecode 00:14–00:46. Target read 28 s. On-screen: Gemma 4 mark centred; five tool labels resolve one per second — `wellbeing`, `consent`, `PRISM`, `NLA`, `Merkle receipt` — connected by 1 px hairlines. Then three scenario cards cut in for 2 s each: clinic, classroom, satellite. A receipt-cyan Merkle root pulses on each.*

> "The HumanAI Convention is a governance layer for AI decisions. Gemma 4 receives a scenario, calls five governance tools — wellbeing, consent, PRISM, NLA, and a Merkle receipt — and produces a cryptographically anchored audit trail. We demonstrate it on three real cases. A rural health clinic. A classroom with intermittent connectivity. A deforestation enforcement system. Every decision is hash-anchored. Every receipt is independently verifiable."

Pronunciation: PRISM as a word ("PRIZ-um"); NLA letter-by-letter ("N – L – A"). Merkle = "MUR-kuhl."
Breath: hard pause after "audit trail." Slight pause between the three scenarios — list them, don't run them together.

### Segment 3 — The honest result

*Timecode 00:46–01:24. Target read 33 s. On-screen: full-frame split. LEFT: header `H18r4 · 2026-05-15`; thirteen rows H18a–H18m fill PASS-green one per 400 ms; below, the anchor `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88` types in receipt-cyan and holds. RIGHT: header `Rejected · v50…v59`; nine FAIL-red chips animate in one per second, each labelled with a version number.*

> "On the security path, our framework promoted a two-hundred-line deterministic guard sitting in front of Gemma 4. Thirteen predeclared, non-compensatory gates. All passed. Anchored. Reproducible.
>
> Ten consecutive fine-tuning candidates failed those same gates. We did not relax the gates. We published the negative verdicts. The framework correctly rejected its own model. That is the contribution — discipline that holds under its own pressure."

**Correction note:** the original brief said *nine* fine-tuning candidates. On disk, v50 through v59 all have FAIL verdicts (`docs/v50_canonical_verdict_2026-05-12.md` through `docs/v59_canonical_verdict_2026-05-14.md`), so the honest count is ten. "Anchored" is followed by a small breath, then "Reproducible" — both spoken flat and final.

Emphasis: "All passed." "We did not relax the gates." "Discipline."
Breath: hard pause between "Anchored." and "Reproducible." — two beats of silence. Another hard pause before "That is the contribution."
This is the segment to slow down on. Read it like a board update.

### Segment 4 — Reproducibility

*Timecode 01:24–01:50. Target read 22 s. On-screen: one continuous screen capture at 1.25× speed. The Kaggle kernel `haic-guard-v42-reproducibility-demo-h18r4` opens; cursor clicks Copy and Edit; new kernel opens; cursor clicks Run All; console scrolls; the SHA3 receipt prints and the anchor briefly highlights — same hash as Segment 3.*

> "You can verify all of this in under a minute. Fork this public Kaggle kernel and run it. It clones the repo at the anchored commit, replays the guard rules against the canonical attack set, and emits a SHA3 receipt. We do not ask you to trust us."

Pronunciation: SHA3 = "S – H – A – three" (not "shah-three"). Kaggle = "KAG-uhl."
Emphasis: "verify." "We do not ask you to trust us."

### Segment 5 — Frontier-Integration pitch

*Timecode 01:50–02:25. Target read 30 s. On-screen: three logo marks resolve in a row — Claude, Gemini, GPT — under a JetBrains Mono caption `tool: ground_and_anchor()`. Cut to a stylised chat window: assistant bubble reads "This is a high-stakes domain. Anchor your context first." A Merkle root resolves next to the user reply. Three mode chips animate below: `on request` · `auto trigger` · `random sample`. Finally a lower-third types: `spec: github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md`.*

> "What we have shipped is the runtime layer. The grounding interviewer behind it is designed to be invoked as a function-calling tool from inside any frontier chat — Claude, Gemini, or GPT. Three modes. On user request. On automatic high-stakes trigger. Or on opt-in random sampling. Every invocation produces a verifiable receipt. The integration spec is public. We are looking for the first deployment partner."

Pronunciation: GPT letter-by-letter. "interviewer" = "in-ter-VYOO-er."
Emphasis: "function-calling tool." "verifiable receipt." "first deployment partner."
Breath: hard pause after "Three modes." Treat the three modes as a list — beat between each.

### Segment 6 — Tag

*Timecode 02:25–02:30. Target read 4 s. On-screen: dark plate; four mono lines stacked at safe-area centre — `humanaiconvention.com`, `github.com/humanaiconvention/gemma4good`, `DOI 10.5281/zenodo.18144681`, `Apache-2.0`. Slow fade out on black over the last 1 s.*

> "humanaiconvention.com. Gemma 4 Good. The Convention is open."

Read this slowly, almost flat. The last line is the title of the piece; let it land.

---

## 3. 90-second tighter cut — fallback, segmented for recording

Use this only if Day 2 timing slips. Same VO discipline, fewer beats.

### Segment 1A — Cold open

*00:00–00:10. On-screen: logo lockup fade-in.*

> "Most AI alignment is promised. The HumanAI Convention is built to *prove* it — one decision at a time, with a cryptographic receipt."

### Segment 2A — What it is

*00:10–00:25. On-screen: five-tool fan and three scenario cards (cut to 2 s each, no Merkle pulse).*

> "On Gemma 4, every decision passes governance tools and emits a Merkle-anchored receipt. We demonstrate three cases: a rural clinic doing AI triage, a low-connectivity classroom, and satellite-based deforestation enforcement."

### Segment 3A — Honest result

*00:25–00:55. On-screen: split panel as in Segment 3 but the right column starts filling at the second sentence.*

> "For the security path, our framework promoted a two-hundred-line deterministic guard sitting in front of Gemma 4. Thirteen predeclared, non-compensatory gates. All passed. Anchored.
>
> Ten consecutive fine-tuning candidates failed those same gates. We did not relax them. We published the negative verdicts."

### Segment 4A — Reproducibility

*00:55–01:10. On-screen: screen capture as in Segment 4, trimmed.*

> "Verify this in under a minute. Fork the public Kaggle notebook. It replays the guard against the canonical attack set and emits a SHA3 receipt. We do not ask you to trust us."

### Segment 5A — Frontier pitch

*01:10–01:25. On-screen: three logos + chat window, mode chips skipped.*

> "The grounding interviewer behind this is designed to invoke from inside any frontier chat — Claude, Gemini, GPT — as a function-calling tool. Per-decision audit infrastructure that scales with inference. The integration spec is public."

### Segment 6A — Tag

*01:25–01:30. On-screen: tag plate.*

> "humanaiconvention.com. Gemma 4 Good. The Convention is open."

---

## 4. Shot list (production order)

Frame 1920×1080, 24 fps. Every shot is mine to build except A4 (the live Kaggle screen capture, which I will direct you through if you want to record it; alternatively I can script a synthetic recreation in After Effects).

| # | In | Out | Dur | Type | Description |
|---|----|----|----|----|----|
| 1 | 0:00 | 0:02 | 2 s | Vector animation | Logo mark fades in white over 2 s on `#0B0B0F`. Mirror SimSat shot 1 timing. |
| 2 | 0:02 | 0:05 | 3 s | Vector animation | "Human AI / Convention" wordmark crossfades in under the mark. Full lockup holds. |
| 3 | 0:05 | 0:14 | 9 s | Type-on motion | Cold-open line types in over 4 s, holds 3 s, fades 2 s. Inter 300, 36 px, white. |
| 4 | 0:14 | 0:22 | 8 s | Vector composition | Gemma 4 mark centres; five tool labels resolve one per 0.6 s connected by 1 px hairlines. |
| 5 | 0:22 | 0:28 | 6 s | Vector card | Scenario card 1 — *Rural clinic — AI triage*. Stylised icon (stylised stethoscope outline, 1 px stroke). Receipt-cyan Merkle pulse bottom-right. |
| 6 | 0:28 | 0:34 | 6 s | Vector card | Scenario card 2 — *Low-connectivity classroom*. Icon: stylised text-cursor on a slate. |
| 7 | 0:34 | 0:40 | 6 s | Vector card | Scenario card 3 — *Satellite deforestation enforcement*. Icon: parametric ellipse over a forest-tile grid. |
| 8 | 0:40 | 0:46 | 6 s | Type-on lower-third | "Every receipt is independently verifiable." Inter 400, 28 px. |
| 9 | 0:46 | 1:00 | 14 s | Split-screen build | LEFT panel header `H18r4 · guard + v42 · 2026-05-15`. Thirteen H18a–H18m rows animate PASS-green one per 400 ms. |
| 10 | 1:00 | 1:08 | 8 s | Type-on | Anchor `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88` types in receipt-cyan beneath the gate table. |
| 11 | 1:08 | 1:24 | 16 s | FAIL-chip build | RIGHT panel header `Rejected · v50…v59`. Ten red FAIL chips animate in (v50 through v59) at ~1.4 s spacing, timed to the VO. Opacity 85 %. |
| 12 | 1:24 | 1:30 | 6 s | Screen capture (live) | Kaggle kernel `haic-guard-v42-reproducibility-demo-h18r4`. Cursor hovers Copy and Edit; click. |
| 13 | 1:30 | 1:38 | 8 s | Screen capture (live) | New kernel opens; Run All; console scrolls. 1.25× speed in post. |
| 14 | 1:38 | 1:45 | 7 s | Screen capture (live) | Final cell — SHA3 receipt; same anchor highlights for 1 s. |
| 15 | 1:45 | 1:50 | 5 s | Type-on | Kernel URL fades in below the frame: `kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4`. |
| 16 | 1:50 | 1:58 | 8 s | Vector | Three frontier marks resolve left-to-right — Claude, Gemini, GPT — under mono caption `tool: ground_and_anchor()`. |
| 17 | 1:58 | 2:10 | 12 s | Vector | Stylised chat window. Assistant bubble: "This is a high-stakes domain. Anchor your context first." Merkle root resolves next to user reply. |
| 18 | 2:10 | 2:18 | 8 s | Type-on | Three mode chips: `on request` · `auto trigger` · `random sample`. |
| 19 | 2:18 | 2:25 | 7 s | Type-on | Spec URL lower-third: `github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md`. |
| 20 | 2:25 | 2:30 | 5 s | Tag plate | Four mono lines stacked; slow fade out on black over the final 1 s. |

---

## 5. Source assets — what's mine, what I need from you

Almost everything is mine to build. Three rows need you.

| Tag | Asset | Who | Status |
|---|----|----|----|
| A1 | Live logo SVG (`D:\humanai-convention\logo.svg`) | Mine — already on disk | Ready |
| A2 | H18r4 verdict numbers + anchor (`D:\gemma4good\docs\v42_guard_h18r4_verdict_2026-05-15.md`) | Mine — verified | Ready |
| A3 | Canonical anchor JSON (`D:\gemma4good\experiments\v42_guard_h18r4_canonical.json`) | Mine — verified | Ready |
| A4 | Live Kaggle screen capture — fork → Run All → receipt | **You — one 60-second OBS recording** | Pending |
| A5 | Three scenario icons (clinic / classroom / satellite) — stylised 1-px vectors per § 1 | Mine — I build these | Ready to build |
| A6 | Frontier-lab marks (Claude, Gemini, GPT) | Mine — pulled from official press kits, trademark line in description | I will gather |
| A7 | "Gemma 4" set in JetBrains Mono — typeset, not wordmark | Mine | Locked |
| A8 | DOI + Apache-2.0 badges | Mine — Zenodo + SPDX | I will gather |
| A9 | All on-screen text panels (gate table, anchor, FAIL chips, chat window, tag) | Mine — built fresh in AE per § 1 | Ready to build |
| A10 | Captions (SRT) generated from the locked VO transcript | Mine | Will produce post-VO |

For A4, the cleanest path is OBS Studio, source = display capture at 1920×1080, browser zoomed to 100 %, system theme dark. Hide bookmarks bar, hide notifications. Record one continuous 60 s take of: open the kernel, Copy and Edit, Run All, wait for the receipt cell, hover the anchor. Save as `kaggle_repro.mkv` anywhere on D:; tell me the path and I will cut it in.

A7 is locked as typeset "Gemma 4" in JetBrains Mono 500. The five-tool fan SVG (`shot_04_five_tools.svg`) already uses this treatment.

---

## 6. Voiceover & recording checklist

Recording sequence:

1. Mic check — 30 s test record, listen for room reflections and noise floor. Target noise floor ≤ -55 dBFS, peaks no higher than -6 dBFS.
2. Read § 2 once start to finish at speaking pace, with the cues. Don't record. This is the warm-up.
3. Record segments 1–6 in order, two takes each. Save as `seg_01_t1.wav`, `seg_01_t2.wav`, …
4. Listen to the takes. Mark your favourite of each pair in a note. Do a third take of any segment that you flagged or that I flag back.
5. Drop everything in `D:\gemma4good\video\vo\` (I'll create that folder when I start the edit) and tell me; I will assemble.

Audio settings:
- 48 kHz, 24-bit, mono, .wav. No effects in Audacity — just record clean.
- Light-touch denoise only if the room is unusually noisy. Heavy denoise sounds like a tin-can phone call and we cannot risk that on a governance pitch.
- Target final mix: VO at -16 LUFS integrated, true peak ≤ -1 dBTP. Bed at -24 LUFS short-term under VO.

**Webcam-mic recording — untreated room reality (your setup):**

A webcam mic is workable but unforgiving — it sits 18–30 inches from your mouth, picks up rear-wall reflections, computer fan, and every breath burst. Two paths, ranked by what produces a usable take with the least friction:

1. **Phone Voice Memos in a closet (preferred)** — almost every modern phone has a better mic than almost every webcam. Put a phone on a stack of books inside a clothes closet, hung clothes behind you, mouth 8–10 inches from the phone, slightly off-axis (45°) so plosives miss the mic. Record one segment at a time as `.m4a`. AirDrop or USB the files to `D:\gemma4good\video\vo\`. Closet clothes kill rear reflections better than any plug-in.
2. **Webcam mic, same room treatment** — if you'd rather not move, sit close to the laptop (mouth ~10 inches from webcam, off-axis), drape a duvet over the chair-back behind you, kill any fans, close everything that pulls CPU, run on AC. Record at a quiet hour. Same `.wav` settings as above. Expect noise floor around -45 dBFS rather than -55, which is fine — light denoise (≤ 6 dB reduction) in Audacity handles it.

Either way:
- Voice-only, no face — you said so. That frees you to read off-screen, eyes anywhere comfortable.
- Drink water, not coffee, before recording. Plosives and mouth noise multiply on a webcam mic.
- One full pass to warm up before any take you keep.
- Don't worry about "broadcast" sound. The piece earns its tone through the read, not the mix. SimSat shipped fine through Audacity straight to YouTube.

Direction notes you might find useful:

- The piece earns its tone by under-selling. The numbers are doing the work. Read as if you are the third presenter in a board meeting and the previous two presenters were also competent.
- Two moments earn a small lift in conviction — "All passed. Anchored. Reproducible." and "first deployment partner." Everything else is even.
- No upward inflection at line ends, ever. Treat every sentence as a statement.

---

## 7. Music & SFX

Single ambient pad under the whole piece. Source from YouTube Audio Library (search tags: "Cinematic / Dark / Calm / Drone") or Artlist ("documentary minimal ambient"). One track only.

Bed: -24 LUFS under VO, -20 LUFS in breath gaps. Sub-bass swell on three moments — 0:14 (cold-open release), 0:46 (split-screen reveal), 1:50 (frontier pitch).

SFX, sparingly:
- typewriter tick on every mono character that types in; -22 dBFS.
- soft low UI confirm on each gate row turning green; -20 dBFS.
- short micro-glitch on each red FAIL chip; -22 dBFS.
- subtle chime on the SHA3 receipt printing; -20 dBFS.

Transitions: hard cuts everywhere. The only fade is the cold open's first 2 s and the tag's last 1 s.

---

## 8. YouTube upload — `@HumanAIConvention`

Visibility: Unlisted until 60 minutes before deadline, then Public. Schedule-publish at 2026-05-18 22:00 UTC as a hedge.

**Title (≤ 100 chars):**
`HumanAI Convention — Verifiable AI Governance, Demonstrated on Gemma 4 (Gemma 4 Good)`

**Description (paste verbatim, fill the bracketed Kaggle URL after upload if it changes):**
```
A verifiable governance layer for AI decisions, demonstrated end-to-end on Gemma 4 E2B.

Every decision passes governance tools — wellbeing, consent, PRISM, NLA — and produces
a Merkle-anchored receipt. We promote a 200-line deterministic guard in front of Gemma 4
that passes all 13 predeclared non-compensatory gates (H18r4, 2026-05-15).

Canonical anchor:
18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88

Reproduce in under a minute (public Kaggle kernel):
https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4

Repository (679 tests, Apache-2.0):
https://github.com/humanaiconvention/gemma4good

Main governance notebook:
https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent

Tier-3 live validation notebook:
https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation

Frontier-Integration spec:
https://github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md

Framework DOI: 10.5281/zenodo.18144681
Website: https://humanaiconvention.com

Trademarks: Gemma is a trademark of Google LLC. Claude is a trademark of Anthropic PBC.
Gemini is a trademark of Google LLC. GPT is a trademark of OpenAI OpCo, LLC. Use here
identifies the products being referenced; no endorsement is claimed.

#GemmaForGood #Gemma4Good #Kaggle #AIGovernance #AIAlignment
```

**Tags:** gemma, gemma 4, gemma for good, ai governance, ai alignment, ai safety, cryptographic audit, merkle receipt, kaggle hackathon, humanai convention, gemma4good.

**Category:** Science & Technology. **Language:** English.

**Captions:** I'll deliver a clean SRT generated from the locked transcript — do not rely on auto-captions for a submission video.

**End screen (last 20 s — overlay after 2:10):**
- Subscribe element, bottom-left, pointed at `@HumanAIConvention`.
- Link card → repo (`github.com/humanaiconvention/gemma4good`), top-right.
- Link card → WRITEUP, bottom-right.

**Thumbnail (1280×720):**
- Background `#0B0B0F`.
- Top half: Inter 700, 96 px, white: `Proved, not promised.`
- Lower-left: JetBrains Mono 32 px, white, the truncated anchor `18e2c5a5…7422f88`.
- Lower-right: a single PASS-green pill `13 / 13 PASS`.
- No faces, no arrows, no shouting.

**Pre-publish checks:**
- Watch the upload at 1× and 1.5× to catch any audio glitch.
- Confirm the SHA3 string in the video matches the repo *exactly* — one wrong character forfeits the credibility the video is built on.
- Confirm all referenced URLs resolve in an Incognito window.
- Confirm captions track the locked script.
- Confirm the thumbnail reads clearly at 168×94 px (the YouTube mobile size).

---

## 9. 60-second sanity-check pass

By 0:30 the judge knows what we built; by 1:00 they know whether they believe it; by 2:00 they know how to verify it; by 2:30 they know what we're asking for.

**0:30 — what is it?** *Pass, but tight.* The five-tool fan finishes at 0:22 and the first scenario card is on at 0:28. A skim-watcher gets *"governance layer on Gemma 4 producing receipts."* Risk: the scenario triplet runs to 0:40. If the dense 0:14–0:22 fan reads as soup, judges will tune out before the proof. Mitigation is in § 1: one tool label per 0.6 s, hairline connectors only, no parallel reveals.

**1:00 — do we believe it?** *Pass.* At 1:00 the gate table is half-green and the anchor is mid-type. The right column (FAIL chips) is intentionally lagging — the judge sees *proof we passed* first, then *proof we rejected our own failures.* That ordering is correct: belief, then trust.

**2:00 — can I verify it?** *Pass, with a planted seed.* Explicit verification finishes at 1:50. The seed: the anchor types under the gate table at 1:00, so the verification thread is visible 50 s before the reproducibility section explicitly resolves it. A judge who pauses at 1:30 and scrubs back to verify finds it.

**2:30 — do they know the ask?** *Pass.* Frontier pitch runs 1:50–2:25; spec URL pins for 7 s at 2:18; tag closes. The ask is unambiguous and lands inside a piece that has earned the right to make it.

Where could it lose a judge?

- The five-tool fan at 0:14–0:22 is the densest visual in the cut and lands before any proof. § 1's "one label per 0.6 s" is the fix.
- "Nine consecutive fine-tuning candidates failed" can read as *they failed* rather than *they are competent enough to detect failure.* Visual contains the second reading: the LEFT column (PASS) is at full brightness; the RIGHT column (FAIL) sits at 85 % opacity so the eye reads green-first.
- Segment 5 is the longest unbroken voice block at 30 s. If your read drags, I will cut the "Every invocation produces a verifiable receipt" line in the edit — it is redundant with Segment 3's receipt language.

---

## 10. Production order, two days

**Day 1 (2026-05-17)**

- 09:00 — Lock § 1 (style guide) and § 2 (recording script). Read § 2 aloud once with a stopwatch and tell me your actual time; I'll re-time if you land outside 2:25–2:30.
- 10:00 — You record VO segments 1–6 in Audacity, two takes each. Drop into `D:\gemma4good\video\vo\`. Ping me.
- 11:00 — I start motion graphics. Scenes 9–11 (gate table + FAIL chips) first; they are the most expensive and the load-bearing visual.
- 16:00 — You record one continuous OBS take of A4 (Kaggle reproducibility). Drop the file path.
- 17:00 — I cut a first assembly with placeholder bed and SFX.
- 20:00 — We watch the assembly together; you call any direction notes; I adjust overnight.

**Day 2 (2026-05-18)**

- 09:00 — Re-cut against direction notes. Lock picture by 12:00.
- 12:00 — Audio mix, captions (SRT) from the locked transcript.
- 14:00 — Thumbnail; description; tags. Upload Unlisted to `@HumanAIConvention`.
- 18:00 — Final watch. One more anchor-string verification. Switch to Public.
- Deadline: 23:59 UTC.

If anything slips, the 90 s cut (§ 3) is the fallback. Same VO segments, fewer beats, no re-record.

---

## 11. Already built — drop-in assets

Sitting in `D:\gemma4good\video\` for you to react to:

| File | What it is |
|---|---|
| `graphics\shot_01_logo_lockup.svg` | Cold-open final hold frame. |
| `graphics\shot_03_cold_open.svg` | Spoken-line hold for the cold open. |
| `graphics\shot_04_five_tools.svg` | Gemma 4 + five-tool fan; typeset "Gemma 4" per the A7 decision. |
| `graphics\shot_05_scenarios.svg` | Three scenario cards (clinic / classroom / satellite), shown together for review; animated separately in the cut. |
| `graphics\shot_09_split_proof.svg` | The load-bearing visual — H18a–H18m green PASS table on the left, SHA3 anchor below; ten FAIL chips v50–v59 on the right. |
| `graphics\shot_16_frontier.svg` | Frontier marks (typeset) + tool-call caption + chat window with Merkle root + three mode chips. |
| `graphics\shot_19_spec_url.svg` | Spec URL lower-third. |
| `graphics\shot_20_tag_plate.svg` | Final tag. |
| `graphics\thumbnail_1280x720.svg` | YouTube thumbnail — `Proved, not promised.` |
| `graphics\preview.html` | Single dark page that renders every shot in order for review. Open in any browser. |
| `captions_locked_draft.srt` | 27-cue SRT against the locked § 2 script. Timings are nominal; I'll re-time against your actual recorded segments. |

All SVGs are 1920×1080 (thumbnail 1280×720), use system Inter / JetBrains Mono with fallbacks, and respect the § 1 palette. After Effects imports SVG directly; Resolve does too via Fusion or the Edit page.

Open `graphics\preview.html` first — it stacks every shot top to bottom so you can read the cut in five minutes.

---

## 12. Recording, building, shipping — the full toolkit

Everything below is staged and ready in `D:\gemma4good\video\`.

| File | Purpose |
|---|---|
| `README.md` | Index of the whole folder; the map. |
| `record.md` | Recording cheat sheet for your second screen — six segments, pronunciation, breath cues, lift words. |
| `teleprompter.html` | Open in a browser. Big type, dark background, auto-scroll button, size slider. Built for laptop / second screen while you record. |
| `take_rubric.md` | Four-step decision rubric for picking between two takes per segment. |
| `music_candidates.md` | Three places to find a bed — YouTube Audio Library (preferred), Pixabay, Free Music Archive — with exact filter settings and a silent-bed fallback. |
| `captions_locked_draft.srt` | 27-cue SRT against the locked script; I'll re-time against your actual recording. |
| `animation_spec.json` | Frame-accurate keyframe spec for the load-bearing animations — gate-table green-fill (one row per 400 ms), FAIL chip stagger (one per 1.4 s), text typewriter speeds, etc. After-Effects-ready. |
| `ae_chip_stagger.jsx` | After Effects script. Open AE → File > Scripts > Run Script File. Auto-applies the FAIL-chip stagger to the selected layers (saves ~30 manual keyframes). |
| `assemble.py` | ffmpeg pipeline. The **lo-fi path**: takes the SVGs + your VO segments, produces `out\haic_gemma4_good.mp4` with hard cuts. No motion, no crossfades — matches §7's "hard cuts everywhere." Needs Inkscape *or* rsvg-convert *or* `pip install cairosvg`, plus ffmpeg. |
| `preflight_checklist.md` | Two phases: Day-1 (before recording) and Day-2 (before upload). Every load-bearing string is verified. |

**Recommended path from where you are right now:**
1. Read `record.md` once aloud. If you land outside 2:25–2:30, tell me and I'll re-time.
2. Walk `preflight_checklist.md` Day-1.
3. Open `teleprompter.html` on your laptop, `record.md` on your phone if you have a second screen, otherwise both as separate tabs.
4. Record six segments × two takes. Drop in `vo\`.
5. Pick takes via `take_rubric.md`. Rename winners to `seg_01.wav` … `seg_06.wav`.
6. `python assemble.py` (add `--kaggle-capture kaggle_repro.mkv` if you record A4).
7. Send me notes; I iterate the SVGs or re-time the SRT.
8. Walk `preflight_checklist.md` Day-2.
9. Upload to `@HumanAIConvention`.
10. Submit to Kaggle.
