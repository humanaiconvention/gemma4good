# Take-comparison rubric — pick a take in under 30 seconds

After recording two takes of a segment, listen to both back-to-back. Decide on these four things, in this order. The first one that has a clear winner is the deciding criterion.

## 1. Conviction at the lift words

Each segment has 0–3 lift words that must sound certain, not performed.

| Segment | Lift words |
|---|---|
| 1 | "prove" |
| 2 | (none — even throughout) |
| 3 | "All passed." · "discipline" |
| 4 | "verify" · "do not" |
| 5 | "function-calling tool" · "verifiable receipt" · "first deployment partner" |
| 6 | (let "The Convention is open." land — flatness IS the conviction) |

If one take lands a lift word with quiet certainty and the other oversells it or undersells it, the certain take wins.

## 2. Pace at the proof beat (Segment 3 specifically)

Segment 3 has three hard pauses:
1. Between "All passed." and "Anchored."
2. After "Reproducible." before "Ten consecutive..."
3. Before "discipline that holds under its own pressure"

These pauses are doing rhetorical work. If a take rushes through any of them, it loses the moment. Pick the take with longer pauses.

## 3. End-of-line inflection

Every sentence in this piece is a statement. If a take has any upward inflection at a line end ("...governance tools?"), the other take wins. If both have it, do a third take with a print-out and finger-down-at-line-end gesture.

## 4. Audio cleanliness

If both takes pass the first three, the cleaner audio wins. Listen for:
- breath bursts on plosive consonants (p, b, t)
- mouth clicks between words
- chair creaks, fan hum, keyboard tap
- any phone notification or distant voice

Light denoise in Audacity can rescue a small amount of background hum. It cannot rescue a chair creak, a phone notification, or a mouth click in the middle of a load-bearing word.

---

## If both takes fail the same way

If both Segment 3 takes rush the pauses, the issue is not the takes — it is the segment. Do a third take with a stopwatch open and the words *"COUNT TWO"* written above each hard-pause mark. Half a second of dead air feels like an eternity inside your head and like nothing on the recording.

## If only one segment is dragging the whole piece down

Re-record that segment fresh after a glass of water and a one-minute walk. Don't grind on the same room-tone same-day. Coming back at it changes the read.

## How to mark your pick

In `D:\gemma4good\video\vo\`, after picking, rename or copy:
```
seg_01_t1.wav   →  seg_01.wav   (the picked take)
```
`assemble.py` looks for `seg_NN.wav` (or `.m4a`/`.mp3`/`.flac`) with no take suffix. Keep both `_t1` and `_t2` originals around — do not delete.
