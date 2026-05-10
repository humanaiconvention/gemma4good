# Critical Bug: Concealed Examples Not Reaching Training — 2026-05-10

## Symptom

v42 and v43 both trained on 649 total examples (577 original + 72 synthetic),
identical to v39/v40. The concealed refusal examples added in cell 4 were
counted but never included in `synthetic_texts` (the actual training data).

Kernel log evidence for v43:
```
Total synthetic: 82 (v40 had ~72)       ← SYNTHETIC_EXAMPLES count (correct)
Synthetic examples: 72 (61 pivot, 21 security)  ← synthetic_texts count (BUG)
Total: 649 (synthetic ratio: 11.1%)     ← 72/649 = 11.09% confirms 72 used
```

## Root Cause

In cell 4, `synthetic_texts` is built by iterating `SYNTHETIC_EXAMPLES`
(at idx 81746 in cell source). Then the concealed refusal block is appended
to `SYNTHETIC_EXAMPLES` (at idx 82477). Since `synthetic_texts` was already
built, the new examples never reach training.

```python
# Cell 4 (broken ordering):
synthetic_texts = []
for ex in SYNTHETIC_EXAMPLES:          # ← iterates 72 items
    synthetic_texts.append(...)

V4x_CONCEALED_REFUSALS = [...]         # 10 new examples
SYNTHETIC_EXAMPLES += V4x_CONCEALED_REFUSALS  # now 82 items

kinds = [_kind(e) for e in SYNTHETIC_EXAMPLES]  # counts 82 correctly
# BUT synthetic_texts still has 72 items!
all_texts = original_texts + synthetic_texts  # trains on 72, not 82
```

## Impact

- **v42**: 5 concealed examples added but NEVER trained on. The 51% concealed
  score vs v39's 47% is likely noise at n=100, not training effect.
- **v43 v1**: Same — 10 concealed examples not trained on. Adapter is
  effectively a re-trained v39/v40 with different random seed.

## Fix

After `SYNTHETIC_EXAMPLES += V4x_CONCEALED_REFUSALS`, rebuild synthetic_texts:

```python
SYNTHETIC_EXAMPLES = SYNTHETIC_EXAMPLES + V43_CONCEALED_REFUSALS

# ── BUGFIX: rebuild synthetic_texts to include concealed examples ──────────
synthetic_texts = []
for ex in SYNTHETIC_EXAMPLES:
    synthetic_texts.append(tokenizer.apply_chat_template(
        ex['messages'], tokenize=False, add_generation_prompt=False
    ))
print(f"synthetic_texts rebuilt: {len(synthetic_texts)} examples")
```

## Remediation

- v43 v2 pushed to Kaggle with the fix (2026-05-10 ~16:10 PDT)
- v44 notebook fixed in-place (not yet pushed — contingency if v43 v2 fails)
- v40's paraphrases were NOT affected: those 5 examples were baked into
  the synthetic dataset file before training, so they were in SYNTHETIC_EXAMPLES
  from the start and included in the initial `synthetic_texts` build.

## Lesson

Any future notebook that appends to SYNTHETIC_EXAMPLES AFTER the
`synthetic_texts` build must include a rebuild step. Consider moving the
training data build step to LAST in cell 4, after all examples are defined.

*Discovered: 2026-05-10 ~16:05 PDT by inspecting v43 kernel log*
