# Kaggle Integration Gotchas (Learned the Hard Way)

**Date:** 2026-05-11
**Status:** Operational notes from today's v46 DPO + canonical eval pushes

When pushing notebooks to Kaggle that reference adapters / datasets /
kernel-source outputs, three things tripped us up today. All three are
fixed in the active notebooks (`haic-gemma4-v46-dpo`, `haic-canonical-
eval-gemma4good`) — but worth recording for future pushes.

---

## 1. Kaggle's actual mount structure has username prefixes

Documentation suggests:
  - Dataset at slug `user/foo` → `/kaggle/input/foo/`
  - Kernel output at slug `user/bar` → `/kaggle/input/bar/`

**Reality (verified by walking /kaggle/input at runtime):**
  - Dataset at `benhaslam/foo` → `/kaggle/input/datasets/benhaslam/foo/`
  - Kernel output at `benhaslam/bar` → `/kaggle/input/notebooks/benhaslam/bar/`

The username AND the source-type prefix (`datasets/` or `notebooks/`)
are in the path. Hard-coded paths break.

**Fix that works** (used in both v46-dpo and canonical-eval):

```python
import glob
candidates = sorted(glob.glob('/kaggle/input/**/adapter_config.json', recursive=True))
for p in candidates:
    if 'v42' in p.lower():
        ADAPTER_PATH = str(Path(p).parent)
        break
```

Use `glob.glob('/kaggle/input/**/<unique_filename>', recursive=True)` for
every input file. Don't hard-code the path.

## 2. `prepare_model_for_kbit_training` OOMs on Gemma-4-E2B (T4)

`peft.prepare_model_for_kbit_training()` casts non-4bit params to float32.
Gemma-4-E2B has a large per-layer token embedding table — when upcast to
fp32 it tries to allocate 8.75 GB on top of the 4-bit weights already
loaded. T4 only has 14.56 GB total; with the model + KV cache also
resident, the upcast OOMs.

**Fix that works** (used in v46-dpo):

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, quantization_config=bnb_config, device_map='auto',
    torch_dtype=torch.float16,
)
# Skip prepare_model_for_kbit_training (Gemma-4-E2B OOM on T4).
# DPOTrainer + PEFT handles the setup with these two calls instead:
base_model.gradient_checkpointing_enable()
if hasattr(base_model, "enable_input_require_grads"):
    base_model.enable_input_require_grads()
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
```

Plus, in DPOConfig: `gradient_checkpointing=True`,
`per_device_train_batch_size=2`, `gradient_accumulation_steps=4` to keep
peak memory low.

## 3. Kernel slug matches the title-derived slug, not what you set

If your `kernel-metadata.json` has `"id": "benhaslam/foo"` and
`"title": "Foo Project Demo"`, Kaggle generates the slug from the
title (`foo-project-demo`), and the `id` you specified is silently
remapped on push:
  - First push: warning about slug mismatch; kernel is created at
    the title-derived URL.
  - Subsequent pushes with the original `id`: **409 Conflict** because
    that id now points at the title-derived slug instead.

**Fix**: pick an `id` that matches the title-derived slug exactly.
Easiest: name the project `<slug-text>` in the title (e.g. title
"HAIC Canonical Eval Gemma4Good" + id `haic-canonical-eval-gemma4good`).

## 4. Tag rejection is non-blocking but verbose

Kaggle accepts arbitrary string tags in `kernel-metadata.json`'s
`keywords` but silently rejects any that aren't in its allowed list.
This prints a noisy warning on every push but doesn't fail. Stick to
the safe set: `gpu`, `tpu`, and any keyword Kaggle has indexed (no
canonical list published).

---

## How to verify a new notebook before training time

For any notebook that loads inputs, add a Cell 2 that walks `/kaggle/
input` and prints the structure. Run a short version of the kernel
(e.g., n=2 sessions, 1 seed) before committing to a 30-min training
run. The walk gives you the real mount layout so the glob discovery
works:

```python
import os
print("="*60); print("INPUT FILE LISTING"); print("="*60)
for d, _, fs in os.walk('/kaggle/input'):
    for f in fs:
        fp = os.path.join(d, f)
        print(f"  {fp}  ({os.path.getsize(fp):,} bytes)")
```

The v45 training notebook had this; the v46 notebook didn't (until v3).
Adding it should be standard practice for new training kernels.

---

## Today's push log

| Kernel | Attempt | Result | Cause |
|---|---|---|---|
| `haic-canonical-eval-gemma4good` | v1 | slug mismatch warning | id != title slug |
| `haic-canonical-eval-gemma4good` | v2 | accepted | id corrected |
| `haic-canonical-eval-gemma4good` | v3 | accepted | adapter path glob-ed |
| `haic-gemma4-v46-dpo` | v1 | OOM | `prepare_model_for_kbit_training` |
| `haic-gemma4-v46-dpo` | v2 | OOM fixed; new error | hard-coded `V42_ADAPTER` path |
| `haic-gemma4-v46-dpo` | v3 | adapter found; new error | hard-coded `DPO_PAIRS_PATH` |
| `haic-gemma4-v46-dpo` | v4 | RUNNING (current) | all paths glob-ed |

Three errors in 8 minutes = real diagnostic friction. The cost was
modest because each error fired at load time, not 25 minutes into
training. Recommend: glob-discover ALL inputs in every Kaggle notebook
going forward.

---

*"Follow the science." Every one of today's three Kaggle errors had a
sensible specific cause and a one-line fix. The pattern (hard-coded
path → mount layout mismatch → glob fix) is the kind of thing that's
obvious in hindsight but invisible without empirical iteration. A
runtime mount-listing in every notebook makes the invisible visible.*
