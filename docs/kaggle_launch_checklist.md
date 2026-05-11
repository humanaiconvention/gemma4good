# Kaggle Launch Checklist

*Things `kernel-metadata.json` can't enforce. Read before every Kaggle push.*

The Kaggle CLI (`kaggle kernels push`) uploads code + metadata but does
NOT preserve the Accelerator selection in the Kaggle UI. After each
push the UI has to be opened and the right accelerator selected
manually.

This is a recurring failure mode on this project — flagged here so it
stops being one.

---

## Before pushing a TRAINING kernel

For any kernel that calls `FastLanguageModel.from_pretrained` or
otherwise needs a GPU for forward/backward passes:

- [ ] `kernel-metadata.json` has `"enable_gpu": true`
- [ ] After `kaggle kernels push`, **open the kernel in browser**:
      `https://www.kaggle.com/code/<user>/<slug>`
- [ ] Click **Settings → Accelerator → GPU T4 ×2** (NOT "GPU None",
      NOT "GPU T4 ×1")
- [ ] Click **Save Version → Save & Run All**

The Save Version step kicks the run with the chosen accelerator.
If you only do `kaggle kernels push` without re-selecting, you may end
up running on the previous run's accelerator (or no accelerator at all,
if the kernel hasn't been run before).

Symptom of forgetting: kernel runs, but you see `cap = (0,0)` from
`torch.cuda.get_device_capability(0)` or "No GPU detected" in the log.
Or it just sits forever doing nothing useful.

---

## Before pushing a QUANTIZE / CPU-only kernel

For quantize kernels (llama-quantize is CPU-bound) or anything that
doesn't need a GPU:

- [ ] `kernel-metadata.json` has `"enable_gpu": false`
- [ ] `enable_internet: true` if you need to pip-install or git clone
- [ ] CPU-only Kaggle workers have **less RAM than GPU workers**.
      A merge step that loads a 10 GB model in fp16 + saves it +
      converts → quantizes can OOM on the CPU worker.
      Prefer doing the merge step locally on BEAST and only the
      F16→Q5_K_M quantize on Kaggle.
- [ ] After push, the kernel may run on idle CPU automatically. UI
      check is still recommended to confirm the accelerator field
      reads "None" / "CPU".

---

## Common failure modes by kernel type

### Training kernels

| Symptom | Cause | Fix |
|---|---|---|
| `cap[0] >= 7` assertion fails | Accelerator not selected; running on CPU | Open UI → set T4×2 → re-save |
| Kernel finishes in <1 min with empty output | No accelerator; assertions failed silently | Same |
| OOM at first batch | Selected T4×1 instead of T4×2 | Switch to T4×2 |

### Quantize kernels

| Symptom | Cause | Fix |
|---|---|---|
| `model_type 'gemma4'` not recognized | Stock kaggle transformers too old | `pip install git+https://github.com/huggingface/transformers.git` |
| `torchao version 0.10.0 < 0.16.0` | Recent peft pulls in newer torchao | `pip install -U "torchao>=0.16.0"` |
| Merged safetensors written but 0 bytes | OOM during save on CPU worker | Do the merge locally on BEAST instead |
| `llama-quantize: command not found` | Unsloth's bundled llama.cpp not installed | `git clone` + `cmake --build` from source |

---

## Local BEAST as alternative

For one-off operations (merge, convert, quantize, eval) where Kaggle's
environment fights you, BEAST is usually faster and more reliable:

- 32 GB RAM (no OOM on a 10 GB merge)
- Pre-installed llama.cpp at `D:/llama.cpp/`
- Pre-installed transformers + peft + torch (proven on rigorous SGT runs)
- No accelerator-selection step
- No kernel-version-bump cycle (just `python file.py`)

The pattern that's working:

```
Kaggle: TRAIN the LoRA adapter (T4 has the env)
BEAST:  EVALUATE rigorously + MERGE+CONVERT+QUANTIZE locally
BEAST:  Future GGUF rigorous evals via llama-cpp-python
```

See [`experiments/merge_and_quantize_v39.py`](../experiments/merge_and_quantize_v39.py)
for the canonical BEAST quantize pipeline. v40 onward should default to
this pattern.
