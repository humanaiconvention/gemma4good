# BEAST Gemma 4 loading limitations

**Date:** 2026-04-07
**Host:** BEAST (Windows 11, RTX 2080 8GB, 32GB RAM)
**Software:** Python 3.13, torch 2.7.1+cu118, transformers 5.5.0, bitsandbytes 0.49.1

## TL;DR

We cannot load any **multimodal** Gemma 4 variant from HuggingFace on this Windows host. The text-only fine-tunes under `experiments/gemma4-conditioned*` load cleanly. This affects which Prism analyses we can produce locally vs. which we need to defer to the Kaggle GPU runtime or an A100 burst.

## What works on BEAST

- `experiments/gemma4-conditioned/` — text-only, ~3GB safetensors, loads via `AutoModelForCausalLM` + bf16 + `device_map="cpu"`. Verified Prism run: `qh=0.9145, outlier=83.17x`.
- `experiments/gemma4-conditioned-aggressive/` — same loader, same path. Verified Prism run: `qh=0.9062, outlier=74.50x`.

## What does NOT work on BEAST

- `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` — these are full multimodal `Gemma4ForConditionalGeneration` checkpoints with text + vision + audio encoders. **Three independent failure modes blocked them**:

  1. **`Gemma4ForConditionalGeneration` segfaults during model construction** (before any weights are read) on this Windows + transformers 5.5.0 stack. Crashes are silent — Python process exit 139 with no Python traceback. The crash is reproducible across `device_map="cpu"`, `device_map="auto"`, and `device_map="cuda:0"`. The audio encoder init (`gemma4_audio` model_type) is the most likely culprit but cannot be confirmed without a debugger build of torch.

  2. **`Gemma4ForCausalLM` (text-only class) loads structurally**, but the public checkpoint stores text-decoder weights under `model.language_model.layers.*` (the multimodal nesting), while `Gemma4ForCausalLM` expects `model.layers.*`. **Every checkpoint key is UNEXPECTED and every model key is MISSING** — the model loads with **randomly initialized weights**. A Prism run against this state produces **plausible-looking but meaningless numbers** (`qh≈0.09`, `outlier≈2x` — the geometry of a random Gaussian init). One such run was generated and immediately purged from the ledger to prevent contamination.

  3. **bitsandbytes 4-bit NF4 path is broken** for Gemma 4 on this BNB version. `fix_4bit_weight_quant_state_from_module` asserts `module.weight.shape[1] == 1` on every linear's first forward pass (`q_proj`, `per_layer_model_projection`, etc). The assertion is BNB's internal sanity check for quantized state and fails because the gemma4 modeling code doesn't pass weights through BNB's expected shape transformation. `llm_int8_skip_modules` does not help — the assertion fires on every linear, not just one.

## Workarounds tried (did not help)

- `__main__` guard on `experiments/gemma4/run_prism_analysis.py` to prevent unwanted auto-run of `LOCAL_MODEL_PATH=gemma4-conditioned` consuming RAM before the requested model loads. **Implemented and committed**, but did not resolve the root segfault.
- `device_map="auto"` to split layers across GPU+CPU. Same segfault.
- bf16 with `Gemma4ForCausalLM`. Loads but with random weights (see failure mode 2).
- 4-bit NF4 with `Gemma4ForCausalLM`. Loads structurally but asserts on first forward (failure mode 3).
- Skipping specific modules from BNB quantization via `llm_int8_skip_modules=['per_layer_model_projection']`. Assertion fires on the next linear (`q_proj`) instead.

## What this means for the gemma4good entry

The gemma4good Kaggle notebook runs on **Kaggle's 2xT4 environment (Linux)**, not BEAST. None of the Windows-specific failure modes above are guaranteed to reproduce on Linux. The notebook's model-load cell has been patched to use `google/gemma-4-26B-A4B-it` (with `google/gemma-4-E4B-it` fallback) instead of the non-existent `gemma-4-12b-it`, and Kaggle judges should be able to load it via the standard `BitsAndBytesConfig` + `device_map="auto"` path. **The entry is not blocked by the BEAST loading issue.**

What is affected: we cannot pre-generate Prism numbers for `gemma-4-26B-A4B-it` or `gemma-4-31B-it` on local hardware to include in the entry's narrative as "here are the actual Kaggle-target numbers." The supporting Prism story uses `gemma4-conditioned` and `gemma4-conditioned-aggressive` instead, which are the same architecture family (text-only Gemma 4) and are documented as proxies in the notebook and the writeup.

## Path to actual Kaggle-target Prism numbers

When we want fresh numbers on the actual `gemma-4-26B-A4B-it` and `gemma-4-31B-it` variants, the options are, in order of cost:

1. **Run on Kaggle itself** — open the gemma4good notebook on Kaggle, drop a Prism cell after the model load, capture the four-metric tuple per layer. This is the cleanest path and reuses the Kaggle GPU runtime that the entry already targets.
2. **A100 burst (Linux)** — Guilherme's A100 box runs Linux and almost certainly does not have the Windows multimodal segfault. Install the same harness (`experiments/gemma4/run_prism_analysis.py`), point at the model id, run.
3. **Windows debugging** — try a newer transformers version (5.6+ may fix the multimodal init), or build torch with debug symbols and trace the audio_tower init. Time-expensive, low expected value.
4. **Pre-process the safetensors** — manually strip the `model.language_model.` prefix from keys, drop audio/vision keys, save as a Gemma4ForCausalLM-compatible checkpoint, then load it. Fragile, doesn't match the upstream checkpoint, and would silently change with each model release.

Recommendation: option (1) or (2), whichever is faster.
