# Gemma 4 v1 Ready — for Gemma4Good / Antigravity

Generated: 2026-04-11 07:07 UTC

A valid Gemma 4 E2B QLoRA fine-tune is now available on BEAST.

## Model

Path: `D:\humanai-convention\experiments\gemma4-v1\gemma4-v1-Q5_K_M.gguf`
Format: Q5_K_M GGUF (llama.cpp compatible)
PRISM hostility: 0.7695 (baseline 0.9146, delta negligible — geometry-stable)

## What it was trained on

SGT grounding sessions (500 × 7-turn), all PIVOT-tagged. Not fine-tuned on Gemma4Good tasks —
this is a grounding/interviewer model, NOT a function-calling or tool-use model.

## How to use with llama.cpp

```bash
D:/llama.cpp/build/bin/llama-server.exe \
  -m D:\humanai-convention\experiments\gemma4-v1\gemma4-v1-Q5_K_M.gguf \
  --port 8080 \
  -c 2048
```

## Relevance to Gemma4Good

- Can serve as a reference for Gemma 4 GGUF inference pipeline setup on BEAST
- PRISM geometry: geometry-stable (LoRA fine-tune does not worsen quantization hostility)
- The base Gemma 4 E2B-it is the same architecture used in Gemma4Good tools
- SGT eval results pending — check D:/humanai-convention/experiments/gemma4-v1/HANDOFF.md

## Note

The Prism arena in D:/gemma4good/tools/haic_tools.py has cached results for other models.
Gemma4-v1 PRISM results are at:
  D:/humanai-convention/experiments/gemma4-v1/prism_gemma4_v1.json
