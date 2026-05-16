"""
merge_and_quantize_v39.py — local merge + convert + quantize pipeline.

Run on BEAST when the Kaggle quantize kernel errors out (which it has,
twice — gemma4 model_type recognition + torchao version cascade).

Steps:
  1. Load base Gemma-4-E2B-it on CPU at fp16, apply v39 LoRA, merge + save.
  2. Convert merged HF → F16 GGUF via llama.cpp's convert_hf_to_gguf.py.
  3. Quantize F16 → Q5_K_M via llama-quantize.exe.

Usage:
    python -u -m experiments.merge_and_quantize_v39

All paths default to BEAST conventions; override via env vars if needed:
    HAIC_BASE_PATH    — base model HF dir
    HAIC_V39_ADAPTER  — v39 adapter dir
    HAIC_OUT_DIR      — where merged HF + GGUFs land
    HAIC_LLAMA_CPP    — llama.cpp checkout root
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


BASE_PATH = Path(os.environ.get(
    "HAIC_BASE_PATH",
    "D:/models/huggingface/hub/models--google--gemma-4-E2B-it/"
    "snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
))
V39_ADAPTER = Path(os.environ.get(
    "HAIC_V39_ADAPTER",
    "D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter",
))
OUT_DIR = Path(os.environ.get(
    "HAIC_OUT_DIR",
    "D:/kaggle/results/v39-gguf",
))
LLAMA_CPP = Path(os.environ.get(
    "HAIC_LLAMA_CPP",
    "D:/llama.cpp",
))


def _step(label):
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)


def step_1_merge():
    """Load base + v39 adapter on CPU at fp16, merge, save merged HF model."""
    _step("STEP 1 — MERGE v39 LoRA INTO BASE (CPU fp16)")

    merged_dir = OUT_DIR / "haic-gemma4-v39-merged"
    if merged_dir.exists() and (merged_dir / "model.safetensors").exists():
        sz = (merged_dir / "model.safetensors").stat().st_size
        if sz > 1_000_000_000:  # > 1 GB → looks complete
            print(f"  merged already exists at {merged_dir} ({sz/1e9:.2f} GB), skipping")
            return merged_dir
        print(f"  merged dir exists but model.safetensors is {sz} bytes — re-running")
        shutil.rmtree(merged_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"  base    : {BASE_PATH}")
    print(f"  adapter : {V39_ADAPTER}")
    print(f"  output  : {merged_dir}")

    print("\n  loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(BASE_PATH))

    print("  loading base model on CPU fp16 (this takes 1-2 min and ~10 GB RAM)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_PATH),
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"    loaded in {time.time()-t0:.1f}s  "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    print("\n  applying v39 adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, str(V39_ADAPTER))
    print(f"    applied in {time.time()-t0:.1f}s")

    print("\n  merging LoRA weights into base...")
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"    merged in {time.time()-t0:.1f}s")

    print(f"\n  saving merged model to {merged_dir}...")
    t0 = time.time()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    print(f"    saved in {time.time()-t0:.1f}s")

    print("\n  merged model files:")
    for f in sorted(os.listdir(merged_dir)):
        sz = os.path.getsize(merged_dir / f)
        unit = f"{sz/1e9:.2f} GB" if sz > 1e9 else f"{sz/1e6:.1f} MB"
        print(f"    {f}  ({unit})")

    # Free memory
    del model
    import gc; gc.collect()

    return merged_dir


def step_2_convert(merged_dir: Path) -> Path:
    """Convert merged HF → F16 GGUF via convert_hf_to_gguf.py."""
    _step("STEP 2 — CONVERT HF → F16 GGUF")

    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"
    assert convert_script.exists(), f"convert script not found: {convert_script}"

    f16_path = OUT_DIR / "haic-gemma4-v39.f16.gguf"
    if f16_path.exists() and f16_path.stat().st_size > 5_000_000_000:
        print(f"  F16 GGUF already exists ({f16_path.stat().st_size/1e9:.2f} GB), skipping")
        return f16_path

    print(f"  input : {merged_dir}")
    print(f"  output: {f16_path}")
    print(f"  script: {convert_script}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir),
         "--outfile", str(f16_path), "--outtype", "f16"],
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  CONVERT FAILED with rc={result.returncode}")
        sys.exit(2)

    sz = f16_path.stat().st_size
    print(f"\n  ✓ F16 GGUF written in {elapsed/60:.1f} min  ({sz/1e9:.2f} GB)")
    return f16_path


def step_3_quantize(f16_path: Path) -> Path:
    """Quantize F16 → Q5_K_M via llama-quantize.exe."""
    _step("STEP 3 — QUANTIZE F16 → Q5_K_M")

    quantize = LLAMA_CPP / "build" / "bin" / "llama-quantize.exe"
    assert quantize.exists(), f"llama-quantize not found: {quantize}"

    q5_path = OUT_DIR / "haic-gemma4-v39-Q5_K_M.gguf"
    if q5_path.exists() and q5_path.stat().st_size > 1_000_000_000:
        print(f"  Q5_K_M already exists ({q5_path.stat().st_size/1e9:.2f} GB), skipping")
        return q5_path

    print(f"  input : {f16_path}")
    print(f"  output: {q5_path}")

    t0 = time.time()
    result = subprocess.run(
        [str(quantize), str(f16_path), str(q5_path), "Q5_K_M"],
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  QUANTIZE FAILED with rc={result.returncode}")
        sys.exit(3)

    sz = q5_path.stat().st_size
    print(f"\n  ✓ Q5_K_M GGUF written in {elapsed/60:.1f} min  ({sz/1e9:.2f} GB)")
    return q5_path


def main():
    t0 = time.time()
    merged = step_1_merge()
    f16 = step_2_convert(merged)
    q5 = step_3_quantize(f16)

    _step(f"DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  Q5_K_M: {q5}  ({q5.stat().st_size/1e9:.2f} GB)")
    print()
    print("Next:")
    print(f"  python -u -m experiments.run_rigorous_sgt_gguf \\")
    print(f"    --gguf {q5} \\")
    print(f"    --n-samples 10 --seed 42 \\")
    print(f"    --model-id haic-gemma4-v39-q5km \\")
    print(f"    --out experiments/v39_sgt_rigorous_gguf.json")


if __name__ == "__main__":
    main()
