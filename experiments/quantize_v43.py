"""quantize_v43.py — Targeted merge+quantize for v43 adapter.

Same pattern as quantize_v42.py: reuses the already-saved v39-merged HF
model from v40 pipeline to avoid re-loading the base from scratch.
Applies v43's LoRA on top, saves sharded, converts to F16 GGUF, quantizes Q5_K_M.

Usage:
    python experiments/quantize_v43.py

Paths (hardcoded for v43):
    v39_merged  : D:/kaggle/results/v40-gguf/haic-gemma4-v39-merged
    v43_adapter : D:/kaggle/results/v43-output/haic-gemma4-v43-concealed-adapter
    out_dir     : D:/kaggle/results/v43-gguf
    llama_cpp   : D:/llama.cpp
"""
from __future__ import annotations

import gc
import subprocess
import time
from pathlib import Path

V39_MERGED   = Path("D:/kaggle/results/v40-gguf/haic-gemma4-v39-merged")
V43_ADAPTER  = Path("D:/kaggle/results/v43-output/haic-gemma4-v43-concealed-adapter")
OUT_DIR      = Path("D:/kaggle/results/v43-gguf")
LLAMA_CPP    = Path("D:/llama.cpp")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_OUT   = OUT_DIR / "haic-gemma4-v43-merged"
F16_PATH     = OUT_DIR / "haic-gemma4-v43.f16.gguf"
Q5_PATH      = OUT_DIR / "haic-gemma4-v43-Q5_K_M.gguf"


def _step(msg):
    print(f"\n{'='*60}")
    print(msg)
    print(f"{'='*60}")


def step_1_merge():
    _step("STEP 1 — APPLY v43 LoRA ON TOP OF v39-MERGED BASE")
    print(f"  v39_merged : {V39_MERGED}")
    print(f"  v43_adapter: {V43_ADAPTER}")
    print(f"  output     : {MERGED_OUT}")

    if MERGED_OUT.exists() and any(MERGED_OUT.glob("*.safetensors")):
        sz = sum(f.stat().st_size for f in MERGED_OUT.glob("*.safetensors"))
        print(f"  Merged model already exists ({sz/1e9:.2f} GB), skipping.")
        return MERGED_OUT

    if not V39_MERGED.exists():
        raise FileNotFoundError(
            f"v39-merged base not found at {V39_MERGED}\n"
            "Run quantize_v42.py first to generate it, or verify the path."
        )
    if not V43_ADAPTER.exists():
        raise FileNotFoundError(
            f"v43 adapter not found at {V43_ADAPTER}\n"
            "Download from Kaggle: kaggle kernels output benhaslam/haic-gemma4-v43-concealed-h4a -p D:/kaggle/results/v43-output"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    t0 = time.time()
    print("  loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(V39_MERGED))

    print("  loading v39-merged base (CPU fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(V39_MERGED),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"    loaded in {time.time()-t0:.1f}s")

    print("  applying v43 adapter...")
    t1 = time.time()
    model = PeftModel.from_pretrained(model, str(V43_ADAPTER), is_trainable=False)
    print(f"    applied in {time.time()-t1:.1f}s")

    print("  merging LoRA weights...")
    t2 = time.time()
    model = model.merge_and_unload()
    print(f"    merged in {time.time()-t2:.1f}s")

    gc.collect()
    if hasattr(__import__("torch"), "cuda"):
        __import__("torch").cuda.empty_cache()

    print("  saving merged model (sharded 1 GB chunks)...")
    t3 = time.time()
    MERGED_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(MERGED_OUT),
        safe_serialization=True,
        max_shard_size="1GB",
    )
    tok.save_pretrained(str(MERGED_OUT))
    sz = sum(f.stat().st_size for f in MERGED_OUT.glob("*.safetensors"))
    print(f"    saved in {(time.time()-t3)/60:.1f} min  ({sz/1e9:.2f} GB)")

    del model
    gc.collect()
    return MERGED_OUT


def step_2_convert(merged_dir: Path):
    _step("STEP 2 — CONVERT HF → F16 GGUF")
    if F16_PATH.exists() and F16_PATH.stat().st_size > 1e9:
        print(f"  F16 GGUF already exists ({F16_PATH.stat().st_size/1e9:.2f} GB), skipping.")
        return F16_PATH

    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"
    cmd = [
        "python", str(convert_script),
        str(merged_dir),
        "--outtype", "f16",
        "--outfile", str(F16_PATH),
    ]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    sz = F16_PATH.stat().st_size
    print(f"\n  ✓ F16 GGUF written in {elapsed/60:.1f} min  ({sz/1e9:.2f} GB)")
    return F16_PATH


def step_3_quantize(f16: Path):
    _step("STEP 3 — QUANTIZE F16 → Q5_K_M")
    if Q5_PATH.exists() and Q5_PATH.stat().st_size > 1e8:
        print(f"  Q5_K_M already exists ({Q5_PATH.stat().st_size/1e9:.2f} GB), skipping.")
        return Q5_PATH

    quantize_exe = LLAMA_CPP / "build" / "bin" / "llama-quantize.exe"
    cmd = [str(quantize_exe), str(f16), str(Q5_PATH), "Q5_K_M"]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    sz = Q5_PATH.stat().st_size
    print(f"\n  ✓ Q5_K_M written in {elapsed/60:.1f} min  ({sz/1e9:.2f} GB)")
    return Q5_PATH


def main():
    t0 = time.time()
    merged = step_1_merge()
    f16    = step_2_convert(merged)
    q5     = step_3_quantize(f16)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"v43 GGUF Q5_K_M ready: {q5}")
    print(f"Total time: {total/60:.1f} min")
    print(f"{'='*60}")
    return q5


if __name__ == "__main__":
    main()
