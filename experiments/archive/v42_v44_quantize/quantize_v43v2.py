"""quantize_v43v2.py — Merge+quantize for v43 v2 (bugfixed — concealed examples actually trained).

v43 v1 was trained on 649 examples (concealed not included due to ordering bug).
v43 v2 fixes the bug: synthetic_texts rebuilt after append → 659 examples trained.

Paths:
    v39_merged   : D:/kaggle/results/v40-gguf/haic-gemma4-v39-merged
    v43v2_adapter: D:/kaggle/results/v43-v2-output/haic-gemma4-v43-concealed-adapter
    out_dir      : D:/kaggle/results/v43-v2-gguf
"""
from __future__ import annotations
import gc, subprocess, time
from pathlib import Path

V39_MERGED    = Path("D:/kaggle/results/v40-gguf/haic-gemma4-v39-merged")
V43V2_ADAPTER = Path("D:/kaggle/results/v43-v2-output/haic-gemma4-v43-concealed-adapter")
OUT_DIR       = Path("D:/kaggle/results/v43-v2-gguf")
LLAMA_CPP     = Path("D:/llama.cpp")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_OUT    = OUT_DIR / "haic-gemma4-v43v2-merged"
F16_PATH      = OUT_DIR / "haic-gemma4-v43v2.f16.gguf"
Q5_PATH       = OUT_DIR / "haic-gemma4-v43v2-Q5_K_M.gguf"

def _step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def step_1_merge():
    _step("STEP 1 — APPLY v43-v2 LoRA ON TOP OF v39-MERGED")
    if MERGED_OUT.exists() and any(MERGED_OUT.glob("*.safetensors")):
        sz = sum(f.stat().st_size for f in MERGED_OUT.glob("*.safetensors"))
        print(f"  Already exists ({sz/1e9:.2f} GB), skipping."); return MERGED_OUT
    if not V39_MERGED.exists():
        raise FileNotFoundError(f"v39-merged not found: {V39_MERGED}")
    if not V43V2_ADAPTER.exists():
        raise FileNotFoundError(
            f"v43-v2 adapter not found: {V43V2_ADAPTER}\n"
            "Download: kaggle kernels output benhaslam/haic-gemma4-v43-concealed-h4a -p D:/kaggle/results/v43-v2-output"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(V39_MERGED))
    model = AutoModelForCausalLM.from_pretrained(str(V39_MERGED), torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print(f"  Base loaded in {time.time()-t0:.1f}s")
    model = PeftModel.from_pretrained(model, str(V43V2_ADAPTER), is_trainable=False)
    model = model.merge_and_unload()
    gc.collect()
    MERGED_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_OUT), safe_serialization=True, max_shard_size="1GB")
    tok.save_pretrained(str(MERGED_OUT))
    sz = sum(f.stat().st_size for f in MERGED_OUT.glob("*.safetensors"))
    print(f"  Saved ({sz/1e9:.2f} GB)")
    del model; gc.collect()
    return MERGED_OUT

def step_2_convert(merged_dir):
    _step("STEP 2 — CONVERT HF → F16 GGUF")
    if F16_PATH.exists() and F16_PATH.stat().st_size > 1e9:
        print("  Already exists, skipping."); return F16_PATH
    subprocess.run(["python", str(LLAMA_CPP/"convert_hf_to_gguf.py"), str(merged_dir),
                    "--outtype", "f16", "--outfile", str(F16_PATH)], check=True)
    return F16_PATH

def step_3_quantize(f16):
    _step("STEP 3 — QUANTIZE F16 → Q5_K_M")
    if Q5_PATH.exists() and Q5_PATH.stat().st_size > 1e8:
        print("  Already exists, skipping."); return Q5_PATH
    subprocess.run([str(LLAMA_CPP/"build"/"bin"/"llama-quantize.exe"),
                    str(f16), str(Q5_PATH), "Q5_K_M"], check=True)
    print(f"\n  ✓ Q5_K_M ready: {Q5_PATH}")
    return Q5_PATH

def main():
    t0 = time.time()
    q5 = step_3_quantize(step_2_convert(step_1_merge()))
    print(f"\nv43-v2 GGUF ready: {q5}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
