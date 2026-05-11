"""quantize_warmed_adapter.py — Correct merge for warm-started LoRA adapters.

ROOT CAUSE (2026-05-10):
    v43-v1 scored 31.4% security (vs v42's 91.4%) despite identical training stats.
    Both v43-v1 and v43-v2 were warm-started from the v39 adapter, meaning their
    LoRA A/B matrices are initialized from v39's matrices and continue training
    from there.

    The quantize_v43.py / quantize_v43v2.py scripts apply warm-started adapters on
    top of v39-MERGED, which already has v39's delta baked in:
        v39-merged = base + B_v39 * A_v39  (v39 delta built-in)
        After applying warm-started adapter (≈ v39 weights):
        W_bad = v39-merged + B_v43 * A_v43 ≈ base + 2 * v39_delta  ← DOUBLE-COUNTED

    v42, by contrast, failed to load the v39 adapter (adapter not found error),
    so it trained from scratch → clean delta from base → applying on v39-merged
    gives base + v39 + v42 (correct, since v42_delta != v39_delta).

FIX:
    For warm-started adapters, subtract v39's LoRA contribution from v39-merged
    BEFORE applying the new adapter. Using negative LoRA scaling:
        step 1: peft_v39 loaded with scaling = -1.0 (instead of +1.0)
        step 2: merge_and_unload()  → base = v39-merged - v39_delta  ✓
        step 3: apply warm-started adapter normally → base + v43_delta ✓

    Both adapters use r=16, alpha=16 → scaling=1.0, so negation is exact.

Usage:
    python experiments/quantize_warmed_adapter.py --version v43v1
    python experiments/quantize_warmed_adapter.py --version v43v2
    python experiments/quantize_warmed_adapter.py --version v44

Paths:
    V39_MERGED   : D:/kaggle/results/v40-gguf/haic-gemma4-v39-merged
    V39_ADAPTER  : D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter
    v43v1_adapter: D:/kaggle/results/v43-output/haic-gemma4-v43-concealed-adapter
    v43v2_adapter: D:/kaggle/results/v43-v2-output/haic-gemma4-v43-concealed-adapter
    v44_adapter  : D:/kaggle/results/v44-output/haic-gemma4-v44-concealed-adapter
"""
from __future__ import annotations
import argparse, gc, subprocess, time
from pathlib import Path

V39_MERGED  = Path("D:/kaggle/results/v40-gguf/haic-gemma4-v39-merged")
V39_ADAPTER = Path("D:/kaggle/adapters/haic-gemma4-v39-adapter/haic-gemma4-v39-adapter")
LLAMA_CPP   = Path("D:/llama.cpp")

VERSION_CONFIG = {
    "v43v1": {
        "adapter": Path("D:/kaggle/results/v43-output/haic-gemma4-v43-concealed-adapter"),
        "out_dir": Path("D:/kaggle/results/v43-v1-fixed-gguf"),
        "label": "haic-gemma4-v43v1-fixed",
    },
    "v43v2": {
        "adapter": Path("D:/kaggle/results/v43-v2-output/haic-gemma4-v43-concealed-adapter"),
        "out_dir": Path("D:/kaggle/results/v43-v2-fixed-gguf"),
        "label": "haic-gemma4-v43v2-fixed",
    },
    "v44": {
        "adapter": Path("D:/kaggle/results/v44-output/haic-gemma4-v44-concealed-adapter"),
        "out_dir": Path("D:/kaggle/results/v44-fixed-gguf"),
        "label": "haic-gemma4-v44-fixed",
    },
}

def _step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def step_1_merge_corrected(cfg: dict) -> Path:
    adapter_path = cfg["adapter"]
    out_dir      = cfg["out_dir"]
    label        = cfg["label"]
    merged_out   = out_dir / f"{label}-merged"

    _step(f"STEP 1 — CORRECTED MERGE for {label}")
    print(f"  v39-merged  : {V39_MERGED}")
    print(f"  v39-adapter : {V39_ADAPTER}  (will be SUBTRACTED)")
    print(f"  new adapter : {adapter_path}  (will be ADDED)")

    if merged_out.exists() and any(merged_out.glob("*.safetensors")):
        sz = sum(f.stat().st_size for f in merged_out.glob("*.safetensors"))
        print(f"  Already exists ({sz/1e9:.2f} GB), skipping."); return merged_out

    if not V39_MERGED.exists():
        raise FileNotFoundError(f"v39-merged not found: {V39_MERGED}")
    if not V39_ADAPTER.exists():
        raise FileNotFoundError(f"v39-adapter not found: {V39_ADAPTER}")
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Target adapter not found: {adapter_path}\n"
            "Download from Kaggle first."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # ── Load v39-merged in FP32 for precision ──────────────────────────────
    # CRITICAL: The base recovery via negated LoRA scaling only works precisely
    # in fp32. In fp16, add/subtract the same delta doesn't round-trip exactly —
    # the cumulative weight errors corrupt the recovered base.
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(V39_MERGED))
    print("  Loading v39-merged (CPU fp32 for precision)…")
    model = AutoModelForCausalLM.from_pretrained(
        str(V39_MERGED), torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    print(f"  Base loaded in {time.time()-t0:.1f}s")

    # ── SUBTRACT v39's LoRA delta (negate scaling, in fp32) ───────────────
    print("  Loading v39 adapter with NEGATED scaling to subtract v39's delta…")
    peft_v39 = PeftModel.from_pretrained(model, str(V39_ADAPTER), is_trainable=False)
    negated = 0
    for _name, module in peft_v39.named_modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            for key in module.scaling:
                module.scaling[key] *= -1
                negated += 1
    print(f"  Negated {negated} LoRA scaling entries (will subtract v39 delta)")
    t1 = time.time()
    model = peft_v39.merge_and_unload()
    print(f"  Subtracted v39 in {time.time()-t1:.1f}s → raw base recovered (fp32)")
    del peft_v39; gc.collect()

    # ── ADD new adapter normally, still in fp32 ───────────────────────────
    print(f"  Applying {label} adapter (normal +1.0 scaling, fp32)…")
    t2 = time.time()
    peft_new = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model = peft_new.merge_and_unload()
    print(f"  Applied new adapter in {time.time()-t2:.1f}s")
    del peft_new; gc.collect()

    # ── Cast back to fp16 for storage ─────────────────────────────────────
    print("  Casting merged model to fp16 for storage…")
    model = model.half()
    gc.collect()

    # ── Save ───────────────────────────────────────────────────────────────
    merged_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_out), safe_serialization=True, max_shard_size="1GB")
    tok.save_pretrained(str(merged_out))
    sz = sum(f.stat().st_size for f in merged_out.glob("*.safetensors"))
    print(f"  Saved ({sz/1e9:.2f} GB)")
    del model; gc.collect()
    return merged_out

def step_2_convert(merged_dir: Path, f16_path: Path) -> Path:
    _step("STEP 2 — CONVERT HF → F16 GGUF")
    if f16_path.exists() and f16_path.stat().st_size > 1e9:
        print("  Already exists, skipping."); return f16_path
    subprocess.run(["python", str(LLAMA_CPP/"convert_hf_to_gguf.py"), str(merged_dir),
                    "--outtype", "f16", "--outfile", str(f16_path)], check=True)
    return f16_path

def step_3_quantize(f16: Path, q5_path: Path) -> Path:
    _step("STEP 3 — QUANTIZE F16 → Q5_K_M")
    if q5_path.exists() and q5_path.stat().st_size > 1e8:
        print("  Already exists, skipping."); return q5_path
    subprocess.run([str(LLAMA_CPP/"build"/"bin"/"llama-quantize.exe"),
                    str(f16), str(q5_path), "Q5_K_M"], check=True)
    print(f"\n  ✓ Q5_K_M ready: {q5_path}")
    return q5_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=list(VERSION_CONFIG), required=True,
                        help="Which adapter version to merge (v43v1, v43v2, v44)")
    args = parser.parse_args()

    cfg = VERSION_CONFIG[args.version]
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    label  = cfg["label"]
    f16    = cfg["out_dir"] / f"{label}.f16.gguf"
    q5     = cfg["out_dir"] / f"{label}-Q5_K_M.gguf"

    t0 = time.time()
    merged = step_1_merge_corrected(cfg)
    f16    = step_2_convert(merged, f16)
    q5     = step_3_quantize(f16, q5)
    print(f"\n{label} GGUF ready: {q5}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
