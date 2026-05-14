"""quantize_warmstart_direct.py — Correct merge for warm-start LoRA adapters.

ROOT CAUSE ANALYSIS (2026-05-10):
    Two prior fix attempts failed:

    Attempt 1 — quantize_v43.py (direct on v39-merged):
        W = v39-merged + v43_delta ≈ base + v39_delta + v43_delta ≈ base + 2*v39_delta
        DOUBLE-COUNTING: v43_delta ≈ v39_delta because warm-started from v39 init.

    Attempt 2 — quantize_warmed_adapter.py (negated-scaling subtraction):
        Negated peft_v39.scaling to subtract v39_delta before adding v43_delta.
        FAILED: model generates empty string (immediate EOS) — degenerate weights.
        Root cause unclear (possibly Unsloth's patched PEFT ignores .scaling in
        merge_and_unload, or fp16→fp32 upcast accumulates alignment errors in the
        subtract step). Practical result: model produces zero output.

CORRECT FIX — apply warm-start adapter on ORIGINAL GEMMA BASE:
    The warm-start LoRA A/B matrices at end-of-training encode the FULL delta
    from base (not just the increment from v39). Training started at v39's init
    but the final A, B reflect everything learned (≈ v39_delta for v43v1 since
    same data; ≈ v39_delta + concealed_increment for v43v2).

    W_result = W_gemma_base_fp16 + B_v43 @ A_v43 * scaling

    For v43v1 (same training data as v39):  result ≈ v39-merged → ~v39 perf
    For v43v2 (659 examples, concealed fix): result ≈ v39-merged + concealed_inc

    This is exactly equivalent to how v39 itself was built (merge_and_quantize_v39.py).
    The adapter is applied on the raw Gemma base — no subtract step, no v39-merged
    intermediary needed.

Paths:
    GEMMA_BASE   : D:/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3
    v43v1_adapter: D:/kaggle/results/v43-output/haic-gemma4-v43-concealed-adapter
    v43v2_adapter: D:/kaggle/results/v43-v2-output/haic-gemma4-v43-concealed-adapter

Usage:
    python experiments/quantize_warmstart_direct.py --version v43v1
    python experiments/quantize_warmstart_direct.py --version v43v2
"""
from __future__ import annotations
import argparse, gc, subprocess, time
from pathlib import Path

GEMMA_BASE = Path(
    "D:/models/huggingface/hub/models--google--gemma-4-E2B-it"
    "/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
)
LLAMA_CPP = Path("D:/llama.cpp")

VERSION_CONFIG = {
    # v43v1: warm-start from v39, same training data (649 examples, concealed bug).
    # Expected: ≈ v39 baseline performance — use as diagnostic.
    "v43v1": {
        "adapter": Path("D:/kaggle/results/v43-output/haic-gemma4-v43-concealed-adapter"),
        "out_dir": Path("D:/kaggle/results/v43-v1-direct-gguf"),
        "label": "haic-gemma4-v43v1-direct",
        "note": "warm-start from v39, 649 examples (concealed bug → no concealed training)",
    },
    # v43v2: warm-start from v39, 659 examples (10 concealed fixed).
    # Primary candidate: should improve concealed vs v39 baseline.
    "v43v2": {
        "adapter": Path("D:/kaggle/results/v43-v2-output/haic-gemma4-v43-concealed-adapter"),
        "out_dir": Path("D:/kaggle/results/v43-v2-direct-gguf"),
        "label": "haic-gemma4-v43v2-direct",
        "note": "warm-start from v39, 659 examples (10 concealed, bugfixed)",
    },
    # v45: warm-start from v39 (v42 pattern, proven), 669 examples (25 concealed = v44's 20 + 5 new).
    # H4d: warm-start + 25 concealed → maintain agg ≥ 90% AND push concealed past 67/100.
    # Adapter pending: train on Kaggle (haic-gemma4-v45-concealed notebook).
    "v45": {
        "adapter": Path("D:/kaggle/results/v45-output/haic-gemma4-v45-concealed-adapter"),
        "out_dir": Path("D:/kaggle/results/v45-gguf"),
        "label": "haic-gemma4-v45",
        "note": "warm-start from v39, 669 examples (25 concealed: 20 from v44 + 5 new framings)",
    },
    # v46: DPO on top of v42, training adapter starts from v42 weights and
    # uses 80 preferred (explicit refusal) / rejected (abstract deflection)
    # pairs targeting concealed-compliance. The saved adapter is v42+v46
    # combined (DPO modifies the v42 LoRA in place), so this is applied
    # directly on the Gemma base like any other warm-start.
    "v46": {
        "adapter": Path("D:/kaggle/results/v46-output/haic-gemma4-v46-dpo-adapter"),
        "out_dir": Path("D:/kaggle/results/v46-gguf"),
        "label": "haic-gemma4-v46",
        "note": "DPO on v42 baseline, 80 preferred/rejected pairs targeting concealed-compliance (Option C from canonical_eval verdict)",
    },
    # v50: DPO on top of v42 (loaded DIRECTLY, not via v46), 150 steps, beta=0.1,
    # LR=5e-5, tokenizer-fix applied. 400 pairs from haic-gemma4-v47-dpo-pairs.
    # The D8 baseline diagnostic showed v42 itself leaks on the hypothetical smoke probe;
    # canonical_eval needed to determine whether aggregate concealment metrics improved.
    "v50": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v50-output/haic-gemma4-v50-dpo-adapter"),
        "out_dir": Path("D:/kaggle/results/v50-gguf"),
        "label": "haic-gemma4-v50",
        "note": "DPO on v42 direct (skip v46), 150 steps, beta=0.1, LR=5e-5, tokenizer fix, 400 pairs",
    },
    # v51: Pure SFT (no DPO) on chosen examples only. 100 steps, LR=5e-5,
    # completion_only_loss=True. Warm-start v42 direct. Tests EOS-attractor
    # hypothesis: if v50's collapse was DPO-specific, SFT should preserve
    # capability while still moving explicit_refusal rate.
    "v51": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v51-output/haic-gemma4-v51-sft-adapter"),
        "out_dir": Path("D:/kaggle/results/v51-gguf"),
        "label": "haic-gemma4-v51",
        "note": "SFT on v42 direct, 100 steps, LR=5e-5, completion-only loss, 400 chosen examples",
    },
    # v52: SFT on v42 direct, same as v51 but with SYSTEM_PROMPT prepended to all
    # training prompts. Fixes the Training-Inference Distribution (TID) mismatch
    # identified in v51 root-cause analysis: v51 trained on user-only format while
    # canonical_eval uses system+user format. H10c (empty<=0.05) is the gate.
    "v52": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v52-output/haic-gemma4-v52-sft-adapter"),
        "out_dir": Path("D:/kaggle/results/v52-gguf"),
        "label": "haic-gemma4-v52",
        "note": "SFT on v42 direct, 100 steps, LR=5e-5, system+user format (TID mismatch fix)",
    },
    # v53: proper [system, user] roles in apply_chat_template (60 steps).
    # v52 failed (~10% explicit refusal) because system-prepended-to-user broke the
    # <user_turn>probe</user_turn> pattern at inference. v53 uses true system+user roles.
    # Diagnostic in Cell 7 of the notebook confirms whether HF tokenizer produces
    # separate turns or embeds system into user (same as v52). Also reduces steps
    # 100→60 to test injection-regression over-generalization hypothesis.
    # H11: agg_security>=0.85, explicit>=0.30, empty<=0.05, leak<=0.20.
    "v53": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v53-output/haic-gemma4-v53-sft-adapter"),
        "out_dir": Path("D:/kaggle/results/v53-gguf"),
        "label": "haic-gemma4-v53",
        "note": "SFT on v42 direct, 60 steps, LR=5e-5, proper system+user roles (v52 TID fix corrected)",
    },
    # v54: same proper [system,user] format as v53 but 100 steps (restores v51 step count).
    # Steps confirmed as primary driver: v52 (100 steps, wrong format)=8.4% > v53 (60 steps, correct)=2.7%.
    # H12: agg_security>=0.85, explicit>=0.30, empty<=0.05, leak<=0.20.
    "v54": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v54-output/haic-gemma4-v54-sft-adapter"),
        "out_dir": Path("D:/kaggle/results/v54-gguf"),
        "label": "haic-gemma4-v54",
        "note": "SFT on v42 direct, 100 steps, LR=5e-5, proper system+user roles (steps-as-primary-driver test)",
    },
    # v55: return to the only proven refusal-binding format (v51 user-only)
    # and add 80 held-out injection-positive examples generated from v42 before
    # SFT. H13 tests whether mixed supervision preserves explicit refusal while
    # repairing adversarial/indirect injection regressions.
    "v55": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v55-output/haic-gemma4-v55-sft-mixed-adapter"),
        "out_dir": Path("D:/kaggle/results/v55-gguf"),
        "label": "haic-gemma4-v55",
        "note": "SFT on v42 direct, 100 steps, LR=5e-5, user-only format, 400 refusal + 80 v42 injection-positive examples",
    },
    # v56: narrow data-coverage test after v55 missed the direct-injection
    # floor by 2pp. Keeps user-only format and 100 steps, but increases direct
    # injection positives and adds social-engineering positives.
    "v56": {
        "adapter": Path("C:/Users/benja/AppData/Local/Temp/v56-output/haic-gemma4-v56-sft-targeted-mixed-adapter"),
        "out_dir": Path("D:/kaggle/results/v56-gguf"),
        "label": "haic-gemma4-v56",
        "note": "SFT on v42 direct, 100 steps, LR=5e-5, user-only format, 400 refusal + 160 targeted-positive examples",
    },
}

# NOTE: v44 uses v41 pattern (fresh LoRA on v39-merged, B=0 init), NOT warm-start LoRA load.
# Cell 3 of haic_gemma4_v44_concealed.ipynb: merges v39 into 4-bit base, attaches fresh LoRA.
# Adapter delta = only increment beyond v39-merged. Use quantize_v44.py (applies on v39-merged).


def _step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def step_1_merge(cfg: dict) -> Path:
    adapter_path = cfg["adapter"]
    out_dir      = cfg["out_dir"]
    label        = cfg["label"]
    merged_out   = out_dir / f"{label}-merged"

    _step(f"STEP 1 — APPLY WARM-START ADAPTER ON GEMMA BASE ({label})")
    print(f"  gemma-base : {GEMMA_BASE}")
    print(f"  adapter    : {adapter_path}  (warm-start, applied on original base)")
    print(f"  note       : {cfg['note']}")
    print()
    print("  This bypasses the v39-merged intermediary entirely.")
    print("  Warm-start A/B matrices encode full delta from base after training.")

    if merged_out.exists() and any(merged_out.glob("*.safetensors")):
        sz = sum(f.stat().st_size for f in merged_out.glob("*.safetensors"))
        print(f"  Already exists ({sz/1e9:.2f} GB), skipping."); return merged_out

    if not GEMMA_BASE.exists():
        raise FileNotFoundError(f"Gemma base not found: {GEMMA_BASE}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(GEMMA_BASE))
    print("  Loading Gemma base (CPU fp16)…")
    model = AutoModelForCausalLM.from_pretrained(
        str(GEMMA_BASE),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"  Base loaded in {time.time()-t0:.1f}s")

    print(f"  Applying {label} adapter (is_trainable=False, standard +1.0 scaling)…")
    t1 = time.time()
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model = model.merge_and_unload()
    print(f"  Merged in {time.time()-t1:.1f}s")
    gc.collect()

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
    subprocess.run([
        "python", str(LLAMA_CPP / "convert_hf_to_gguf.py"),
        str(merged_dir), "--outtype", "f16", "--outfile", str(f16_path),
    ], check=True)
    return f16_path


def step_3_quantize(f16: Path, q5_path: Path) -> Path:
    _step("STEP 3 — QUANTIZE F16 → Q5_K_M")
    if q5_path.exists() and q5_path.stat().st_size > 1e8:
        print("  Already exists, skipping."); return q5_path
    subprocess.run([
        str(LLAMA_CPP / "build" / "bin" / "llama-quantize.exe"),
        str(f16), str(q5_path), "Q5_K_M",
    ], check=True)
    print(f"\n  ✓ Q5_K_M ready: {q5_path}")
    return q5_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=list(VERSION_CONFIG), required=True,
                        help="Which adapter version to merge")
    args = parser.parse_args()

    cfg = VERSION_CONFIG[args.version]
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    label  = cfg["label"]
    f16    = cfg["out_dir"] / f"{label}.f16.gguf"
    q5     = cfg["out_dir"] / f"{label}-Q5_K_M.gguf"

    t0 = time.time()
    merged = step_1_merge(cfg)
    f16    = step_2_convert(merged, f16)
    q5     = step_3_quantize(f16, q5)
    print(f"\n{label} GGUF ready: {q5}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
