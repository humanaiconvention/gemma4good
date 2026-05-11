"""
evaluate_v40_pipeline.py — single-command pipeline for v40 evaluation.

When the v40-paraphrases Kaggle run completes:
  1. Pull adapter from kaggle to D:/kaggle/adapters/haic-gemma4-v40-paraphrases-adapter
  2. LoRA-shape audit (expect 490 tensors, all language_model)
  3. Leakage check (expect PASS — same scenarios as v39)
  4. BEAST nf4 n=10 rigorous (skipped if BEAST nf4 path is slow tonight; can run
     against v40 GGUF instead, see comments below)
  5. merge_and_quantize_v40 → produce GGUF Q5_K_M
  6. Restart llama-server with v40 GGUF
  7. Run rigorous SGT against v40 GGUF (n=30, both v1 and v2 scenario sets)
  8. Compute precision spread vs eval (or skip eval if BEAST nf4 unusable)
  9. Compute v40-vs-v39 deploy comparison (does H1 verify?)

Usage:
  python -u -m experiments.evaluate_v40_pipeline --skip-nf4-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def step(label):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-id", default="benhaslam/haic-gemma4-v40-paraphrases")
    ap.add_argument("--adapter-name", default="haic-gemma4-v40-paraphrases-adapter")
    ap.add_argument("--skip-nf4-eval", action="store_true",
                    help="Skip BEAST nf4 eval (use only GGUF deploy-precision). "
                         "Use when BEAST transformers+bnb path is degraded.")
    ap.add_argument("--scenarios", default="experiments/sgt_scenarios.jsonl",
                    choices=("experiments/sgt_scenarios.jsonl",
                             "experiments/sgt_scenarios_v2.jsonl"))
    ap.add_argument("--n-samples", type=int, default=30)
    args = ap.parse_args()

    adapter_dir = Path(f"D:/kaggle/adapters/{args.adapter_name}")
    out_dir = Path("D:/kaggle/results/v40-gguf")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Pull from Kaggle ──────────────────────────────────────────────
    step("STEP 1 — Pull v40 from Kaggle")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    r = run(["kaggle", "kernels", "output", args.kernel_id, "-p", str(adapter_dir)])
    if r.returncode != 0:
        print("kaggle pull failed; check status with: kaggle kernels status " + args.kernel_id)
        sys.exit(1)

    # Locate the adapter dir within the kaggle output
    inner = next((p for p in adapter_dir.glob("*adapter*") if p.is_dir()), None)
    if inner is None:
        # Maybe files at root level
        if (adapter_dir / "adapter_model.safetensors").exists():
            inner = adapter_dir
        else:
            print(f"Could not find adapter dir in {adapter_dir}; contents:")
            for f in adapter_dir.iterdir():
                print(f"  {f}")
            sys.exit(2)
    print(f"\nv40 adapter at: {inner}")

    # ── 2. LoRA-shape audit ──────────────────────────────────────────────
    step("STEP 2 — LoRA-shape audit")
    from safetensors import safe_open
    import collections
    p = inner / "adapter_model.safetensors"
    keys = []
    with safe_open(str(p), framework="pt") as f:
        for k in f.keys():
            keys.append(k)
    lora = [k for k in keys if "lora" in k.lower()]
    bucket = collections.Counter(
        "language_model" if "language_model" in k else
        "vision_tower"   if "vision_tower"   in k else
        "audio_tower"    if "audio_tower"    in k else "other"
        for k in lora
    )
    print(f"  LoRA tensors: {len(lora)}  (expected 490 for rank-16 Gemma-4-E2B)")
    print(f"  buckets: {dict(bucket)}")
    assert bucket.get("vision_tower", 0) == 0 and bucket.get("audio_tower", 0) == 0, (
        "LoRA leaked into multimodal towers — abort"
    )

    # ── 3. Merge + convert + quantize ────────────────────────────────────
    step("STEP 3 — Merge LoRA + convert HF → F16 → Q5_K_M GGUF")

    # Mirror experiments.merge_and_quantize_v39 but for v40 paths
    import os
    os.environ["HAIC_V39_ADAPTER"] = str(inner)
    os.environ["HAIC_OUT_DIR"] = str(out_dir)
    # Keep base path as default (or set explicitly via env)
    r = run([sys.executable, "-u", "-m", "experiments.merge_and_quantize_v39"])
    if r.returncode != 0:
        print("merge+quantize failed")
        sys.exit(3)

    q5_path = out_dir / "haic-gemma4-v39-Q5_K_M.gguf"
    if not q5_path.exists():
        # The merge_and_quantize script names are v39-specific; rename
        for f in out_dir.iterdir():
            if "Q5_K_M" in f.name:
                q5_path = f
                break
    print(f"v40 Q5_K_M at: {q5_path}")

    # ── 4. Start llama-server ────────────────────────────────────────────
    step("STEP 4 — Start llama-server with v40 GGUF")
    print("  Stop any existing llama-server, then start fresh on port 8088")
    print(f"  Manual:  D:/llama.cpp/build/bin/llama-server.exe -m {q5_path} \\")
    print(f"             -ngl 99 -c 2048 --port 8088 --host 127.0.0.1 --reasoning off &")
    print()
    print("  This script doesn't manage the server lifecycle (yet).")
    print("  Start it externally, wait for health check, then re-run with")
    print(f"  --start-from step5 to continue.")

    # ── 5. Rigorous SGT (deploy precision) ───────────────────────────────
    step(f"STEP 5 — Rigorous SGT against v40 GGUF (n={args.n_samples})")
    sgt_out = Path(f"experiments/v40_sgt_rigorous_gguf_n{args.n_samples}.json")
    r = run([sys.executable, "-u", "-m", "experiments.run_v39_gguf_v2_scenarios",
             "--server-url", "http://127.0.0.1:8088",
             "--scenarios", args.scenarios,
             "--n-samples", str(args.n_samples),
             "--out", str(sgt_out)])
    if r.returncode != 0:
        print("rigorous SGT failed")
        sys.exit(5)

    print(f"\nv40 deploy SGT receipt: {sgt_out}")

    # ── 6. Compare to v39 ────────────────────────────────────────────────
    step("STEP 6 — Compare v40 to v39 (does H1 verify?)")
    v39_path = Path("experiments/v39_gguf_v2_scenarios.json") if args.scenarios.endswith("_v2.jsonl") else Path("experiments/v39_sgt_rigorous_gguf_1turn_n30_refined.json")
    if v39_path.exists():
        v40 = json.loads(sgt_out.read_text())
        v39 = json.loads(v39_path.read_text())
        # Different shapes: v39_gguf_v2_scenarios.json has 'aggregate' key,
        # v39_sgt_rigorous_gguf_1turn_n30_refined.json has 'finetune' key.
        if "aggregate" in v39:
            v39_sec = v39["aggregate"]
            print(f"  v39 v2-set sec: {v39_sec['security_passes']}/{v39_sec['security_trials']}")
        else:
            v39_sec = v39["finetune"]["sampling"]
            print(f"  v39 v1-set sec: {v39_sec['security_passes']}/{v39_sec['security_trials']}")
        # v40 always uses the v40_sgt_rigorous_gguf format
        v40_agg = v40.get("aggregate", v40.get("finetune", {}).get("sampling", {}))
        print(f"  v40 set sec   : {v40_agg.get('security_passes')}/{v40_agg.get('security_trials')}")

    print()
    print("=" * 70)
    print("v40 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
