"""
PRISM geometry trajectory scan — v55 / v56 / v57 / v58 merged models.

Compares quantization_hostility across fine-tuning checkpoints against the
known Tier 3 baseline:
  Gemma 4 E2B base : qh=0.9141 (Tier 3 v11/v12, Kaggle T4, bfloat16)
  v35-gov adapter  : qh=0.9186 (Tier 3 v11/v12)

Outputs: experiments/prism_geometry_trajectory_2026-05-15.json
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("D:/prism/src")))
from prism.geometry import scan_model_geometry  # noqa: E402

# ── known baselines (Tier 3, bfloat16, Kaggle T4) ─────────────────────────────
KNOWN_BASELINES = {
    "gemma-4-e2b-it-base": {
        "quantization_hostility": 0.9141,
        "outlier_ratio": 98.8852,
        "activation_kurtosis": 1124.8143,
        "cardinal_proximity": 0.808,
        "layers_sampled": 20,
        "source": "tier3_v12",
    },
    "v35-gov-adapter": {
        "quantization_hostility": 0.9186,
        "outlier_ratio": 96.2314,
        "activation_kurtosis": 1138.6332,
        "cardinal_proximity": 0.8144,
        "layers_sampled": 20,
        "source": "tier3_v12",
    },
}

# ── local merged models to scan ────────────────────────────────────────────────
MODELS = [
    ("v55", "D:/kaggle/results/v55-gguf/haic-gemma4-v55-merged"),
    ("v56", "D:/kaggle/results/v56-gguf/haic-gemma4-v56-merged"),
    ("v57", "D:/kaggle/results/v57-gguf/haic-gemma4-v57-merged"),
    ("v58", "D:/kaggle/results/v58-gguf/haic-gemma4-v58-merged"),
]

OUT = Path("D:/gemma4good/experiments/prism_geometry_trajectory_2026-05-15.json")


def scan_one(label: str, path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Scanning {label}: {path}")
    print(f"{'='*60}")
    t0 = time.time()

    result = scan_model_geometry(
        path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit=False,  # bfloat16 for consistency with Tier 3 baseline
    )

    elapsed = round(time.time() - t0, 1)
    qh = result["mean_quantization_hostility"]
    print(f"  → qh={qh:.4f}  ({elapsed}s)")
    print(f"     outlier_ratio={result['layers'][0].get('outlier_ratio','?'):.4f}  "
          f"n_layers={result['n_layers']}")

    return {
        "label": label,
        "path": path,
        "quantization_hostility": round(qh, 4),
        "n_layers": result["n_layers"],
        "n_hostile_layers": result["n_hostile_layers"],
        "worst_layer_hostility": round(result["worst_layer_hostility"], 4),
        "best_layer_hostility": round(result["best_layer_hostility"], 4),
        "elapsed_s": elapsed,
    }


def main():
    results = []

    for label, path in MODELS:
        if not Path(path).exists():
            print(f"SKIP {label}: path not found — {path}")
            continue

        entry = scan_one(label, path)
        results.append(entry)

        # free GPU/CPU memory before next model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── assemble output ────────────────────────────────────────────────────────
    out = {
        "scan_date": "2026-05-15",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "load_mode": "bfloat16_device_map_auto",
        "known_baselines": KNOWN_BASELINES,
        "scanned": results,
        "summary": [],
    }

    baseline_qh = KNOWN_BASELINES["gemma-4-e2b-it-base"]["quantization_hostility"]
    for r in results:
        delta = round(r["quantization_hostility"] - baseline_qh, 4)
        out["summary"].append({
            "label": r["label"],
            "qh": r["quantization_hostility"],
            "delta_vs_base": delta,
            "direction": "↓ improved" if delta < -0.01
                         else ("↑ worsened" if delta > 0.01 else "≈ unchanged"),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n✓ Saved to {OUT}")

    # ── print summary table ────────────────────────────────────────────────────
    print("\nGeometry trajectory vs base (qh=0.9141):")
    print(f"{'Model':<12} {'qh':>8} {'Δ vs base':>12} {'Direction'}")
    print("-" * 48)
    print(f"{'base':<12} {baseline_qh:>8.4f} {'—':>12}")
    print(f"{'v35-gov':<12} {0.9186:>8.4f} {'+0.0045':>12}  ↑ worsened")
    for s in out["summary"]:
        sign = "+" if s["delta_vs_base"] >= 0 else ""
        print(f"{s['label']:<12} {s['qh']:>8.4f} {sign+str(s['delta_vs_base']):>12}  {s['direction']}")


if __name__ == "__main__":
    main()
