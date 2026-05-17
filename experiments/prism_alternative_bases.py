"""
prism_alternative_bases.py — PRISM geometry scan on alternative base models.

Per docs/passing_model_demo_plan.md Phase 1, this is the cheap pre-screen
that asks: do any open base models have ``quantization_hostility <= 0.72``,
the threshold the Viability Condition treats as the boundary between
viable and non-viable for the Convention's SGT-formatted training?

We measure ``mean_quantization_hostility`` via PRISM's ``scan_model_geometry``
on each candidate. The known baselines (committed in
``experiments/prism_geometry_trajectory_2026-05-15.json``) are:

  Gemma 4 E2B base                       qh = 0.9141  (Tier 3, Kaggle T4)
  v55 / v56 / v57 / v58 merged           qh = 0.9122-0.9127 (BEAST, local scan)

The Viability Condition threshold for SGT-style adoption is qh <= 0.72,
based on the PRISM README v2 adapter case study where qh = 0.7398 was
the only checkpoint that produced behaviorally adequate SGT scores AND
satisfied the viability inequality.

Outputs: ``experiments/prism_alternative_bases_<DATE>.json`` with one
entry per candidate. No promotion claim made on the basis of qh alone;
this is the Phase 1 pre-screen specified in the H-passing-1 plan. A
qh-passing candidate then advances to Phase 2 (small LoRA fine-tune)
and Phase 3 (canonical eval) — see the H-passing-1 plan.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("D:/prism/src")))
from prism.geometry import scan_model_geometry  # noqa: E402

# Send the HF cache to D: so we don't fill C:
os.environ.setdefault("HF_HOME", "D:/models/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CANDIDATES = [
    # Smaller first — cheapest data points
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7B"),
    # Compare against the published baselines (no need to re-scan Gemma 4
    # base — already in prism_geometry_trajectory_2026-05-15.json)
]

KNOWN_BASELINES = {
    "gemma-4-e2b-it": {"qh": 0.9141, "source": "tier3_v12_kaggle"},
    "haic-gemma4-v35-gov": {"qh": 0.9186, "source": "tier3_v12_kaggle"},
    "haic-gemma4-v55-merged": {"qh": 0.9125, "source": "prism_geometry_trajectory"},
    "haic-gemma4-v56-merged": {"qh": 0.9125, "source": "prism_geometry_trajectory"},
    "haic-gemma4-v57-merged": {"qh": 0.9127, "source": "prism_geometry_trajectory"},
    "haic-gemma4-v58-merged": {"qh": 0.9122, "source": "prism_geometry_trajectory"},
}

VIABILITY_THRESHOLD = 0.72

OUT = Path(f"D:/gemma4good/experiments/prism_alternative_bases_{time.strftime('%Y-%m-%d')}.json")


def scan_one(model_id: str, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Scanning {label}: {model_id}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = scan_model_geometry(
            model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=False,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.time() - t0, 1)
        print(f"  → ERROR ({elapsed}s): {type(exc).__name__}: {exc}")
        return {
            "label": label,
            "model_id": model_id,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "elapsed_s": elapsed,
        }

    elapsed = round(time.time() - t0, 1)
    qh = result["mean_quantization_hostility"]
    n_layers = result["n_layers"]
    n_hostile = result["n_hostile_layers"]
    print(f"  → qh={qh:.4f}  n_layers={n_layers}  n_hostile={n_hostile}  ({elapsed}s)")
    print(f"     verdict: {'VIABLE' if qh <= VIABILITY_THRESHOLD else 'VIOLATED'} "
          f"(threshold {VIABILITY_THRESHOLD})")
    return {
        "label": label,
        "model_id": model_id,
        "quantization_hostility": round(qh, 4),
        "n_layers": n_layers,
        "n_hostile_layers": n_hostile,
        "worst_layer_hostility": round(result["worst_layer_hostility"], 4),
        "best_layer_hostility": round(result["best_layer_hostility"], 4),
        "viability_verdict": "VIABLE" if qh <= VIABILITY_THRESHOLD else "VIOLATED",
        "viability_threshold": VIABILITY_THRESHOLD,
        "elapsed_s": elapsed,
    }


def main():
    results = []

    for model_id, label in CANDIDATES:
        entry = scan_one(model_id, label)
        results.append(entry)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {
        "scan_date": time.strftime("%Y-%m-%d"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "load_mode": "bfloat16_device_map_auto",
        "viability_threshold": VIABILITY_THRESHOLD,
        "known_baselines": KNOWN_BASELINES,
        "scanned": results,
        "summary": [],
    }
    for r in results:
        if "error" in r:
            out["summary"].append({"label": r["label"], "error": True})
        else:
            out["summary"].append({
                "label": r["label"],
                "qh": r["quantization_hostility"],
                "verdict": r["viability_verdict"],
            })

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n✓ Saved to {OUT}\n")

    print("\n══════════════════════════════════════════════════════════")
    print("Geometry pre-screen summary (Phase 1 of H-passing-1)")
    print(f"  Threshold: qh ≤ {VIABILITY_THRESHOLD}")
    print("══════════════════════════════════════════════════════════")
    print(f"{'Model':<30} {'qh':>8}   Verdict")
    print("-" * 60)
    for label, info in KNOWN_BASELINES.items():
        print(f"{label:<30} {info['qh']:>8.4f}   {'VIABLE' if info['qh'] <= VIABILITY_THRESHOLD else 'VIOLATED':<10}  (baseline)")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<30} {'ERR':>8}   {r['error'][:30]}")
        else:
            print(f"{r['label']:<30} {r['quantization_hostility']:>8.4f}   {r['viability_verdict']:<10}  (this scan)")


if __name__ == "__main__":
    main()
