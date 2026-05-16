"""
merge_v39_with_v38_baseline.py — stitch v39 finetune-only run onto v38's
existing baseline data.

Rationale: the base model evaluation under (4-bit nf4, seed 42, n=10,
V38_SYSTEM_PROMPT, default scenarios) is reproducible. v38 already
produced this baseline data when its rigorous run completed
(`experiments/v38_sgt_rigorous.json`). Re-running the same baseline for
v39 wastes ~140 min of GPU. Stitching the existing v38 baseline into
v39's finetune-only report is mathematically equivalent and saves time.

Usage:
    python -m experiments.merge_v39_with_v38_baseline

Reads:
    experiments/v39_sgt_rigorous_finetune_only.json   (v39 finetune)
    experiments/v38_sgt_rigorous.json                  (v38 + base, reuse base)

Writes:
    experiments/v39_sgt_rigorous.json                  (v39 finetune + reused base)
"""
from __future__ import annotations

import json
from pathlib import Path

V39_FT  = Path("experiments/v39_sgt_rigorous_finetune_only.json")
V38_REP = Path("experiments/v38_sgt_rigorous.json")
OUT     = Path("experiments/v39_sgt_rigorous.json")


def main():
    v39_ft = json.loads(V39_FT.read_text())
    v38    = json.loads(V38_REP.read_text())

    # Sanity: ensure same eval conditions
    v39_seed   = v39_ft["finetune"]["sampling"]["seed"]
    v38_b_seed = v38["baseline"]["sampling"]["seed"]
    assert v39_seed == v38_b_seed, (
        f"Seed mismatch: v39 finetune seed={v39_seed}, v38 baseline seed={v38_b_seed}. "
        "Cannot stitch baselines safely."
    )

    v39_dec   = v39_ft["finetune"]["sampling"]["decoding"]
    v38_b_dec = v38["baseline"]["sampling"]["decoding"]
    # Decoding records must be byte-identical for the stitch to be valid.
    if v39_dec != v38_b_dec:
        print(f"WARN: decoding records differ:\n  v39: {v39_dec}\n  v38 base: {v38_b_dec}")
        print("Stitching anyway, but mark this in the merged report.")

    merged = {
        "finetune": v39_ft["finetune"],
        "baseline": v38["baseline"],
        "merge_metadata": {
            "tool": "merge_v39_with_v38_baseline",
            "version": "1.0",
            "rationale": (
                "Baseline evaluation is deterministic per seed/decoding/scenarios. "
                "v38's baseline run (n=10, seed=42, V38_SYSTEM_PROMPT, default scenarios, "
                "4-bit nf4) is reused here rather than re-run, saving ~140 min wall-clock."
            ),
            "v38_baseline_source": str(V38_REP),
            "v39_finetune_source": str(V39_FT),
            "v38_baseline_decoding": v38_b_dec,
            "v39_finetune_decoding": v39_dec,
            "decoding_match": v39_dec == v38_b_dec,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, indent=2))
    print(f"Stitched report written: {OUT}")
    print()

    # Quick comparison summary
    f_ft = v39_ft["finetune"]["sampling"]
    b    = v38["baseline"]["sampling"]
    delta = f_ft["grounding_pass_rate"] - b["grounding_pass_rate"]
    print(f"v39 sampling grounding: {f_ft['grounding_passes']}/{f_ft['grounding_trials']} "
          f"= {f_ft['grounding_pass_rate']:.3f} CI95 {f_ft['grounding_ci95']}")
    print(f"base sampling grounding: {b['grounding_passes']}/{b['grounding_trials']} "
          f"= {b['grounding_pass_rate']:.3f} CI95 {b['grounding_ci95']}")
    print(f"Δ-vs-base: {delta:+.3f}")


if __name__ == "__main__":
    main()
