import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebook" / "haic_gemma4_governance.ipynb"


UPDATED_HARDWARE_CELL = """---
## Deployment Notes

Earlier local Gemma 4 deployment work hit a real architecture constraint: the
`per_layer_token_embd.weight` tensor is enormous, and the older `v1` export path
could not get enough of the model onto BEAST's RTX 2080 to stay production-fast.
That is why the first Gemma 4 experiments looked deployment-blocked even when the
interview quality was strong.

That is **no longer the current state**. The project now has a validated local
Gemma 4 deployment path for the public `0.1` release.

The practical lesson is not that Gemma 4 was impossible to run locally, but that
artifact discipline mattered. Once the exact adapter, merge, GGUF conversion, and
quantization path were lined up correctly, the model became deployable on the same
consumer workstation this notebook discusses.
"""


UPDATED_AUDIT_CELL = """# --- Scenario 0 - Gemma4Good 0.1 release snapshot ------------------------------
# Lightweight ledger snapshot of the public Gemma4Good 0.1 release.
# This cell intentionally avoids re-deriving the PRISM composite inline. The
# canonical numeric sources live in the archived result bundle and benchmark JSON.

PUBLIC_RELEASE = {
    "training": {
        "loss_final": 0.4645,
        "steps": 290,
        "epochs": 2,
        "examples": 577,
        "lora_r": 16,
        "lora_scope": "model.language_model.layers.* only (205 LoRA modules, 0 vision/audio)",
    },
    "prism": {
        "quantization_hostility": 0.8706,
        "mean_activation_kurtosis": 673.02,
        "max_activation_kurtosis": 1227.76,
        "mean_activation_norm": 64.5921,
        "num_layers_sampled": 108,
        "data_status": "verified_partial",
    },
    "sgt_any_turn": {
        "score": 10.0,
        "security_fails": 0,
        "pivot_count": 3,
        "eval_method": "2-turn any-turn scoring (PIVOT in T1 or T2 counts)",
    },
    "deployment": {
        "artifact": "external quantized runtime artifact",
        "artifact_size_gb": 3.62,
        "quant": "Q5_K_M",
        "throughput_reference_tps": 30.1,
        "release": "0.1",
        "benchmark_json": "external archive artifact: v35_beast_benchmark_with_prompt.json",
    },
}

trace = GovernanceTrace("gemma4good_0_1_release_snapshot")
trace.record("training_metadata", {}, PUBLIC_RELEASE["training"])
trace.record("prism_metadata", {}, PUBLIC_RELEASE["prism"])
trace.record("sgt_metadata", {}, PUBLIC_RELEASE["sgt_any_turn"])
trace.record("deployment_metadata", {}, PUBLIC_RELEASE["deployment"])
v35_snapshot_receipt = trace.finalize()

print("=" * 70)
print("GEMMA4GOOD 0.1 RELEASE SNAPSHOT")
print("=" * 70)
print(f"Training loss         : {PUBLIC_RELEASE['training']['loss_final']}")
print(f"Any-turn SGT          : {PUBLIC_RELEASE['sgt_any_turn']['score']}/10")
print(f"Security fails        : {PUBLIC_RELEASE['sgt_any_turn']['security_fails']}")
print(f"Quant hostility (qh)  : {PUBLIC_RELEASE['prism']['quantization_hostility']}")
print(f"Reference TPS         : {PUBLIC_RELEASE['deployment']['throughput_reference_tps']}")
print(f"Artifact              : {PUBLIC_RELEASE['deployment']['artifact']}")
print(f"Release               : {PUBLIC_RELEASE['deployment']['release']}")
print()
print("Merkle-signed receipt:")
print(f"  receipt_id          : {v35_snapshot_receipt['receipt_id']}")
print(f"  merkle_root         : {v35_snapshot_receipt['merkle_root']}")
print(f"  zk_digest           : {v35_snapshot_receipt['zk_digest']}")
print(f"  leaf_count          : {v35_snapshot_receipt['leaf_count']}")
print("=" * 70)
"""


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "## The Hardware Reality: AltUp Architecture Limits" in source or "## Deployment Notes" in source:
            cell["source"] = [line + "\n" for line in UPDATED_HARDWARE_CELL.strip("\n").split("\n")]
        elif "# --- Scenario 0 - v34 self-audit" in source or "# --- Scenario 0 - v35-gov deployment snapshot" in source or "# --- Scenario 0 - Gemma4Good 0.1 release snapshot" in source:
            cell["source"] = [line + "\n" for line in UPDATED_AUDIT_CELL.strip("\n").split("\n")]

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Updated notebook state in {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
