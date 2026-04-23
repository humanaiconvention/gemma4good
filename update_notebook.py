import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebook" / "haic_gemma4_governance.ipynb"


UPDATED_HARDWARE_CELL = """---
## The Hardware Reality: AltUp Architecture Limits

Earlier local Gemma 4 deployment work hit a real architecture constraint: the
`per_layer_token_embd.weight` tensor is enormous, and the older `v1` export path
could not get enough of the model onto BEAST's RTX 2080 to stay production-fast.
That is why the first Gemma 4 experiments looked deployment-blocked even when the
interview quality was strong.

That is **no longer the current state**. The HAIC interviewer line now has two
working local Gemma 4 Q5_K_M deployments on BEAST:

- `haic-gemma4-v34-Q5_K_M.gguf` as the immediate rollback target
- `haic-gemma4-v35-gov-Q5_K_M.gguf` as the current promoted runtime

The practical lesson is not that Gemma 4 was impossible to run locally, but that
artifact discipline mattered. Once the exact adapter, merge, GGUF conversion, and
quantization path were lined up correctly, the model became deployable on the same
consumer workstation this notebook discusses.
"""


UPDATED_AUDIT_CELL = """# --- Scenario 0 - v35-gov deployment snapshot ---------------------------------
# Lightweight ledger snapshot of the promoted HAIC Gemma 4 interviewer.
# This cell intentionally avoids re-deriving the PRISM composite inline. The
# canonical numeric sources now live in the archived Kaggle result bundle and the
# local BEAST benchmark JSON.

V35_PROMOTED = {
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
        "gguf_path": "external runtime artifact: haic-gemma4-v35-gov-Q5_K_M.gguf",
        "gguf_size_gb": 3.62,
        "quant": "Q5_K_M",
        "tps_rtx2080_prompted": 30.1,
        "port": 8081,
        "replaces": "haic-gemma4-v34-Q5_K_M.gguf",
        "deployment_date": "2026-04-21",
        "benchmark_json": "external archive artifact: v35_beast_benchmark_with_prompt.json",
    },
}

trace = GovernanceTrace("v35_gov_promoted_snapshot")
trace.record("training_metadata", {}, V35_PROMOTED["training"])
trace.record("prism_metadata", {}, V35_PROMOTED["prism"])
trace.record("sgt_metadata", {}, V35_PROMOTED["sgt_any_turn"])
trace.record("deployment_metadata", {}, V35_PROMOTED["deployment"])
v35_snapshot_receipt = trace.finalize()

print("=" * 70)
print("v35-gov DEPLOYMENT SNAPSHOT")
print("=" * 70)
print(f"Training loss         : {V35_PROMOTED['training']['loss_final']}")
print(f"Any-turn SGT          : {V35_PROMOTED['sgt_any_turn']['score']}/10")
print(f"Security fails        : {V35_PROMOTED['sgt_any_turn']['security_fails']}")
print(f"Quant hostility (qh)  : {V35_PROMOTED['prism']['quantization_hostility']}")
print(f"BEAST prompted TPS    : {V35_PROMOTED['deployment']['tps_rtx2080_prompted']}")
print(f"Live GGUF             : {V35_PROMOTED['deployment']['gguf_path']}")
print(f"Rollback              : {V35_PROMOTED['deployment']['replaces']}")
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
        if "## The Hardware Reality: AltUp Architecture Limits" in source:
            cell["source"] = [line + "\n" for line in UPDATED_HARDWARE_CELL.strip("\n").split("\n")]
        elif "# --- Scenario 0 - v34 self-audit" in source:
            cell["source"] = [line + "\n" for line in UPDATED_AUDIT_CELL.strip("\n").split("\n")]

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Updated notebook state in {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
