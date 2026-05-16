"""build_v50_nb.py — Build v50 DPO training notebook from v49 template.

## v49 post-mortem (H7 REFUTED — but D7 succeeded)

D7 diagnostic FIRED: "BASELINE LEAKS ❌" — the v46 warm-start adapter was
already leaking BEFORE any DPO training ran. Post-training (300 steps,
LR=1e-4) produced byte-identical leak. This confirms v47/v48/v49 all
stacked DPO on a broken foundation.

The v46 adapter (rank 16, 48MB, BF16) lost v42's concealment training.

## v50 fix — H8

Skip v46 entirely. Load v42 adapter DIRECTLY (rank 16, 101MB, F32, 490 tensors).

New Kaggle dataset uploaded:
  benhaslam/haic-gemma4-v42-concealed-adapter

The glob logic in cell 5 already prefers paths containing "v42", so it will
auto-select this dataset over any other adapter (and we remove v46 from
kernel_sources to be safe).

## v50 config

Revert to CONSERVATIVE training settings — isolate the warm-start as the
only variable vs v48:
- max_steps: 300 → **150** (back to v48; if baseline is good, less training needed)
- learning_rate: 1e-4 → **5e-5** (back to v48)
- beta: 0.1 (unchanged)
- Tokenizer fix: kept (apply_chat_template)
- D7 baseline probe: kept (must confirm v42 REFUSES at baseline)

## Hypothesis H8 (v50)

Loading v42 adapter DIRECTLY (not v46) as warm-start + same v48 DPO config
+ D7 baseline diagnostic will yield:
- D7: BASELINE REFUSES ✅ (v42 has 88% aggregate security — should refuse probe)
- H8a: aggregate_security ≥ 0.88
- H8b: strict_concealed_refusal ≥ 0.50
- H8c: strict_concealed_leak ≤ 0.10

Falsification path: if D7 still shows LEAKS with v42, the entire concealment
training story is wrong — v42's 88% aggregate doesn't cover hypothetical
framing. Pivot to SFT on chosen examples (H9).

## Usage

    python experiments/build_v50_nb.py
    # Then: kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v50-dpo/
"""

import json
import copy
from pathlib import Path

SRC     = "D:/kaggle/notebooks/haic-gemma4-v49-dpo/haic_gemma4_v49_dpo.ipynb"
DST_DIR = Path("D:/kaggle/notebooks/haic-gemma4-v50-dpo")
DST     = DST_DIR / "haic_gemma4_v50_dpo.ipynb"

DST_DIR.mkdir(parents=True, exist_ok=True)

nb49 = json.load(open(SRC, encoding="utf-8"))
nb50 = copy.deepcopy(nb49)
cells = nb50["cells"]


def set_src(cell, lines):
    cell["source"] = lines if isinstance(lines, list) else [lines]


# ── Cell 0: Markdown — update hypothesis to H8 ───────────────────────────────
set_src(cells[0], [
    "# HAIC Gemma 4 v50 — DPO from v42 DIRECTLY (skip v46)\n",
    "\n",
    "**Hypothesis H8 (v50):**\n",
    "Loading **v42 adapter directly** (not the degraded v46 adapter) as warm-start,\n",
    "with same v48 DPO config (150 steps, LR=5e-5, beta=0.1, tokenizer fix)\n",
    "and D7 baseline diagnostic, will pass all concealment predicates.\n",
    "\n",
    "**v49 result (H7 REFUTED — but D7 succeeded):**\n",
    "D7 baseline probe fired: 'BASELINE LEAKS ❌' BEFORE DPO ran.\n",
    "The v46 warm-start was already broken. Every prior DPO experiment\n",
    "(v47/v48/v49) was stacking gradient updates on a leaking foundation.\n",
    "300 steps + 2× LR did not flip the argmax because the starting point\n",
    "was wrong, not the training.\n",
    "\n",
    "**v50 critical change:**\n",
    "Load **v42 adapter** (rank 16, 101MB, F32, 490 tensors —\n",
    "the original concealment-trained model with 88% aggregate security)\n",
    "DIRECTLY from new Kaggle dataset `benhaslam/haic-gemma4-v42-concealed-adapter`.\n",
    "Skip v46 entirely.\n",
    "\n",
    "**Config (reverted to v48 conservative):**\n",
    "- max_steps: 150 (was 300 in v49) — isolate warm-start as only variable\n",
    "- learning_rate: 5e-5 (was 1e-4 in v49)\n",
    "- beta: 0.1 (unchanged)\n",
    "- Tokenizer fix: kept (0 mismatch warnings expected)\n",
    "- D7 baseline probe: kept\n",
    "\n",
    "**Diagnostic predicate D8:**\n",
    "- D8: baseline probe shows REFUSAL with v42 adapter\n",
    "  (If D8 fails, v42's 88% aggregate doesn't cover hypothetical framing —\n",
    "   pivot to SFT on chosen examples or new pair distribution)\n",
    "\n",
    "**Falsifiable predicates (committed before training):**\n",
    "- H8a: aggregate_security >= 0.88\n",
    "- H8b: strict_concealed_refusal >= 0.50  ← primary target\n",
    "- H8c: strict_concealed_leak <= 0.10\n",
    "\n",
    "See `docs/v49_verdict_2026-05-12.md` for v49 post-mortem.\n",
])

# ── Cell 9: D7 baseline probe — rename diagnostic to D8 ──────────────────────
src9 = "".join(cells[9]["source"])
src9 = src9.replace("D7", "D8").replace("H7", "H8").replace("H8 likely", "H8 likely")
# Update fallback message to reference H9 (SFT pivot) instead of H8 (which is now this)
src9 = src9.replace(
    "v46 DPO undid v42 security?",
    "v42 itself may not cover hypothetical framing?"  # avoid apostrophe — breaks single-quoted print()
).replace(
    "fallback = reload v42 directly as warm-start (H8)",
    "fallback = SFT on chosen examples or new pairs (H9)"
)
set_src(cells[9], src9)

# ── Cell 11: DPO training — revert to v48 conservative config ────────────────
set_src(cells[11], [
    "from trl import DPOTrainer, DPOConfig\n",
    "\n",
    "OUTPUT_DIR = '/kaggle/working/haic-gemma4-v50-dpo-adapter'\n",
    "\n",
    "# v50 changes vs v49:\n",
    "#   - max_steps: 300 → 150  (revert to v48; isolate warm-start as only variable)\n",
    "#   - learning_rate: 1e-4 → 5e-5  (revert to v48 conservative)\n",
    "#\n",
    "# Unchanged from v48/v49 (all confirmed working):\n",
    "#   - beta=0.1\n",
    "#   - apply_chat_template prompts (0 tokenizer mismatch warnings)\n",
    "#   - tokenizer.padding_side = 'left'\n",
    "#   - per_device_train_batch_size=2, gradient_accumulation_steps=4 → eff_batch=8\n",
    "#   - fp16=False mandatory (T4 grad scaler bug with LoRA)\n",
    "#   - optim='adamw_torch' mandatory (adamw_8bit NotImplementedError on T4)\n",
    "#\n",
    "# Critical change (cell 5 model load): now uses v42 adapter directly\n",
    "# via new Kaggle dataset benhaslam/haic-gemma4-v42-concealed-adapter.\n",
    "dpo_config = DPOConfig(\n",
    "    output_dir=OUTPUT_DIR,\n",
    "    max_steps=150,\n",
    "    warmup_steps=15,\n",
    "    per_device_train_batch_size=2,\n",
    "    gradient_accumulation_steps=4,\n",
    "    gradient_checkpointing=True,\n",
    "    learning_rate=5e-5,\n",
    "    beta=0.1,\n",
    "    logging_steps=10,\n",
    "    save_strategy='no',\n",
    "    save_total_limit=1,\n",
    "    optim='adamw_torch',\n",
    "    fp16=False,\n",
    "    remove_unused_columns=False,\n",
    "    report_to='none',\n",
    ")\n",
    "\n",
    "trainer = DPOTrainer(\n",
    "    model=model,\n",
    "    ref_model=None,  # PEFT uses disabled base adapter as reference\n",
    "    args=dpo_config,\n",
    "    train_dataset=ds,\n",
    "    processing_class=tokenizer,\n",
    ")\n",
    "\n",
    "print(f'Starting v50 DPO: {dpo_config.max_steps} steps, beta={dpo_config.beta}, '\n",
    "      f'lr={dpo_config.learning_rate}, '\n",
    "      f'effective_batch={dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}')\n",
    "print('v50 critical change: warm-start = v42 adapter DIRECTLY (skip v46)')\n",
    "print('v48 fixes retained: chat-template prompts + beta=0.1 + conservative LR')\n",
    "trainer.train()\n",
])

# ── Cell 13: Save — update path ──────────────────────────────────────────────
src13 = "".join(cells[13]["source"])
src13 = src13.replace("v49", "v50")
set_src(cells[13], src13)

# ── Cell 15: Smoke test — update labels ──────────────────────────────────────
src15 = "".join(cells[15]["source"])
src15 = src15.replace("v49", "v50").replace("H7", "H8")
set_src(cells[15], src15)

# ── Last cell: next steps markdown ───────────────────────────────────────────
src_last = "".join(cells[-1]["source"])
src_last = src_last.replace("v49", "v50").replace("H7", "H8")
set_src(cells[-1], src_last)

# ── Save notebook ─────────────────────────────────────────────────────────────
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb50, f, ensure_ascii=False, indent=1)
print(f"Saved: {DST}")

# ── Write kernel-metadata.json ────────────────────────────────────────────────
# Critical change: REMOVE v46 from kernel_sources, ADD v42 adapter dataset
meta = {
    "id": "benhaslam/haic-gemma4-v50-dpo",
    "title": "HAIC Gemma4 v50 DPO from v42 direct",
    "code_file": "haic_gemma4_v50_dpo.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "benhaslam/haic-gemma4-v47-dpo-pairs",            # same 400 pairs
        "benhaslam/haic-gemma4-v42-concealed-adapter",    # v42 adapter — NEW
    ],
    "competition_sources": [],
    "kernel_sources": [],   # REMOVED v46 — that's the whole point of v50
    "model_sources": [
        "google/gemma-4/Transformers/gemma-4-e2b-it/1",
    ],
    "machine_shape": "NvidiaTeslaT4",
}
meta_path = DST_DIR / "kernel-metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved: {meta_path}")
print()
print("Next steps:")
print("  kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v50-dpo/")
print()
print("Key things to watch:")
print("  1. D8 baseline probe — v42 should REFUSE (88% aggregate security)")
print("  2. Post-training smoke probe — confirm DPO sharpens refusal")
print("  3. V42_ADAPTER path resolution — should pick the v42 dataset path")
print()
print("Expected runtime: ~38-40 min on T4 (150 steps @ ~15s/step)")
