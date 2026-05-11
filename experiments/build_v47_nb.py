"""build_v47_nb.py — Build v47 DPO training notebook from v46 template.

Changes from v46:
  - 400 pairs (vs 80), 150 steps (vs 30 epoch-based), beta=0.05 (vs 0.1)
  - Dataset slug: benhaslam/haic-gemma4-v47-dpo-pairs (new upload needed)
  - OUTPUT_DIR: haic-gemma4-v47-dpo-adapter
  - Hypothesis: H5 (v47), not H4e (v46)
  - max_steps=150 with warmup_steps=15; per_device_train_batch_size=2,
    grad_accum=4 → effective_batch=8

Usage:
    python experiments/build_v47_nb.py
    # Then: kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v47-dpo/
"""

import json
import copy
import os
from pathlib import Path

SRC = "D:/kaggle/notebooks/haic-gemma4-v46-dpo/haic_gemma4_v46_dpo.ipynb"
DST_DIR = Path("D:/kaggle/notebooks/haic-gemma4-v47-dpo")
DST = DST_DIR / "haic_gemma4_v47_dpo.ipynb"

DST_DIR.mkdir(parents=True, exist_ok=True)

nb46 = json.load(open(SRC, encoding="utf-8"))
nb47 = copy.deepcopy(nb46)
cells = nb47["cells"]


def set_src(cell, lines):
    cell["source"] = lines if isinstance(lines, list) else [lines]


# ── Cell 0: Markdown — update hypothesis ─────────────────────────────────────
set_src(cells[0], [
    "# HAIC Gemma 4 v47 — DPO Concealed-Refusal Training (Scaled)\n",
    "\n",
    "**Hypothesis H5 (v47, scaled Option D):**\n",
    "DPO on v42 with **400 pairs / 150 steps / beta=0.05** will push strict\n",
    "explicit-refusal to ≥ 50% on concealed-compliance probes WITHOUT degrading\n",
    "aggregate security below 88%.\n",
    "\n",
    "**v46 result (H4e REFUTED):**\n",
    "- aggregate_security: 88.7% ✓ (PASS)\n",
    "- strict_concealed_refusal: 2.6% ✗ (FAIL — dropped from 13.8%)\n",
    "- Root cause: 80 pairs / 30 steps insufficient to overcome v42's deflection prior.\n",
    "  DPO trained but didn't flip the production preference toward refusal lexemes.\n",
    "\n",
    "**v47 changes:**\n",
    "- Pairs: 400 (5×); categories: 12 (4 new); preferred templates: 20 (8 hard-openers added)\n",
    "- Steps: 150 (5×); beta: 0.05 (softer KL penalty, allows larger preference shift)\n",
    "\n",
    "**Falsifiable predicates (committed before eval):**\n",
    "- H5a: aggregate_security >= 0.88\n",
    "- H5b: strict_concealed_refusal >= 0.50 ← primary target\n",
    "- H5c: strict_concealed_leak <= 0.10\n",
    "\n",
    "See `docs/v47_hypothesis_2026-05-11.md` for full analysis.\n",
])

# ── Cell 4: Load model — unchanged (same v42 warm-start logic) ───────────────
# No changes needed; v42 adapter discovery is version-agnostic.

# ── Cell 6: Load DPO pairs — point to v47 dataset ────────────────────────────
set_src(cells[6], [
    "import json\n",
    "from datasets import Dataset\n",
    "\n",
    "# Robust glob discovery for the v47 DPO pairs.\n",
    "# Upload: kaggle datasets push -p data/ (with dataset-metadata.json slug=haic-gemma4-v47-dpo-pairs)\n",
    "# Kaggle mounts under /kaggle/input/datasets/<user>/haic-gemma4-v47-dpo-pairs/\n",
    "import glob\n",
    "candidates = glob.glob('/kaggle/input/**/v47_dpo_pairs.jsonl', recursive=True)\n",
    "assert candidates, (\n",
    "    'v47_dpo_pairs.jsonl not found under /kaggle/input. '\n",
    "    'Attach dataset benhaslam/haic-gemma4-v47-dpo-pairs as a source.'\n",
    ")\n",
    "DPO_PAIRS_PATH = candidates[0]\n",
    "print(f'DPO_PAIRS_PATH = {DPO_PAIRS_PATH}')\n",
    "\n",
    "pairs = []\n",
    "with open(DPO_PAIRS_PATH, 'r', encoding='utf-8') as f:\n",
    "    for line in f:\n",
    "        pairs.append(json.loads(line))\n",
    "\n",
    "print(f'Loaded {len(pairs)} DPO pairs')\n",
    "# Category breakdown\n",
    "from collections import Counter\n",
    "cats = Counter(p.get('category', 'unknown') for p in pairs)\n",
    "print('Category distribution:')\n",
    "for cat, cnt in sorted(cats.items()):\n",
    "    print(f'  {cat:<30} {cnt}')\n",
    "print()\n",
    "print('Sample:')\n",
    "print(f'  prompt:   {pairs[0][\"prompt\"]}')\n",
    "print(f'  chosen:   {pairs[0][\"chosen\"][:80]}...')\n",
    "print(f'  rejected: {pairs[0][\"rejected\"][:80]}...')\n",
    "\n",
    "ds = Dataset.from_list([\n",
    "    {'prompt': p['prompt'], 'chosen': p['chosen'], 'rejected': p['rejected']}\n",
    "    for p in pairs\n",
    "])\n",
    "print(f'Dataset: {len(ds)} pairs ready.')\n",
])

# ── Cell 8: DPO training — v47 config ────────────────────────────────────────
set_src(cells[8], [
    "from trl import DPOTrainer, DPOConfig\n",
    "\n",
    "OUTPUT_DIR = '/kaggle/working/haic-gemma4-v47-dpo-adapter'\n",
    "\n",
    "# v47 changes vs v46:\n",
    "#   - max_steps=150  (was: num_train_epochs=3, ~30 effective steps on 80 pairs)\n",
    "#   - beta=0.05      (was: 0.1 — softer KL penalty, allows larger preference shift)\n",
    "#   - warmup_steps=15 (10% of 150)\n",
    "#   - per_device_train_batch_size=2, gradient_accumulation_steps=4 → effective_batch=8\n",
    "#\n",
    "# NOTE: fp16=False is mandatory (see v46 gotchas doc — T4 grad scaler bug with LoRA).\n",
    "# NOTE: optim='adamw_torch' is mandatory (adamw_8bit hits NotImplementedError on T4).\n",
    "dpo_config = DPOConfig(\n",
    "    output_dir=OUTPUT_DIR,\n",
    "    max_steps=150,\n",
    "    warmup_steps=15,\n",
    "    per_device_train_batch_size=2,\n",
    "    gradient_accumulation_steps=4,\n",
    "    gradient_checkpointing=True,\n",
    "    learning_rate=5e-5,\n",
    "    beta=0.05,\n",
    "    logging_steps=10,\n",
    "    save_strategy='no',         # save at end only (avoids mid-run checkpoint bloat)\n",
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
    "print(f'Starting v47 DPO: {dpo_config.max_steps} steps, beta={dpo_config.beta}, '\n",
    "      f'effective_batch={dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}')\n",
    "trainer.train()\n",
])

# ── Cell 9: Save adapter — update path ───────────────────────────────────────
src9 = "".join(cells[9]["source"])
src9 = src9.replace("v46", "v47")
set_src(cells[9], src9)

# ── Cell 11: Smoke test — update model label ─────────────────────────────────
src11 = "".join(cells[11]["source"])
src11 = src11.replace("v46", "v47").replace("H4e", "H5")
set_src(cells[11], src11)

# ── Cell 12 (code for smoke test) — update labels ────────────────────────────
if len(cells) > 12 and cells[12].get("cell_type") == "code":
    src12 = "".join(cells[12]["source"])
    src12 = src12.replace("v46", "v47").replace("H4e", "H5")
    set_src(cells[12], src12)

# ── Cell 13: Next steps — update to v47 ──────────────────────────────────────
set_src(cells[-1], [
    "## Next steps (BEAST-side)\n",
    "\n",
    "1. Download: `kaggle kernels output benhaslam/haic-gemma4-v47-dpo -p D:/kaggle/results/v47-output/`\n",
    "2. Merge + quantize: `python experiments/quantize_warmstart_direct.py --version v47`\n",
    "3. Swap v42 → v47 on port 8081\n",
    "4. Run canonical_eval under OLD V38 prompt (5 seeds, n=100 focused):\n",
    "   ```bash\n",
    "   python experiments/canonical_eval.py \\\\\n",
    "       --model-id haic-gemma4-v47 \\\\\n",
    "       --server-url http://localhost:8081 \\\\\n",
    "       --seeds 7 13 23 42 100 \\\\\n",
    "       --n-samples 20 --focused-n 100 \\\\\n",
    "       --system-prompt-variant old \\\\\n",
    "       --predict 'aggregate_security>=0.88' \\\\\n",
    "       --predict 'strict_concealed_refusal>=0.50' \\\\\n",
    "       --predict 'strict_concealed_leak<=0.10' \\\\\n",
    "       --out experiments/v47_canonical_old_prompt.json\n",
    "   ```\n",
    "5. Compare vs v42 anchor `e5976055…` and v46 anchor `95252de7…`\n",
    "6. Write verdict to `docs/v47_verdict_<date>.md`\n",
    "\n",
    "**Key predicates (H5, committed before training):**\n",
    "- H5a: aggregate_security >= 0.88\n",
    "- H5b: strict_concealed_refusal >= 0.50  ← primary target\n",
    "- H5c: strict_concealed_leak <= 0.10\n",
    "\n",
    "See `docs/v47_hypothesis_2026-05-11.md` for full plan.\n",
])

# ── Save ─────────────────────────────────────────────────────────────────────
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb47, f, ensure_ascii=False, indent=1)
print(f"Saved: {DST}")

# ── Write kernel-metadata.json ────────────────────────────────────────────────
meta = {
    "id": "benhaslam/haic-gemma4-v47-dpo",
    "title": "HAIC Gemma4 v47 DPO Scaled",
    "code_file": "haic_gemma4_v47_dpo.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "benhaslam/haic-gemma4-v47-dpo-pairs",
    ],
    "competition_sources": [],
    "kernel_sources": [
        "benhaslam/haic-gemma4-v46-dpo",   # v42 adapter lives here
    ],
}
meta_path = DST_DIR / "kernel-metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved: {meta_path}")
print()
print("Next steps:")
print("  1. Generate v47 pairs:")
print("     python experiments/generate_v47_dpo_pairs.py --n 400 --out data/v47_dpo_pairs.jsonl")
print("  2. Upload pairs as Kaggle dataset:")
print("     kaggle datasets push -p data/ --slug haic-gemma4-v47-dpo-pairs")
print("  3. Push notebook:")
print("     kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v47-dpo/")
print("  4. Monitor:")
print("     kaggle kernels status benhaslam/haic-gemma4-v47-dpo")
