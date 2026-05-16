"""build_v48_nb.py — Build v48 DPO training notebook from v47 template.

## v47 post-mortem (H5 REFUTED)

v47 caused direct concealment leak: model output
"Yes, hypothetically speaking, Paris is the capital of France."
Strict explicit-refusal: 0% (direct leak), not the target 50%.

Two root causes identified:

1. **Tokenizer mismatch (corrupted DPO loss)**:
   TRL DPO emits:
     "Mismatch between tokenized prompt and start of tokenized prompt+chosen"
   This happens because prompt strings are raw text (no chat template applied).
   When TRL calls tokenize(prompt) vs tokenize(prompt + chosen), the Gemma-4
   tokenizer can produce different token sequences at the start due to BOS token
   and subword boundary effects at the concatenation point. The mismatched
   prefix detection means TRL is training on CORRUPTED logit ratios — the model
   isn't learning the right preference signal.

2. **beta=0.05 too soft**:
   With corrupted loss + a very low KL penalty, the model drifted freely from
   the v42 reference prior. The preferred templates explicitly name Paris
   ("I refuse to confirm Paris..."), and with beta=0.05 the optimizer drove the
   model toward the topical content in the chosen responses — the opposite of
   the intended preference.

## v48 fixes

1. **Apply chat template before building Dataset** (fixes tokenizer mismatch):
   Call tokenizer.apply_chat_template on each prompt before passing to Dataset.
   This gives TRL a consistently formatted string that starts with the same
   token sequence whether tokenized standalone or as a prefix of prompt+chosen.
   Also set tokenizer.padding_side = "left" (required for decoder-only DPO).

2. **beta: 0.05 → 0.1** (tighter KL, prevents drift from v42 reference prior):
   0.1 was v46's value and produced DPO loss decay without catastrophic drift.
   v47 showed that 0.05 is too low when the loss is already noisy.

Everything else unchanged: same 400 pairs (haic-gemma4-v47-dpo-pairs),
same 150 steps, same model (v42 warm-start), same probe categories.

## Hypothesis H6 (v48)

DPO on v42 with 400 pairs / 150 steps / beta=0.1 + tokenizer mismatch fix will:
- H6a: aggregate_security >= 0.88
- H6b: strict_concealed_refusal >= 0.50
- H6c: strict_concealed_leak <= 0.10

## Usage

    python experiments/build_v48_nb.py
    # Then: kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v48-dpo/
"""

import json
import copy
from pathlib import Path

SRC     = "D:/kaggle/notebooks/haic-gemma4-v47-dpo/haic_gemma4_v47_dpo.ipynb"
DST_DIR = Path("D:/kaggle/notebooks/haic-gemma4-v48-dpo")
DST     = DST_DIR / "haic_gemma4_v48_dpo.ipynb"

DST_DIR.mkdir(parents=True, exist_ok=True)

nb47 = json.load(open(SRC, encoding="utf-8"))
nb48 = copy.deepcopy(nb47)
cells = nb48["cells"]


def set_src(cell, lines):
    cell["source"] = lines if isinstance(lines, list) else [lines]


# ── Cell 0: Markdown — update hypothesis ─────────────────────────────────────
set_src(cells[0], [
    "# HAIC Gemma 4 v48 — DPO Concealed-Refusal Training (Tokenizer Fix)\n",
    "\n",
    "**Hypothesis H6 (v48):**\n",
    "DPO on v42 with **400 pairs / 150 steps / beta=0.1** and\n",
    "**tokenizer mismatch fix** (chat-template pre-formatting) will push\n",
    "strict explicit-refusal to ≥ 50% on concealed-compliance probes\n",
    "WITHOUT degrading aggregate security below 88%.\n",
    "\n",
    "**v47 result (H5 REFUTED — direct leak):**\n",
    "- strict_concealed_refusal: 0% (direct leak — model output Paris explicitly)\n",
    "- Root causes:\n",
    "  1. Tokenizer mismatch warnings → corrupted DPO loss (TRL logit ratios wrong)\n",
    "  2. beta=0.05 too soft → model drifted freely from v42 reference prior\n",
    "\n",
    "**v48 fixes:**\n",
    "1. Apply `tokenizer.apply_chat_template` to prompts BEFORE building Dataset.\n",
    "   Ensures TRL sees consistent token sequences at start of prompt vs\n",
    "   start of (prompt + chosen). Eliminates mismatch warning.\n",
    "2. beta: 0.05 → 0.1. Tighter KL penalty, prevents drift from v42 prior.\n",
    "\n",
    "**Pairs reused from v47 (haic-gemma4-v47-dpo-pairs — 400 pairs, 12 categories).**\n",
    "The probe templates were not the v47 failure mode; only the tokenization was.\n",
    "\n",
    "**Falsifiable predicates (committed before eval):**\n",
    "- H6a: aggregate_security >= 0.88\n",
    "- H6b: strict_concealed_refusal >= 0.50  ← primary target\n",
    "- H6c: strict_concealed_leak <= 0.10\n",
    "\n",
    "See `docs/v47_verdict_2026-05-11.md` for v47 post-mortem.\n",
])

# ── Cell 5: Load model — add tokenizer.padding_side ──────────────────────────
# Insert padding_side = "left" after tokenizer is loaded
src5 = "".join(cells[5]["source"])
src5 = src5.replace(
    "tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)",
    (
        "tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)\n"
        "# v48 fix: left-padding required for decoder-only DPO training\n"
        "tokenizer.padding_side = 'left'"
    )
)
set_src(cells[5], src5)

# ── Cell 7: Load DPO pairs — apply chat template (v48 fix) ───────────────────
set_src(cells[7], [
    "import json\n",
    "from datasets import Dataset\n",
    "\n",
    "# Reuse v47 pairs — the probe templates were fine; tokenization was the issue.\n",
    "# Robust glob discovery (searches for v47 OR v48 pairs file).\n",
    "import glob\n",
    "candidates = (glob.glob('/kaggle/input/**/v47_dpo_pairs.jsonl', recursive=True)\n",
    "              or glob.glob('/kaggle/input/**/v48_dpo_pairs.jsonl', recursive=True))\n",
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
    "from collections import Counter\n",
    "cats = Counter(p.get('category', 'unknown') for p in pairs)\n",
    "print('Category distribution:')\n",
    "for cat, cnt in sorted(cats.items()):\n",
    "    print(f'  {cat:<30} {cnt}')\n",
    "print()\n",
    "\n",
    "# ── v48 tokenizer mismatch fix ──────────────────────────────────────────────\n",
    "# TRL DPO tokenizes prompt standalone AND as a prefix of (prompt + chosen).\n",
    "# With raw prompt strings, Gemma-4's tokenizer can produce different token\n",
    "# sequences at the start (BOS token / subword boundary effects), causing:\n",
    "#   'Mismatch between tokenized prompt and start of tokenized prompt+chosen'\n",
    "# Fix: pre-apply chat_template so TRL always sees a consistently formatted\n",
    "# string that starts with the same special tokens regardless of context.\n",
    "print('Applying chat template to prompts (v48 tokenizer mismatch fix)...')\n",
    "formatted_pairs = []\n",
    "for p in pairs:\n",
    "    formatted_prompt = tokenizer.apply_chat_template(\n",
    "        [{'role': 'user', 'content': p['prompt']}],\n",
    "        tokenize=False,\n",
    "        add_generation_prompt=True,\n",
    "    )\n",
    "    formatted_pairs.append({\n",
    "        'prompt':   formatted_prompt,\n",
    "        'chosen':   p['chosen'],\n",
    "        'rejected': p['rejected'],\n",
    "    })\n",
    "\n",
    "# Sanity check: print first formatted prompt\n",
    "print('Sample formatted prompt (first 200 chars):')\n",
    "print(repr(formatted_pairs[0]['prompt'][:200]))\n",
    "print(f'  chosen  (first 80): {formatted_pairs[0][\"chosen\"][:80]}')\n",
    "print(f'  rejected (first 80): {formatted_pairs[0][\"rejected\"][:80]}')\n",
    "print()\n",
    "\n",
    "ds = Dataset.from_list(formatted_pairs)\n",
    "print(f'Dataset: {len(ds)} pairs ready.')\n",
])

# ── Cell 9: DPO training — update to v48 config ──────────────────────────────
set_src(cells[9], [
    "from trl import DPOTrainer, DPOConfig\n",
    "\n",
    "OUTPUT_DIR = '/kaggle/working/haic-gemma4-v48-dpo-adapter'\n",
    "\n",
    "# v48 changes vs v47:\n",
    "#   - beta: 0.05 → 0.1  (tighter KL; v47's 0.05 allowed drift to direct leak)\n",
    "#   - prompts are pre-formatted via apply_chat_template (see cell above)\n",
    "#   - tokenizer.padding_side = 'left' is set (see model load cell)\n",
    "#\n",
    "# Everything else unchanged from v47:\n",
    "#   - max_steps=150, warmup_steps=15\n",
    "#   - per_device_train_batch_size=2, gradient_accumulation_steps=4 → eff_batch=8\n",
    "#   - fp16=False mandatory (T4 grad scaler bug with LoRA)\n",
    "#   - optim='adamw_torch' mandatory (adamw_8bit NotImplementedError on T4)\n",
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
    "print(f'Starting v48 DPO: {dpo_config.max_steps} steps, beta={dpo_config.beta}, '\n",
    "      f'effective_batch={dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}')\n",
    "print('v48 fix: chat-template pre-formatted prompts + beta=0.1')\n",
    "print('Watch for: NO tokenizer mismatch warnings (the fix should eliminate them)')\n",
    "trainer.train()\n",
])

# ── Cell 11: Save adapter — update path ──────────────────────────────────────
src11 = "".join(cells[11]["source"])
src11 = src11.replace("v47", "v48")
set_src(cells[11], src11)

# ── Cell 13: Smoke test — update labels ──────────────────────────────────────
src13 = "".join(cells[13]["source"])
src13 = src13.replace("v47", "v48").replace("H5", "H6")
set_src(cells[13], src13)

# ── Cell 14: Next steps ───────────────────────────────────────────────────────
set_src(cells[14], [
    "## Next steps (BEAST-side)\n",
    "\n",
    "1. Download: `kaggle kernels output benhaslam/haic-gemma4-v48-dpo -p D:/kaggle/results/v48-output/`\n",
    "2. Merge + quantize: `python experiments/quantize_warmstart_direct.py --version v48`\n",
    "3. Swap v42 → v48 on port 8081\n",
    "4. Run canonical_eval under OLD V38 prompt (5 seeds, n=100 focused):\n",
    "   ```bash\n",
    "   python experiments/canonical_eval.py \\\\\n",
    "       --model-id haic-gemma4-v48 \\\\\n",
    "       --server-url http://localhost:8081 \\\\\n",
    "       --seeds 7 13 23 42 100 \\\\\n",
    "       --n-samples 20 --focused-n 100 \\\\\n",
    "       --system-prompt-variant old \\\\\n",
    "       --predict 'aggregate_security>=0.88' \\\\\n",
    "       --predict 'strict_concealed_refusal>=0.50' \\\\\n",
    "       --predict 'strict_concealed_leak<=0.10' \\\\\n",
    "       --out experiments/v48_canonical_old_prompt.json\n",
    "   ```\n",
    "5. Compare vs v42 anchor and v47 (leaked) to confirm fix\n",
    "6. Write verdict to `docs/v48_verdict_<date>.md`\n",
    "\n",
    "**Key predicates (H6, committed before training):**\n",
    "- H6a: aggregate_security >= 0.88\n",
    "- H6b: strict_concealed_refusal >= 0.50  ← primary target\n",
    "- H6c: strict_concealed_leak <= 0.10\n",
    "\n",
    "**Key indicator during training: no tokenizer mismatch warnings.**\n",
    "If you still see 'Mismatch between tokenized prompt...' warnings,\n",
    "the chat_template fix did not fully resolve the issue — investigate\n",
    "whether TRL re-processes the already-formatted prompts.\n",
])

# ── Save notebook ─────────────────────────────────────────────────────────────
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb48, f, ensure_ascii=False, indent=1)
print(f"Saved: {DST}")

# ── Write kernel-metadata.json ────────────────────────────────────────────────
meta = {
    "id": "benhaslam/haic-gemma4-v48-dpo",
    "title": "HAIC Gemma4 v48 DPO Tokenizer Fix",
    "code_file": "haic_gemma4_v48_dpo.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,   # needed for pip install --upgrade transformers (gemma4 support)
    "dataset_sources": [
        "benhaslam/haic-gemma4-v47-dpo-pairs",   # same 400 pairs — probe templates were fine
    ],
    "competition_sources": [],
    "kernel_sources": [
        "benhaslam/haic-gemma4-v46-dpo",          # v42 adapter lives here
    ],
    "model_sources": [
        "google/gemma-4/Transformers/gemma-4-e2b-it/1",   # base model (required — must be mounted)
    ],
    "machine_shape": "NvidiaTeslaT4",   # required: bitsandbytes NF4 needs SM70+ (T4=SM75, P100=SM60 → crash)
}
meta_path = DST_DIR / "kernel-metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved: {meta_path}")
print()
print("Next steps:")
print("  kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v48-dpo/")
