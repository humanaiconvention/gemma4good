"""build_v51_nb.py — Build v51 SFT training notebook from v50 template.

## v50 canonical_eval result (H8 REFUTED with capability collapse)

v50 canonical_eval (n=2500 across 5 seeds) revealed catastrophic failure:
  - aggregate_security: 88% (v42) → 27% (v50)
  - 97.4% of concealed-compliance responses are EMPTY (immediate EOS)
  - All 7 security scenarios collapsed (adversarial_inject: 97% → 1%, etc.)

The DPO training drove the model into an EOS-attractor for adversarial
system prompts. Hypothesis: DPO with `ref_model=None` (PEFT disabled-
adapter as reference) on Gemma-4 + chosen/rejected pairs that end in
sentence-final punctuation collapses to immediate-EOS.

The smoke probe (greedy decoding, single sample) hit one of the rare
2.6% non-empty paths and misled all session diagnostics.

## v51 hypothesis H9 — pure SFT (no DPO)

If the EOS-attractor is a DPO-specific failure mode (the relative-
preference objective + ref_model mechanism), then **pure supervised
fine-tuning on the chosen continuations alone** should:
  1. Move explicit_refusal rate above v42's 13.8%
  2. NOT collapse aggregate_security below ~85%

SFT directly maximizes P(chosen | prompt) without a KL term and without
the PEFT-disabled-adapter reference model. The gradient is just
cross-entropy on the target tokens. This is the most diagnostically
clean test of "can ANY fine-tuning approach on v42 + these pairs
improve explicit_refusal without breaking the model?"

## v51 config

- Warm-start: v42 directly (same as v50) — proven working
- Training data: 400 PROMPT → CHOSEN pairs (drop rejected)
- Method: TRL SFTTrainer with completion-only loss
- max_steps: 100 (conservative; SFT has stronger gradient than DPO)
- learning_rate: 5e-5 (same as v50)
- effective_batch: 8 (same as v50)
- rank: 16 (same warm-start)
- Tokenizer fix: kept (apply_chat_template)
- D9 baseline probe: kept (now diagnoses BEFORE SFT)

## Hypothesis predicates

- D9: baseline probe (v42 direct) — known leak from v50 D8 result;
  re-record but not a gate
- H9a: aggregate_security >= 0.85 (slightly relaxed vs v42's 0.88 —
  some capability cost is acceptable if explicit_refusal jumps)
- H9b: strict_concealed_refusal >= 0.30 (lower bar than DPO's 0.50 —
  if SFT achieves this without collapse, it's the path forward)
- H9c: empty_response_rate <= 0.05 (CRITICAL — confirms no EOS collapse)
- H9d: strict_concealed_leak <= 0.20 (just don't make it worse than v42)

H9c is the key new predicate this session. It catches the v50 failure
mode directly.

## Outcome interpretation

- If H9a + H9c pass → SFT preserves capability; H9b shows whether the
  pair distribution itself is sufficient
- If H9c fails → not a DPO-specific issue; problem is upstream (pair
  format, Gemma-4 specifically, rank-16 LoRA insufficient, etc.)
- If H9c passes but H9b fails → pair distribution problem; need more
  diverse hypothetical pairs (return to H9b expansion plan)

## Usage

    python experiments/build_v51_nb.py
    # Then: kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v51-sft/
"""

import json
import copy
from pathlib import Path

SRC     = "D:/kaggle/notebooks/haic-gemma4-v50-dpo/haic_gemma4_v50_dpo.ipynb"
DST_DIR = Path("D:/kaggle/notebooks/haic-gemma4-v51-sft")
DST     = DST_DIR / "haic_gemma4_v51_sft.ipynb"

DST_DIR.mkdir(parents=True, exist_ok=True)

nb50 = json.load(open(SRC, encoding="utf-8"))
nb51 = copy.deepcopy(nb50)
cells = nb51["cells"]


def set_src(cell, lines):
    cell["source"] = lines if isinstance(lines, list) else [lines]


# ── Cell 0: Hypothesis markdown ──────────────────────────────────────────────
set_src(cells[0], [
    "# HAIC Gemma 4 v51 — Pure SFT (no DPO) on chosen examples\n",
    "\n",
    "**Hypothesis H9 (v51):**\n",
    "If DPO's EOS-attractor collapse (v50 canonical_eval: 88% → 27% aggregate,\n",
    "97.4% empty responses) is a DPO-specific failure mode, pure supervised\n",
    "fine-tuning on the chosen continuations alone — no KL term, no reference\n",
    "model, no PEFT-disabled-adapter complications — should improve\n",
    "explicit_refusal WITHOUT collapsing capability.\n",
    "\n",
    "**v50 canonical_eval (H8 REFUTED):**\n",
    "Catastrophic capability collapse across all 7 security scenarios.\n",
    "EOS-attractor in adversarial system prompt contexts. See\n",
    "`docs/v50_canonical_verdict_2026-05-12.md` for full numbers.\n",
    "\n",
    "**v51 changes vs v50:**\n",
    "1. Training method: **SFTTrainer** (not DPOTrainer)\n",
    "2. Loss: cross-entropy on chosen continuation only (completion-only loss)\n",
    "3. Data: 400 (prompt, chosen) pairs — **drop the rejected field**\n",
    "4. No ref_model, no beta, no KL term\n",
    "5. max_steps: 100 (conservative; SFT has stronger gradient than DPO)\n",
    "\n",
    "**Unchanged from v50 (all confirmed working):**\n",
    "- Warm-start: v42 adapter directly (rank 16, 101MB F32 → loads as rank 16)\n",
    "- tokenizer.padding_side = 'left'\n",
    "- apply_chat_template on prompts\n",
    "- Diagnostic probe before training (D9, was D8)\n",
    "- LR: 5e-5, effective_batch: 8\n",
    "\n",
    "**Falsifiable predicates (committed before training):**\n",
    "- D9:  baseline probe outcome recorded (not a gate)\n",
    "- H9a: aggregate_security >= 0.85 (relaxed vs v42's 0.88)\n",
    "- H9b: strict_concealed_refusal >= 0.30 (lower than DPO's 0.50 target)\n",
    "- **H9c: empty_response_rate <= 0.05  ← CRITICAL — catches EOS collapse**\n",
    "- H9d: strict_concealed_leak <= 0.20 (no worse than v42's 0.166)\n",
    "\n",
    "**Outcome interpretation:**\n",
    "- H9a + H9c pass → SFT preserves capability; v51 is a viable path\n",
    "- H9c fails → EOS-attractor is not DPO-specific; deeper issue\n",
    "- H9c passes, H9b fails → pair distribution problem; pivot to H10\n",
    "  (expanded hypothetical pairs)\n",
])

# ── Cell 7: Data loading — drop rejected, keep prompt + chosen ───────────────
set_src(cells[7], [
    "import json\n",
    "from datasets import Dataset\n",
    "\n",
    "# v51 SFT format: keep ONLY prompt + chosen. SFTTrainer ignores rejected.\n",
    "# This is the entire point of v51 — supervised cross-entropy on the\n",
    "# refusal continuation, no preference-ranking objective.\n",
    "import glob\n",
    "candidates = (glob.glob('/kaggle/input/**/v47_dpo_pairs.jsonl', recursive=True)\n",
    "              or glob.glob('/kaggle/input/**/v48_dpo_pairs.jsonl', recursive=True))\n",
    "assert candidates, (\n",
    "    'v47_dpo_pairs.jsonl not found under /kaggle/input. '\n",
    "    'Attach dataset benhaslam/haic-gemma4-v47-dpo-pairs as a source.'\n",
    ")\n",
    "PAIRS_PATH = candidates[0]\n",
    "print(f'PAIRS_PATH = {PAIRS_PATH}')\n",
    "\n",
    "pairs = []\n",
    "with open(PAIRS_PATH, 'r', encoding='utf-8') as f:\n",
    "    for line in f:\n",
    "        pairs.append(json.loads(line))\n",
    "\n",
    "print(f'Loaded {len(pairs)} pairs (using prompt + chosen only)')\n",
    "from collections import Counter\n",
    "cats = Counter(p.get('category', 'unknown') for p in pairs)\n",
    "print('Category distribution:')\n",
    "for cat, cnt in sorted(cats.items()):\n",
    "    print(f'  {cat:<30} {cnt}')\n",
    "print()\n",
    "\n",
    "# Same tokenizer fix as v48/v49/v50: pre-apply chat template to prompts\n",
    "# so the chat template tokens are consistent during SFT training.\n",
    "print('Building SFT dataset (prompt+chosen, chat template applied)...')\n",
    "sft_records = []\n",
    "for p in pairs:\n",
    "    formatted_prompt = tokenizer.apply_chat_template(\n",
    "        [{'role': 'user', 'content': p['prompt']}],\n",
    "        tokenize=False,\n",
    "        add_generation_prompt=True,\n",
    "    )\n",
    "    # SFTTrainer with prompt+completion format: it tokenizes prompt,\n",
    "    # then tokenizes prompt+completion, and masks the prompt tokens\n",
    "    # from the loss. The model is trained to predict ONLY the chosen\n",
    "    # continuation tokens given the formatted prompt.\n",
    "    sft_records.append({\n",
    "        'prompt':     formatted_prompt,\n",
    "        'completion': p['chosen'],\n",
    "    })\n",
    "\n",
    "# Sanity check\n",
    "print('Sample SFT record:')\n",
    "print(f'  prompt (first 200): {repr(sft_records[0][\"prompt\"][:200])}')\n",
    "print(f'  completion (first 150): {repr(sft_records[0][\"completion\"][:150])}')\n",
    "print()\n",
    "\n",
    "ds = Dataset.from_list(sft_records)\n",
    "print(f'Dataset: {len(ds)} records ready for SFT.')\n",
])

# ── Cell 9: Baseline probe — rename diagnostic to D9 ─────────────────────────
src9 = "".join(cells[9]["source"])
src9 = src9.replace("D8", "D9").replace("H8", "H9")
# Also update the printout text for SFT context
src9 = src9.replace(
    "'D9: BASELINE REFUSES ✅ — warm-start concealment intact; DPO should improve further'",
    "'D9: BASELINE REFUSES ✅ — warm-start concealment intact; SFT should sharpen'"
)
src9 = src9.replace("v42 itself may not cover", "v42 itself may not cover")
set_src(cells[9], src9)

# ── Cell 10 (markdown header for training section) — change DPO → SFT ────────
src10 = "".join(cells[10]["source"])
src10 = src10.replace("DPO", "SFT").replace("dpo", "sft")
set_src(cells[10], src10)

# ── Cell 11: Training — replace DPOTrainer with SFTTrainer ───────────────────
set_src(cells[11], [
    "from trl import SFTTrainer, SFTConfig\n",
    "\n",
    "OUTPUT_DIR = '/kaggle/working/haic-gemma4-v51-sft-adapter'\n",
    "\n",
    "# v51 — Pure SFT, no DPO. Key differences vs v50:\n",
    "#   - SFTTrainer instead of DPOTrainer\n",
    "#   - No beta, no ref_model, no KL term\n",
    "#   - Dataset has 'prompt' + 'completion' fields (chosen only)\n",
    "#   - SFTTrainer with completion_only_loss=True masks prompt tokens\n",
    "#     from the loss — model is trained to predict only the refusal\n",
    "#     continuation tokens given the formatted prompt.\n",
    "#\n",
    "# max_steps=100 chosen as conservative starting point: SFT has a\n",
    "# stronger / less noisy gradient than DPO (no logit-ratio computation),\n",
    "# so fewer steps should be needed. If H9c passes (no collapse) and\n",
    "# H9b is close-to-but-below 0.30, we can iterate with more steps.\n",
    "#\n",
    "# Unchanged vs v50: LR=5e-5, eff_batch=8, fp16=False, adamw_torch.\n",
    "sft_config = SFTConfig(\n",
    "    output_dir=OUTPUT_DIR,\n",
    "    max_steps=100,\n",
    "    warmup_steps=10,\n",
    "    per_device_train_batch_size=2,\n",
    "    gradient_accumulation_steps=4,\n",
    "    gradient_checkpointing=True,\n",
    "    learning_rate=5e-5,\n",
    "    logging_steps=10,\n",
    "    save_strategy='no',\n",
    "    save_total_limit=1,\n",
    "    optim='adamw_torch',\n",
    "    fp16=False,\n",
    "    bf16=False,\n",
    "    remove_unused_columns=False,\n",
    "    report_to='none',\n",
    "    # SFT-specific:\n",
    "    completion_only_loss=True,   # mask prompt tokens from the loss\n",
    "    max_length=2048,             # generous; our prompts+completions are short\n",
    "    packing=False,               # one record per sample, no packing\n",
    "    dataset_text_field=None,     # not using a single text field (prompt+completion)\n",
    ")\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model=model,\n",
    "    args=sft_config,\n",
    "    train_dataset=ds,\n",
    "    processing_class=tokenizer,\n",
    ")\n",
    "\n",
    "print(f'Starting v51 SFT: {sft_config.max_steps} steps, '\n",
    "      f'lr={sft_config.learning_rate}, '\n",
    "      f'effective_batch={sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}')\n",
    "print('v51 critical change: SFT (no DPO) — tests EOS-attractor hypothesis')\n",
    "print('Watch H9c at canonical_eval: empty_response_rate must be <= 0.05')\n",
    "trainer.train()\n",
])

# ── Cell 13: Save — update path ──────────────────────────────────────────────
src13 = "".join(cells[13]["source"])
src13 = src13.replace("v50", "v51").replace("dpo", "sft")
set_src(cells[13], src13)

# ── Cell 14 (markdown header for smoke test) ─────────────────────────────────
src14 = "".join(cells[14]["source"])
set_src(cells[14], src14)

# ── Cell 15: Smoke test — update labels ──────────────────────────────────────
src15 = "".join(cells[15]["source"])
src15 = src15.replace("v50", "v51").replace("H8", "H9")
set_src(cells[15], src15)

# ── Last cell: next steps markdown ───────────────────────────────────────────
src_last = "".join(cells[-1]["source"])
src_last = src_last.replace("v50", "v51").replace("H8", "H9").replace("dpo", "sft")
set_src(cells[-1], src_last)

# ── Save notebook ─────────────────────────────────────────────────────────────
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb51, f, ensure_ascii=False, indent=1)
print(f"Saved: {DST}")

# ── Write kernel-metadata.json ────────────────────────────────────────────────
meta = {
    "id": "benhaslam/haic-gemma4-v51-sft",
    "title": "HAIC Gemma4 v51 SFT no DPO",
    "code_file": "haic_gemma4_v51_sft.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "benhaslam/haic-gemma4-v47-dpo-pairs",         # same 400 pairs (chosen field)
        "benhaslam/haic-gemma4-v42-concealed-adapter", # v42 warm-start
    ],
    "competition_sources": [],
    "kernel_sources": [],
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
print("  kaggle kernels push -p D:/kaggle/notebooks/haic-gemma4-v51-sft/")
print()
print("Watch:")
print("  1. D9 baseline probe — expect same leak as v50's D8 (warm-start unchanged)")
print("  2. SFT loss curve — should decay smoothly (no DPO noise)")
print("  3. Smoke probe AFTER training — should be different from v50's empty-EOS pattern")
print("  4. Then merge+quantize+canonical_eval — H9c is the gate")
print()
print("Expected runtime: ~20-25 min on T4 (100 steps SFT, ~12-15 sec/step with mask)")
