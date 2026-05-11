"""build_nla_smoke_nb.py — Build the NLA Stage 1 smoke-test Kaggle notebook.

Smoke test: 50K passages, 3 epochs, ~3h on T4, free.
Validates the AR SFT pipeline end-to-end before committing $50 RunPod full run.
Gate: final FVE > 0.20 on held-out 5K pairs.

Usage:
    python experiments/build_nla_smoke_nb.py
    # Then from the output dir:
    # kaggle kernels push -p D:/kaggle/notebooks/haic-nla-stage1-smoke/
"""

import json
from pathlib import Path

DST_DIR = Path("D:/kaggle/notebooks/haic-nla-stage1-smoke")
DST     = DST_DIR / "haic_nla_stage1_smoke.ipynb"
DST_DIR.mkdir(parents=True, exist_ok=True)

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source
                  if isinstance(source, list) else [source]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": source if isinstance(source, list) else [source]})

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
md([
    "# NLA Stage 1 Smoke Test — AR SFT on Gemma-4-E2B\n",
    "\n",
    "**Hypothesis H-NLA1 (from docs/nla_stage1_hypothesis_2026-05-11.md):**\n",
    "A warm-start Activation Reconstructor (AR) SFT for Gemma-4-E2B-it achieves\n",
    "FVE ≥ 0.40 after full training (~$50 on RunPod H100).\n",
    "\n",
    "**This notebook:** smoke test only — 50K passages, 3 epochs, ~3h on T4 (free).\n",
    "Gate: FVE > 0.20 → confirms the pipeline works end-to-end before RunPod commit.\n",
    "\n",
    "**Falsifiable predicates:**\n",
    "- H-NLA1a: full-run FVE >= 0.40 (smoke won't hit this — it's a pipeline check)\n",
    "- H-NLA1b: full-run FVE >= 0.30\n",
    "- Smoke gate: FVE > 0.20 (proves activation collection + AR training pipeline is sound)\n",
])

# ── Cell 1: Input listing ─────────────────────────────────────────────────────
md("## Input listing")
code([
    "import os\n",
    "print('=' * 60); print('INPUT FILE LISTING'); print('=' * 60)\n",
    "for d, _, fs in os.walk('/kaggle/input'):\n",
    "    for f in fs:\n",
    "        fp = os.path.join(d, f)\n",
    "        try:\n",
    "            print(f'  {fp}  ({os.path.getsize(fp):,} bytes)')\n",
    "        except OSError:\n",
    "            print(f'  {fp}  (unavailable)')\n",
])

# ── Cell 2: Install deps ──────────────────────────────────────────────────────
md("## Setup")
code([
    "import os\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n",
    "!pip install -q --upgrade transformers datasets accelerate bitsandbytes h5py\n",
    "import torch\n",
    "print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | '\n",
    "      f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')\n",
])

# ── Cell 3: Collect activations ───────────────────────────────────────────────
md([
    "## Step 1: Collect Gemma-4-E2B layer-18 activations\n",
    "\n",
    "Forward-pass 50K TinyStories passages through Gemma-4-E2B-it with a residual-stream\n",
    "hook at layer 18. Save (mean-pooled embedding, layer-18 activation) pairs.\n",
    "~10 min on T4 at batch_size=16.\n",
])
code([
    "import torch, json, math, hashlib, h5py\n",
    "from pathlib import Path\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "from datasets import load_dataset\n",
    "\n",
    "# Locate model (Kaggle model registry)\n",
    "import glob\n",
    "model_candidates = glob.glob('/kaggle/input/**/*.safetensors', recursive=True)\n",
    "MODEL_PATH = '/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1'\n",
    "if not Path(MODEL_PATH).exists():\n",
    "    # fallback: find via glob\n",
    "    for p in model_candidates:\n",
    "        candidate_dir = str(Path(p).parent)\n",
    "        if 'gemma' in candidate_dir.lower():\n",
    "            MODEL_PATH = candidate_dir\n",
    "            break\n",
    "print(f'Model path: {MODEL_PATH}')\n",
    "\n",
    "TARGET_LAYER  = 18\n",
    "D_MODEL       = 1536\n",
    "N_COLLECT     = 50_000\n",
    "N_HOLDOUT     = 5_000\n",
    "BATCH_COLLECT = 16\n",
    "MAX_SEQ_LEN   = 128\n",
    "OUT_PATH      = Path('/kaggle/working/nla_smoke_activations.h5')\n",
    "\n",
    "print(f'Loading Gemma-4-E2B-it ...')\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto'\n",
    ")\n",
    "model.eval()\n",
    "print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')\n",
    "\n",
    "# Hook layer-18 residual stream (last-token position)\n",
    "_acts = {}\n",
    "def _hook(m, inp, out):\n",
    "    _acts['act'] = out[0][:, -1, :].detach().cpu().float()\n",
    "hook = model.model.layers[TARGET_LAYER].register_forward_hook(_hook)\n",
    "\n",
    "print(f'Loading TinyStories corpus (streaming) ...')\n",
    "dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)\n",
    "\n",
    "act_list, emb_list = [], []\n",
    "batch_texts = []\n",
    "n_collected = 0\n",
    "\n",
    "for item in dataset:\n",
    "    text = item.get('text', '')\n",
    "    if len(text.strip()) < 20:\n",
    "        continue\n",
    "    batch_texts.append(text[:512])\n",
    "\n",
    "    if len(batch_texts) < BATCH_COLLECT and n_collected + len(batch_texts) < N_COLLECT:\n",
    "        continue\n",
    "\n",
    "    inputs = tokenizer(\n",
    "        batch_texts, return_tensors='pt', padding=True,\n",
    "        truncation=True, max_length=MAX_SEQ_LEN\n",
    "    ).to('cuda')\n",
    "    with torch.no_grad():\n",
    "        emb = model.model.embed_tokens(inputs['input_ids'])\n",
    "        mask = inputs['attention_mask'].unsqueeze(-1).float()\n",
    "        mean_emb = (emb * mask).sum(1) / mask.sum(1)\n",
    "        _ = model(**inputs)\n",
    "\n",
    "    act_list.append(_acts['act'].clone())\n",
    "    emb_list.append(mean_emb.detach().cpu().float())\n",
    "    n_collected += len(batch_texts)\n",
    "    batch_texts = []\n",
    "\n",
    "    if n_collected % 5_000 == 0:\n",
    "        print(f'  Collected {n_collected:,} / {N_COLLECT:,} pairs ...')\n",
    "    if n_collected >= N_COLLECT:\n",
    "        break\n",
    "\n",
    "hook.remove()\n",
    "del model  # free VRAM for AR training\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "all_acts = torch.cat(act_list)\n",
    "all_embs = torch.cat(emb_list)\n",
    "print(f'Collected {len(all_acts):,} pairs. '\n",
    "      f'Activation shape: {all_acts.shape}, Embedding shape: {all_embs.shape}')\n",
    "\n",
    "# Save to HDF5\n",
    "import numpy as np\n",
    "with h5py.File(OUT_PATH, 'w') as f:\n",
    "    f.create_dataset('activations', data=all_acts.numpy(), compression='gzip')\n",
    "    f.create_dataset('embeddings',  data=all_embs.numpy(), compression='gzip')\n",
    "    f.attrs['n_pairs']   = len(all_acts)\n",
    "    f.attrs['layer_idx'] = TARGET_LAYER\n",
    "    f.attrs['d_model']   = D_MODEL\n",
    "    f.attrs['model_id']  = MODEL_PATH\n",
    "\n",
    "# Stats\n",
    "stats = {\n",
    "    'n_pairs': int(len(all_acts)),\n",
    "    'activation_mean': float(all_acts.mean()),\n",
    "    'activation_std':  float(all_acts.std()),\n",
    "    'activation_norm_mean': float(all_acts.norm(dim=1).mean()),\n",
    "    'embedding_dim': int(all_embs.shape[1]),\n",
    "}\n",
    "print('Activation stats:', json.dumps(stats, indent=2))\n",
    "print(f'Saved to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)')\n",
])

# ── Cell 4: Train AR ──────────────────────────────────────────────────────────
md([
    "## Step 2: Train Activation Reconstructor (AR)\n",
    "\n",
    "4-layer MLP, MSE loss on bfloat16 activations, 3 epochs over 45K pairs,\n",
    "FVE logged every 2K steps on 5K held-out pairs. ~2.5h on T4.\n",
])
code([
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import TensorDataset, DataLoader\n",
    "import time\n",
    "\n",
    "# Load activations\n",
    "with h5py.File(OUT_PATH, 'r') as f:\n",
    "    all_acts = torch.from_numpy(f['activations'][:])\n",
    "    all_embs = torch.from_numpy(f['embeddings'][:])\n",
    "\n",
    "# Normalize activations\n",
    "act_mean = all_acts.mean(0, keepdim=True)\n",
    "act_std  = all_acts.std(0,  keepdim=True).clamp(min=1e-6)\n",
    "acts_norm = (all_acts - act_mean) / act_std\n",
    "\n",
    "holdout_acts = acts_norm[:N_HOLDOUT]\n",
    "holdout_embs = all_embs[:N_HOLDOUT]\n",
    "train_acts   = acts_norm[N_HOLDOUT:]\n",
    "train_embs   = all_embs[N_HOLDOUT:]\n",
    "print(f'Train: {len(train_acts):,} | Holdout: {len(holdout_acts):,} | Emb dim: {all_embs.shape[1]}')\n",
    "\n",
    "# AR model\n",
    "INPUT_DIM  = all_embs.shape[1]\n",
    "HIDDEN_DIM = 2048\n",
    "DEVICE     = 'cuda'\n",
    "\n",
    "class AR(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN_DIM)\n",
    "        self.layers = nn.ModuleList([\n",
    "            nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.05))\n",
    "            for _ in range(3)\n",
    "        ])\n",
    "        self.norms  = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(3)])\n",
    "        self.output_proj = nn.Linear(HIDDEN_DIM, D_MODEL)\n",
    "\n",
    "    def forward(self, x):\n",
    "        h = F.gelu(self.input_proj(x))\n",
    "        for layer, norm in zip(self.layers, self.norms):\n",
    "            h = norm(h + layer(h))\n",
    "        return self.output_proj(h)\n",
    "\n",
    "ar = AR().to(DEVICE)\n",
    "print(f'AR parameters: {sum(p.numel() for p in ar.parameters()):,}')\n",
    "\n",
    "dataset = TensorDataset(train_embs, train_acts)\n",
    "loader  = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)\n",
    "optimizer = optim.AdamW(ar.parameters(), lr=1e-4, weight_decay=1e-4)\n",
    "EPOCHS = 3\n",
    "total_steps = EPOCHS * len(loader)\n",
    "scheduler   = optim.lr_scheduler.OneCycleLR(\n",
    "    optimizer, max_lr=1e-4, total_steps=total_steps, pct_start=0.05\n",
    ")\n",
    "\n",
    "def compute_fve(model, acts, embs, batch=256):\n",
    "    model.eval()\n",
    "    preds = []\n",
    "    with torch.no_grad():\n",
    "        for i in range(0, len(embs), batch):\n",
    "            preds.append(model(embs[i:i+batch].to(DEVICE)).cpu())\n",
    "    pred = torch.cat(preds)\n",
    "    mse  = F.mse_loss(pred, acts[:len(pred)]).item()\n",
    "    var  = acts[:len(pred)].var().item()\n",
    "    return round(1.0 - mse / max(var, 1e-10), 6)\n",
    "\n",
    "fve_curve = []\n",
    "global_step = 0\n",
    "LOG_EVERY = 2_000\n",
    "t0 = time.time()\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    ar.train()\n",
    "    for emb_b, act_b in loader:\n",
    "        emb_b, act_b = emb_b.to(DEVICE), act_b.to(DEVICE)\n",
    "        pred = ar(emb_b)\n",
    "        loss = F.mse_loss(pred, act_b)\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        nn.utils.clip_grad_norm_(ar.parameters(), 1.0)\n",
    "        optimizer.step()\n",
    "        scheduler.step()\n",
    "        global_step += 1\n",
    "        if global_step % LOG_EVERY == 0:\n",
    "            fve = compute_fve(ar, holdout_acts, holdout_embs)\n",
    "            elapsed = int(time.time() - t0)\n",
    "            print(f'step {global_step:5d}  loss={loss.item():.4f}  FVE={fve:.4f}  '\n",
    "                  f'lr={scheduler.get_last_lr()[0]:.2e}  {elapsed}s')\n",
    "            fve_curve.append({'step': global_step, 'fve': fve, 'loss': float(loss.item())})\n",
    "            ar.train()\n",
    "\n",
    "final_fve = compute_fve(ar, holdout_acts, holdout_embs)\n",
    "print(f'\\nFinal FVE (held-out 5K): {final_fve:.4f}')\n",
    "\n",
    "# Save checkpoint\n",
    "ckpt_path = Path('/kaggle/working/nla_ar_smoke_final.pt')\n",
    "torch.save(ar.state_dict(), ckpt_path)\n",
    "print(f'Checkpoint saved: {ckpt_path}')\n",
    "\n",
    "# Save FVE curve\n",
    "fve_path = Path('/kaggle/working/nla_smoke_fve_curve.json')\n",
    "fve_curve.append({'step': global_step, 'fve': final_fve, 'final': True})\n",
    "fve_path.write_text(json.dumps(fve_curve, indent=2))\n",
    "print(f'FVE curve: {fve_path}')\n",
])

# ── Cell 5: Evaluate predicates ───────────────────────────────────────────────
md("## Results — predicate evaluation")
code([
    "# Evaluate against H-NLA1 predicates\n",
    "smoke_gate   = final_fve > 0.20\n",
    "nla1b_target = final_fve >= 0.30   # likely not reached in smoke\n",
    "nla1a_target = final_fve >= 0.40   # definitely not in smoke\n",
    "\n",
    "results = {\n",
    "    'final_fve':    final_fve,\n",
    "    'total_steps':  global_step,\n",
    "    'n_train':      len(train_acts),\n",
    "    'n_holdout':    len(holdout_acts),\n",
    "    'fve_curve':    fve_curve,\n",
    "    'predicates': {\n",
    "        'smoke_gate (fve > 0.20)':   {'passed': smoke_gate,   'actual': final_fve},\n",
    "        'H_NLA1b (fve >= 0.30)':     {'passed': nla1b_target, 'actual': final_fve},\n",
    "        'H_NLA1a (fve >= 0.40)':     {'passed': nla1a_target, 'actual': final_fve},\n",
    "    },\n",
    "    'recommendation': (\n",
    "        'PROCEED_TO_RUNPOD_FULL_RUN'  if smoke_gate   else\n",
    "        'ABORT_NLA_TRACK'\n",
    "    )\n",
    "}\n",
    "\n",
    "results_path = Path('/kaggle/working/nla_smoke_results.json')\n",
    "results_path.write_text(json.dumps(results, indent=2))\n",
    "\n",
    "print('=' * 60)\n",
    "print('NLA Stage 1 SMOKE TEST — RESULTS')\n",
    "print('=' * 60)\n",
    "print(f'  Final FVE (5K held-out):  {final_fve:.4f}')\n",
    "print(f'  Smoke gate (FVE > 0.20):  {\"PASS\" if smoke_gate else \"FAIL\"}')\n",
    "print(f'  H-NLA1b (FVE >= 0.30):    {\"PASS\" if nla1b_target else \"FAIL\"} (full-run target)')\n",
    "print(f'  H-NLA1a (FVE >= 0.40):    {\"PASS\" if nla1a_target else \"FAIL\"} (full-run target)')\n",
    "print()\n",
    "print(f'  Recommendation: {results[\"recommendation\"]}')\n",
    "print()\n",
    "if smoke_gate:\n",
    "    print('  → Pipeline validated. Proceed to RunPod H100 full run (~$50).')\n",
    "    print('    python experiments/nla_stage1_ar_sft.py --mode full')\n",
    "else:\n",
    "    print('  → FVE below 0.20. Pipeline has a problem.')\n",
    "    print('    Check activation collection (stats above) and AR architecture.')\n",
    "    print('    Do NOT proceed to RunPod full run until smoke gate passes.')\n",
    "print()\n",
    "print(f'  Results: /kaggle/working/nla_smoke_results.json')\n",
    "print(f'  FVE curve: /kaggle/working/nla_smoke_fve_curve.json')\n",
    "print(f'  AR checkpoint: /kaggle/working/nla_ar_smoke_final.pt')\n",
])

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Saved: {DST}")

# ── kernel-metadata.json ──────────────────────────────────────────────────────
meta = {
    "id": "benhaslam/haic-nla-stage1-smoke",
    "title": "haic-nla-stage1-smoke",
    "code_file": "haic_nla_stage1_smoke.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": ["gpu"],
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [
        "google/gemma-4/Transformers/gemma-4-e2b-it/1"
    ],
    "machine_shape": "NvidiaTeslaT4",
}
meta_path = DST_DIR / "kernel-metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved: {meta_path}")
print()
print("Next: kaggle kernels push -p D:/kaggle/notebooks/haic-nla-stage1-smoke/")
