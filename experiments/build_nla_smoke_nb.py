"""build_nla_smoke_nb.py — Build the NLA Stage 1 smoke-test Kaggle notebook.

Smoke test: 50K passages, 3 epochs, ~3h on T4, free.
Validates the AR SFT pipeline end-to-end before committing $50 RunPod full run.
Gate: final FVE > 0.20 on held-out 5K pairs.

v5 changes (on top of v4) — ROOT CAUSE FIX:
  - AR input now uses LAST-TOKEN position instead of mean-pool:
      h_in[:, -1, :]   (was: h_in.mean(dim=1))
    Root cause of v4 FVE=0.163 plateau: mean-pooled input discards the
    positional signal that transformer attention actually operates on.
    The AR cannot reconstruct last-token output from a mean-pool because
    attention is global; the crucial per-position context is destroyed.
    Fix: use the same last-token position for both hook captures so the
    AR approximates the local (per-position) transform of the layer.
    Expected FVE improvement: 0.16 → 0.35–0.50.
  - LOG_EVERY: 2000 → 200. In v4, only one FVE log point was captured
    before the plateau. Need to see the learning curve during training,
    not just its endpoint.

v4 changes (on top of v3):
  - Fast-path layer lookup: model.model.language_model.layers[idx]
    (confirmed by v3 run: Gemma4TextDecoderLayer, 35 layers, score=410;
     also matches v47 DPO adapter_config target_modules regex).
    Scanner retained as fallback for other architectures.
  - logits_to_keep=0 on forward pass: skips lm_head (256K-vocab × seq_len
    tensor ≈ 1 GB bfloat16 on T4), which caused OOM at 10K pairs in v3.
  - BATCH_COLLECT 16 → 8 for additional safety margin.
  - torch.cuda.empty_cache() after each batch to prevent fragmentation.

v3 changes (on top of v2):
  - Layer discovery via named_modules() ModuleList scanner.
  - Dual hook (layer input + output), no embed_tokens needed.
  - Diagnosed Gemma4Model children: vision_tower(167M), language_model(4629M),
    audio_tower(305M), embed_vision, embed_audio.

v2 changes:
  - Dynamic layer-path discovery for Gemma4ForConditionalGeneration
    (multimodal model; layers are NOT at model.model.layers)
  - Dual hook captures both layer INPUT and OUTPUT so we need no
    separate embed_tokens call (eliminates that path-discovery problem too)
  - AR: maps residual-stream BEFORE layer-N → residual-stream AFTER layer-N
    (a principled task; the layer IS that mapping)
  - dtype=torch.bfloat16 (deprecated torch_dtype removed)
  - Cell id fields added (nbformat 5.x compliance)

Usage:
    python experiments/build_nla_smoke_nb.py
    # Then from the output dir:
    # kaggle kernels push -p D:/kaggle/notebooks/haic-nla-stage1-smoke/
"""

import json, uuid
from pathlib import Path

DST_DIR = Path("D:/kaggle/notebooks/haic-nla-stage1-smoke")
DST     = DST_DIR / "haic_nla_stage1_smoke.ipynb"
DST_DIR.mkdir(parents=True, exist_ok=True)

cells = []

def _cid():
    return uuid.uuid4().hex[:8]

def md(source):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    })

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
md([
    "# NLA Stage 1 Smoke Test — AR SFT on Gemma-4-E2B (v5 — last-token fix)\n",
    "\n",
    "**Hypothesis H-NLA1 (from docs/nla_stage1_hypothesis_2026-05-11.md):**\n",
    "A warm-start Activation Reconstructor (AR) SFT for Gemma-4-E2B-it achieves\n",
    "FVE ≥ 0.40 after full training (~$50 on RunPod H100).\n",
    "\n",
    "**This notebook:** smoke test only — 50K passages, 3 epochs, ~3h on T4 (free).\n",
    "Gate: FVE > 0.20 → confirms the pipeline works end-to-end before RunPod commit.\n",
    "\n",
    "**v5 root-cause fix:** v4 achieved FVE=0.163 and plateaued. Root cause:\n",
    "mean-pooled AR input discards the positional signal that attention operates on.\n",
    "Fix: use the **last-token position** for *both* the AR input and AR target.\n",
    "The AR now approximates the local per-position transform of layer 18.\n",
    "Expected improvement: FVE 0.16 → 0.35–0.50.\n",
    "\n",
    "**Task:** AR maps residual-stream BEFORE layer-N (last token) →\n",
    "residual-stream AFTER layer-N (last token).\n",
    "Dual hook captures both sides in one forward pass — no embed_tokens path needed.\n",
    "\n",
    "**Falsifiable predicates:**\n",
    "- Smoke gate: FVE > 0.20 (proves activation collection + AR training pipeline is sound)\n",
    "- H-NLA1b: full-run FVE >= 0.30 (post-RunPod)\n",
    "- H-NLA1a: full-run FVE >= 0.40 (post-RunPod)\n",
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

# ── Cell 3: Load model + discover architecture ────────────────────────────────
md([
    "## Step 0: Load model + discover layer path\n",
    "\n",
    "Gemma-4-E2B-it is `Gemma4ForConditionalGeneration` (multimodal, 5.1B params).\n",
    "Confirmed layer path: `model.model.language_model.layers[18]`\n",
    "(Gemma4TextDecoderLayer, 35 layers, score=410 in v3 named_modules() scan;\n",
    "also confirmed by v47 DPO adapter_config target_modules regex).\n",
    "Fast path used with ModuleList scanner as fallback.\n",
])
code([
    "import torch, json, math, hashlib, h5py\n",
    "from pathlib import Path\n",
    "import torch.nn as nn\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "from datasets import load_dataset\n",
    "\n",
    "# Locate model (Kaggle model registry)\n",
    "import glob\n",
    "model_candidates = glob.glob('/kaggle/input/**/*.safetensors', recursive=True)\n",
    "MODEL_PATH = '/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1'\n",
    "if not Path(MODEL_PATH).exists():\n",
    "    for p in model_candidates:\n",
    "        candidate_dir = str(Path(p).parent)\n",
    "        if 'gemma' in candidate_dir.lower():\n",
    "            MODEL_PATH = candidate_dir\n",
    "            break\n",
    "print(f'Model path: {MODEL_PATH}')\n",
    "\n",
    "TARGET_LAYER  = 18\n",
    "D_MODEL       = 1536   # will be overwritten from actual activation shape\n",
    "N_COLLECT     = 50_000\n",
    "N_HOLDOUT     = 5_000\n",
    "BATCH_COLLECT = 8      # 16 → OOM (lm_head 256K-vocab logit tensor ~1 GB on T4)\n",
    "MAX_SEQ_LEN   = 128\n",
    "OUT_PATH      = Path('/kaggle/working/nla_smoke_activations.h5')\n",
    "\n",
    "print('Loading Gemma-4-E2B-it ...')\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_PATH, dtype=torch.bfloat16, device_map='auto'\n",
    ")\n",
    "model.eval()\n",
    "print(f'Model class: {type(model).__name__}')\n",
    "print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')\n",
    "\n",
    "# ── Print direct children of model and model.model for diagnostics ─────────\n",
    "print('\\nDirect children of model:')\n",
    "for name, child in model.named_children():\n",
    "    print(f'  model.{name}: {type(child).__name__}')\n",
    "\n",
    "print('\\nDirect children of model.model:')\n",
    "for name, child in model.model.named_children():\n",
    "    sz = sum(p.numel() for p in child.parameters()) if hasattr(child, 'parameters') else 0\n",
    "    print(f'  model.model.{name}: {type(child).__name__}  ({sz/1e6:.1f}M params)')\n",
    "\n",
    "# ── Scan ALL ModuleLists and score by transformer-layer signatures ──────────\n",
    "print(f'\\nScanning for decoder layers (need idx {TARGET_LAYER})...')\n",
    "\n",
    "def find_decoder_layer(model, layer_idx):\n",
    "    \"\"\"\n",
    "    Find the nn.Module for decoder layer `layer_idx`.\n",
    "    Fast path: known Gemma-4 hierarchy confirmed by:\n",
    "      (a) v47 DPO adapter_config target_modules regex\n",
    "      (b) named_modules() scan in v3 run (score=410, Gemma4TextDecoderLayer)\n",
    "    Falls back to a scored ModuleList scan for other architectures.\n",
    "    \"\"\"\n",
    "    # Fast path: known Gemma-4 hierarchy\n",
    "    try:\n",
    "        layer = model.model.language_model.layers[layer_idx]\n",
    "        print(f'Layer found via fast path: model.model.language_model.layers[{layer_idx}]')\n",
    "        print(f'  type: {type(layer).__name__}')\n",
    "        return layer\n",
    "    except (AttributeError, IndexError):\n",
    "        pass\n",
    "\n",
    "    # Fallback: scan all ModuleLists, score by transformer-layer signatures\n",
    "    print('Fast path missed — running ModuleList scanner...')\n",
    "    scored = []\n",
    "    for dot_path, module in model.named_modules():\n",
    "        if not isinstance(module, nn.ModuleList):\n",
    "            continue\n",
    "        if len(module) <= layer_idx:\n",
    "            continue\n",
    "        elem = module[0]\n",
    "        score = 0\n",
    "        if hasattr(elem, 'self_attn'):        score += 200\n",
    "        if hasattr(elem, 'mlp'):              score += 100\n",
    "        if hasattr(elem, 'input_layernorm'):  score += 50\n",
    "        if hasattr(elem, 'post_feedforward_layernorm'): score += 25\n",
    "        score += min(len(module), 60)\n",
    "        scored.append((score, dot_path, module))\n",
    "        print(f'  [{score:4d}] model.{dot_path} ({len(module)} items, '\n",
    "              f'elem={type(elem).__name__})')\n",
    "\n",
    "    if not scored:\n",
    "        raise AttributeError(\n",
    "            f'No ModuleList with index {layer_idx} found in '\n",
    "            f'{type(model).__name__}. Direct children of model.model printed above.'\n",
    "        )\n",
    "    scored.sort(key=lambda x: -x[0])\n",
    "    best_score, best_path, best_list = scored[0]\n",
    "    print(f'\\n→ Selected: model.{best_path}[{layer_idx}]  (score={best_score})')\n",
    "    return best_list[layer_idx]\n",
    "\n",
    "target_layer = find_decoder_layer(model, TARGET_LAYER)\n",
    "print(f'Target layer type: {type(target_layer).__name__}')\n",
    "print('\\nLayer attributes:', [a for a in dir(target_layer) if not a.startswith(\"_\")][:20])\n",
])

# ── Cell 4: Collect activations ───────────────────────────────────────────────
md([
    "## Step 1: Collect layer-18 activations (dual hook — last-token)\n",
    "\n",
    "**v5 fix:** Both AR input (`emb`) and AR target (`act`) now use the\n",
    "**last-token position** (`[:, -1, :]`). v4 used mean-pool for the input,\n",
    "which caused FVE to plateau at 0.163 — the mean-pool destroyed the\n",
    "positional signal that attention operates on.\n",
    "\n",
    "With last-token → last-token, the AR approximates the local per-position\n",
    "transform of layer 18. The FFN component is pointwise (fully local) and\n",
    "should be highly approximable by the 4-layer MLP. The attention component\n",
    "adds cross-token signal, but the residual stream already carries prior-layer\n",
    "attention summaries, making the input richer than a mean-pool.\n",
    "\n",
    "- `emb` (AR input):  residual stream ENTERING layer 18 at last-token position\n",
    "- `act` (AR target): residual stream EXITING  layer 18 at last-token position\n",
    "\n",
    "~10 min on T4 at batch_size=8.\n",
])
code([
    "# ── Dual hook: capture last-token position for BOTH input and output ─────────\n",
    "# v5 fix: use last-token position for AR input (was mean-pool in v4)\n",
    "# Reason: attention is global; mean-pool discards per-position context.\n",
    "# last-token → last-token lets the AR approximate the local layer transform.\n",
    "_buf = {}\n",
    "def _hook(m, inp, out):\n",
    "    # inp[0]: residual stream entering the layer  [B, T, D]\n",
    "    # out[0]: residual stream exiting the layer   [B, T, D]\n",
    "    h_in  = inp[0] if isinstance(inp, tuple) else inp\n",
    "    h_out = out[0] if isinstance(out, tuple) else out\n",
    "    # AR input:  last-token residual stream BEFORE layer (v5 fix: was mean-pool)\n",
    "    _buf['emb'] = h_in[:, -1, :].detach().cpu().float()\n",
    "    # AR target: last-token residual stream AFTER  layer\n",
    "    _buf['act'] = h_out[:, -1, :].detach().cpu().float()\n",
    "\n",
    "hook = target_layer.register_forward_hook(_hook)\n",
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
    "        # logits_to_keep=0: skip lm_head (256K-vocab tensor ~1 GB on T4)\n",
    "        # We only need the hook's residual-stream captures, not logits.\n",
    "        _ = model(**inputs, logits_to_keep=0)\n",
    "\n",
    "    act_list.append(_buf['act'].clone())\n",
    "    emb_list.append(_buf['emb'].clone())\n",
    "    n_collected += len(batch_texts)\n",
    "    batch_texts = []\n",
    "    torch.cuda.empty_cache()  # prevent fragmentation across 50K-pass collection\n",
    "\n",
    "    if n_collected % 5_000 == 0:\n",
    "        print(f'  Collected {n_collected:,} / {N_COLLECT:,} pairs ...')\n",
    "    if n_collected >= N_COLLECT:\n",
    "        break\n",
    "\n",
    "hook.remove()\n",
    "\n",
    "all_acts = torch.cat(act_list)\n",
    "all_embs = torch.cat(emb_list)\n",
    "\n",
    "# Validate D_MODEL from actual shape\n",
    "actual_d = all_acts.shape[1]\n",
    "print(f'Actual d_model from activations: {actual_d} (expected {D_MODEL})')\n",
    "if actual_d != D_MODEL:\n",
    "    print(f'  WARNING: updating D_MODEL {D_MODEL} → {actual_d}')\n",
    "    D_MODEL = actual_d\n",
    "\n",
    "del model  # free VRAM for AR training\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "print(f'Collected {len(all_acts):,} pairs. '\n",
    "      f'Act shape: {all_acts.shape}, Emb shape: {all_embs.shape}')\n",
    "\n",
    "# Activation norm stats — expect lower variance between emb and act\n",
    "# with last-token than with mean-pool (mean-pool compresses the distribution)\n",
    "import numpy as np\n",
    "act_norms = all_acts.norm(dim=1)\n",
    "emb_norms = all_embs.norm(dim=1)\n",
    "print(f'Act (layer-out last-token) norm: mean={act_norms.mean():.2f}, std={act_norms.std():.2f}')\n",
    "print(f'Emb (layer-in  last-token) norm: mean={emb_norms.mean():.2f}, std={emb_norms.std():.2f}')\n",
    "\n",
    "# Save to HDF5\n",
    "with h5py.File(OUT_PATH, 'w') as f:\n",
    "    f.create_dataset('activations', data=all_acts.numpy(), compression='gzip')\n",
    "    f.create_dataset('embeddings',  data=all_embs.numpy(), compression='gzip')\n",
    "    f.attrs['n_pairs']   = len(all_acts)\n",
    "    f.attrs['layer_idx'] = TARGET_LAYER\n",
    "    f.attrs['d_model']   = D_MODEL\n",
    "    f.attrs['model_id']  = MODEL_PATH\n",
    "    f.attrs['hook_mode'] = 'last_token_both'  # v5: last-token for both in+out\n",
    "\n",
    "# Stats\n",
    "stats = {\n",
    "    'n_pairs': int(len(all_acts)),\n",
    "    'activation_mean': float(all_acts.mean()),\n",
    "    'activation_std':  float(all_acts.std()),\n",
    "    'activation_norm_mean': float(act_norms.mean()),\n",
    "    'embedding_norm_mean':  float(emb_norms.mean()),\n",
    "    'embedding_dim': int(all_embs.shape[1]),\n",
    "    'd_model': int(D_MODEL),\n",
    "    'hook_mode': 'last_token_both',\n",
    "}\n",
    "print('Activation stats:', json.dumps(stats, indent=2))\n",
    "print(f'Saved to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)')\n",
])

# ── Cell 5: Train AR ──────────────────────────────────────────────────────────
md([
    "## Step 2: Train Activation Reconstructor (AR)\n",
    "\n",
    "4-layer MLP, MSE loss on normalized activations, 3 epochs over 45K pairs,\n",
    "FVE logged every 200 steps on 5K held-out pairs. ~2.5h on T4.\n",
    "\n",
    "AR task: last-token residual-stream-in → last-token residual-stream-out at layer 18.\n",
    "Input dim = d_model (from actual activations), hidden = 2048, output = d_model.\n",
    "\n",
    "LOG_EVERY=200 (was 2000 in v4) so we can see the learning curve rather than\n",
    "just the final plateau point.\n",
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
    "    D_MODEL  = int(f.attrs['d_model'])\n",
    "\n",
    "# Normalize activations (per-feature mean/std)\n",
    "act_mean = all_acts.mean(0, keepdim=True)\n",
    "act_std  = all_acts.std(0,  keepdim=True).clamp(min=1e-6)\n",
    "acts_norm = (all_acts - act_mean) / act_std\n",
    "\n",
    "# Normalize embeddings (same per-feature approach)\n",
    "emb_mean = all_embs.mean(0, keepdim=True)\n",
    "emb_std  = all_embs.std(0,  keepdim=True).clamp(min=1e-6)\n",
    "embs_norm = (all_embs - emb_mean) / emb_std\n",
    "\n",
    "holdout_acts = acts_norm[:N_HOLDOUT]\n",
    "holdout_embs = embs_norm[:N_HOLDOUT]\n",
    "train_acts   = acts_norm[N_HOLDOUT:]\n",
    "train_embs   = embs_norm[N_HOLDOUT:]\n",
    "print(f'Train: {len(train_acts):,} | Holdout: {len(holdout_acts):,}')\n",
    "print(f'Input dim (d_model): {D_MODEL} | Emb dim: {all_embs.shape[1]}')\n",
    "\n",
    "# AR model — input and output both D_MODEL (last-token layer-in → last-token layer-out)\n",
    "HIDDEN_DIM = 2048\n",
    "DEVICE     = 'cuda'\n",
    "\n",
    "class AR(nn.Module):\n",
    "    def __init__(self, d_in, d_out, hidden=2048):\n",
    "        super().__init__()\n",
    "        self.input_proj = nn.Linear(d_in, hidden)\n",
    "        self.layers = nn.ModuleList([\n",
    "            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.05))\n",
    "            for _ in range(3)\n",
    "        ])\n",
    "        self.norms  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])\n",
    "        self.output_proj = nn.Linear(hidden, d_out)\n",
    "\n",
    "    def forward(self, x):\n",
    "        h = F.gelu(self.input_proj(x))\n",
    "        for layer, norm in zip(self.layers, self.norms):\n",
    "            h = norm(h + layer(h))\n",
    "        return self.output_proj(h)\n",
    "\n",
    "ar = AR(d_in=D_MODEL, d_out=D_MODEL, hidden=HIDDEN_DIM).to(DEVICE)\n",
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
    "# LOG_EVERY=200 (was 2000 in v4 — need to see learning curve, not just plateau)\n",
    "LOG_EVERY = 200\n",
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

# ── Cell 6: Evaluate predicates ───────────────────────────────────────────────
md("## Results — predicate evaluation")
code([
    "# Evaluate against H-NLA1 predicates\n",
    "smoke_gate   = final_fve > 0.20\n",
    "nla1b_target = final_fve >= 0.30   # achievable in smoke with last-token fix\n",
    "nla1a_target = final_fve >= 0.40   # likely requires full RunPod run\n",
    "\n",
    "results = {\n",
    "    'final_fve':    final_fve,\n",
    "    'total_steps':  global_step,\n",
    "    'n_train':      len(train_acts),\n",
    "    'n_holdout':    len(holdout_acts),\n",
    "    'fve_curve':    fve_curve,\n",
    "    'hook_mode':    'last_token_both',\n",
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
    "print('NLA Stage 1 SMOKE TEST v5 — RESULTS')\n",
    "print('=' * 60)\n",
    "print(f'  Hook mode:                last_token_both (v5 fix)')\n",
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
    "    print('  → FVE below 0.20 even with last-token fix.')\n",
    "    print('    Check: (1) Are emb/act from same position? (2) Is hook firing?')\n",
    "    print('    (3) Print act/emb norm stats — are they correlated?')\n",
    "    print('    Do NOT proceed to RunPod full run until smoke gate passes.')\n",
    "print()\n",
    "print(f'  Results:       /kaggle/working/nla_smoke_results.json')\n",
    "print(f'  FVE curve:     /kaggle/working/nla_smoke_fve_curve.json')\n",
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
