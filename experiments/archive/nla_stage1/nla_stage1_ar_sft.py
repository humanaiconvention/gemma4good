#!/usr/bin/env python3
"""
nla_stage1_ar_sft.py — Activation Reconstructor (AR) SFT for Gemma-4-E2B-it.

Stage 1 of the HAIC NLA training pipeline.
See docs/nla_stage1_hypothesis_2026-05-11.md for the full plan and falsifiable
predicates (H-NLA1a: FVE >= 0.40, H-NLA1b: FVE >= 0.30).

## What this script does

1. Loads Gemma-4-E2B-it and hooks the residual stream at layer 18.
2. Passes a text corpus through the model in forward-pass-only mode and
   collects (passage, activation) pairs.
3. Trains a small MLP Activation Reconstructor (AR) to predict the layer-18
   activation from passage tokens (mean-pooled from a lightweight encoder).
4. Logs FVE on a held-out 10K split every LOG_EVERY steps.
5. Saves checkpoints and a training log.

## Usage

    # Full run (RunPod H100, ~20h, ~$50):
    python experiments/nla_stage1_ar_sft.py --mode full

    # Smoke test (Kaggle T4, ~3h, free):
    python experiments/nla_stage1_ar_sft.py --mode smoke

    # Resume from checkpoint:
    python experiments/nla_stage1_ar_sft.py --mode full --resume /path/to/ar_model_step50000.pt

## Falsifiable predicates

    H-NLA1a: final_fve >= 0.40   (full run target)
    H-NLA1b: final_fve >= 0.30   (marginal viability)
    smoke_floor: final_fve >= 0.20  (smoke test validation floor)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_MODEL_ID = "google/gemma-4-2b-it"  # Gemma-4-E2B-it on HuggingFace
TARGET_LAYER = 18                           # residual stream layer to hook
D_MODEL = 1536                              # Gemma-4-E2B residual stream width

CORPUS_NAME = "roneneldan/TinyStories"     # HF dataset for activation collection
MAX_SEQ_LEN = 128                           # passage truncation length

# Smoke test (Kaggle T4, ~3h)
SMOKE_COLLECT  = 50_000   # passages to collect
SMOKE_TRAIN    = 50_000   # pairs to train on
SMOKE_HOLDOUT  = 5_000    # held-out pairs for FVE
SMOKE_EPOCHS   = 3
SMOKE_BATCH    = 32

# Full run (RunPod H100, ~20h)
FULL_COLLECT   = 500_000
FULL_TRAIN     = 500_000
FULL_HOLDOUT   = 10_000
FULL_EPOCHS    = 1
FULL_BATCH     = 64


# ── AR model ──────────────────────────────────────────────────────────────────

class ActivationReconstructor(nn.Module):
    """Small MLP that predicts a residual-stream activation from passage embeddings.

    Architecture: 4-layer MLP with GELU activations and residual connections.
    Input: mean-pooled token embeddings from a lightweight sentence encoder
           (or directly from a small embedding projection layer).
    Output: d_model-dimensional activation vector.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048,
                 output_dim: int = D_MODEL, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(3)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) mean-pooled embeddings → (batch, d_model)"""
        h = F.gelu(self.input_proj(x))
        for layer, norm in zip(self.layers, self.layer_norms):
            h = norm(h + layer(h))
        return self.output_proj(h)


# ── Activation collection ─────────────────────────────────────────────────────

@dataclass
class CollectionConfig:
    model_id: str = TARGET_MODEL_ID
    layer_idx: int = TARGET_LAYER
    d_model: int = D_MODEL
    corpus_name: str = CORPUS_NAME
    n_passages: int = FULL_COLLECT
    max_seq_len: int = MAX_SEQ_LEN
    batch_size: int = 16
    out_dir: Path = Path("data/nla_activations")
    device: str = "cuda"


def collect_activations(cfg: CollectionConfig) -> Path:
    """Collect (passage_embedding, layer-l activation) pairs from Gemma-4-E2B-it.

    Returns: path to saved HDF5 file (or directory of .pt shards if h5py unavailable).
    """
    try:
        import h5py
        HAS_H5PY = True
    except ImportError:
        HAS_H5PY = False
        log.warning("h5py not found — saving activations as .pt shards.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"layer{cfg.layer_idx}_n{cfg.n_passages}.h5"
    if out_path.exists():
        log.info("Activation cache already exists: %s — skipping collection.", out_path)
        return out_path

    log.info("Loading target model %s …", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
    )
    model.eval()

    # Hook: capture last-token residual stream at target layer
    _acts: dict[str, torch.Tensor] = {}
    def _hook(module, inp, out):
        # out[0]: (batch, seq_len, d_model)
        _acts["act"] = out[0][:, -1, :].detach().cpu().float()

    hook_handle = model.model.layers[cfg.layer_idx].register_forward_hook(_hook)

    log.info("Loading corpus %s …", cfg.corpus_name)
    dataset = load_dataset(cfg.corpus_name, split="train", streaming=True)

    passages: list[str] = []
    act_list: list[torch.Tensor] = []
    emb_list: list[torch.Tensor] = []

    n_collected = 0
    for item in dataset:
        text = item.get("text", item.get("story", ""))
        if not text or len(text.strip()) < 20:
            continue
        passages.append(text[:512])  # truncate for memory
        if len(passages) < cfg.batch_size and n_collected + len(passages) < cfg.n_passages:
            continue

        # Tokenize and forward
        inputs = tokenizer(
            passages, return_tensors="pt",
            padding=True, truncation=True, max_length=cfg.max_seq_len,
        ).to(cfg.device)
        with torch.no_grad():
            # Also collect mean-pooled embeddings as AR input
            embed_out = model.model.embed_tokens(inputs["input_ids"])
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            mean_emb = (embed_out * mask).sum(1) / mask.sum(1)  # (B, embed_dim)
            _ = model(**inputs)  # triggers hook

        act = _acts["act"]                          # (B, d_model)
        emb = mean_emb.detach().cpu().float()       # (B, embed_dim)

        act_list.append(act)
        emb_list.append(emb)
        n_collected += len(passages)
        passages = []

        if n_collected % 10_000 == 0:
            log.info("  collected %d / %d pairs …", n_collected, cfg.n_passages)

        if n_collected >= cfg.n_passages:
            break

    hook_handle.remove()
    del model  # free VRAM

    all_acts = torch.cat(act_list, dim=0)   # (N, d_model)
    all_embs = torch.cat(emb_list, dim=0)   # (N, embed_dim)
    log.info("Collected %d pairs. Saving to %s …", len(all_acts), out_path)

    if HAS_H5PY:
        with h5py.File(out_path, "w") as f:
            f.create_dataset("activations", data=all_acts.numpy(), compression="gzip")
            f.create_dataset("embeddings",  data=all_embs.numpy(), compression="gzip")
            f.attrs["n_pairs"] = len(all_acts)
            f.attrs["layer_idx"] = cfg.layer_idx
            f.attrs["d_model"] = cfg.d_model
            f.attrs["model_id"] = cfg.model_id
    else:
        torch.save({"activations": all_acts, "embeddings": all_embs}, out_path.with_suffix(".pt"))
        out_path = out_path.with_suffix(".pt")

    # Statistics for the hypothesis doc
    stats = {
        "n_pairs": int(len(all_acts)),
        "activation_mean": float(all_acts.mean()),
        "activation_std":  float(all_acts.std()),
        "activation_norm_mean": float(all_acts.norm(dim=1).mean()),
        "embedding_dim": int(all_embs.shape[1]),
    }
    stats_path = cfg.out_dir / "activation_sample_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Stats: %s", stats)
    return out_path


# ── FVE computation ──────────────────────────────────────────────────────────

def compute_fve(model: ActivationReconstructor, acts: torch.Tensor,
                embs: torch.Tensor, batch_size: int = 256) -> float:
    """Fraction of Variance Explained on a held-out set.

    FVE = 1 - MSE(predicted, actual) / Var(actual)
    """
    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(embs), batch_size):
            e = embs[i:i+batch_size].to(next(model.parameters()).device)
            preds.append(model(e).cpu())
    pred = torch.cat(preds)
    mse = F.mse_loss(pred, acts[:len(pred)]).item()
    var = acts[:len(pred)].var().item()
    fve = 1.0 - mse / max(var, 1e-10)
    return round(float(fve), 6)


# ── Training loop ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    n_train: int = FULL_TRAIN
    n_holdout: int = FULL_HOLDOUT
    n_epochs: int = FULL_EPOCHS
    batch_size: int = FULL_BATCH
    learning_rate: float = 1e-4
    warmup_steps: int = 2_000
    log_every: int = 5_000
    save_every: int = 25_000
    ar_hidden_dim: int = 2_048
    ar_dropout: float = 0.05
    device: str = "cuda"
    out_dir: Path = Path("data/nla_ar_checkpoints")
    resume_from: str | None = None
    # Predicates
    fve_target_pass: float = 0.40
    fve_target_marginal: float = 0.30
    fve_smoke_floor: float = 0.20


def train_ar(activation_path: Path, cfg: TrainConfig) -> dict:
    """Train the AR and return a results dict with the final FVE and predicate verdicts."""
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, random_split

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.out_dir / "training_log.jsonl"
    fve_path  = cfg.out_dir / "fve_curve.json"

    # Load activations
    log.info("Loading activations from %s …", activation_path)
    if activation_path.suffix == ".h5":
        import h5py
        with h5py.File(activation_path, "r") as f:
            acts = torch.from_numpy(f["activations"][:])
            embs = torch.from_numpy(f["embeddings"][:])
    else:
        d = torch.load(activation_path, weights_only=True)
        acts = d["activations"]
        embs = d["embeddings"]

    # Normalize activations (improves MSE convergence)
    act_mean = acts.mean(0, keepdim=True)
    act_std  = acts.std(0, keepdim=True).clamp(min=1e-6)
    acts_norm = (acts - act_mean) / act_std

    n = min(len(acts_norm), cfg.n_train + cfg.n_holdout)
    acts_norm = acts_norm[:n]
    embs      = embs[:n]

    holdout_acts = acts_norm[:cfg.n_holdout]
    holdout_embs = embs[:cfg.n_holdout]
    train_acts   = acts_norm[cfg.n_holdout:cfg.n_holdout + cfg.n_train]
    train_embs   = embs[cfg.n_holdout:cfg.n_holdout + cfg.n_train]

    log.info("Train: %d pairs | Holdout: %d pairs | Embedding dim: %d → d_model: %d",
             len(train_acts), len(holdout_acts), embs.shape[1], D_MODEL)

    input_dim = embs.shape[1]
    ar = ActivationReconstructor(
        input_dim=input_dim,
        hidden_dim=cfg.ar_hidden_dim,
        output_dim=D_MODEL,
        dropout=cfg.ar_dropout,
    ).to(cfg.device)
    log.info("AR parameters: %s", sum(p.numel() for p in ar.parameters()))

    if cfg.resume_from:
        log.info("Resuming from %s", cfg.resume_from)
        ar.load_state_dict(torch.load(cfg.resume_from, weights_only=True))

    dataset = TensorDataset(train_embs, train_acts)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(ar.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    total_steps = cfg.n_epochs * len(loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.learning_rate,
        total_steps=total_steps, pct_start=cfg.warmup_steps / max(total_steps, 1),
    )

    fve_curve: list[dict] = []
    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.n_epochs):
        ar.train()
        for emb_b, act_b in loader:
            emb_b = emb_b.to(cfg.device)
            act_b = act_b.to(cfg.device)

            pred = ar(emb_b)
            loss = F.mse_loss(pred, act_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % cfg.log_every == 0:
                fve = compute_fve(ar, holdout_acts, holdout_embs)
                elapsed = time.time() - t0
                log_entry = {
                    "step": global_step,
                    "loss": round(float(loss.item()), 6),
                    "fve": fve,
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "elapsed_s": round(elapsed),
                }
                fve_curve.append(log_entry)
                log.info("step %d  loss=%.4f  FVE=%.4f  lr=%.2e  elapsed=%ds",
                         global_step, loss.item(), fve, scheduler.get_last_lr()[0], elapsed)
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                ar.train()

            if global_step % cfg.save_every == 0:
                ckpt = cfg.out_dir / f"ar_model_step{global_step}.pt"
                torch.save(ar.state_dict(), ckpt)
                log.info("Saved checkpoint: %s", ckpt)

    # Final evaluation
    final_fve = compute_fve(ar, holdout_acts, holdout_embs)
    log.info("Final FVE (held-out, %d pairs): %.4f", cfg.n_holdout, final_fve)

    # Save final checkpoint
    final_ckpt = cfg.out_dir / "ar_model_final.pt"
    torch.save(ar.state_dict(), final_ckpt)
    log.info("Saved final checkpoint: %s", final_ckpt)

    # Save FVE curve
    fve_curve.append({"step": global_step, "fve": final_fve, "final": True})
    fve_path.write_text(json.dumps(fve_curve, indent=2))

    # Evaluate predicates
    results = {
        "final_fve": final_fve,
        "total_steps": global_step,
        "n_train": len(train_acts),
        "n_holdout": len(holdout_acts),
        "fve_curve": fve_curve,
        "predicates": {
            "H_NLA1a (fve >= 0.40)": {
                "predicate": "final_fve >= 0.40",
                "actual": final_fve,
                "passed": final_fve >= 0.40,
                "verdict": "PASS" if final_fve >= 0.40 else "FAIL",
            },
            "H_NLA1b (fve >= 0.30)": {
                "predicate": "final_fve >= 0.30",
                "actual": final_fve,
                "passed": final_fve >= 0.30,
                "verdict": "PASS" if final_fve >= 0.30 else "FAIL",
            },
        },
        "recommendation": (
            "AUTHORIZE_STAGE2" if final_fve >= 0.40
            else "REQUEST_OPERATOR_DECISION" if final_fve >= 0.30
            else "ABORT_NLA_TRACK"
        ),
    }

    results_path = cfg.out_dir / "stage1_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results: %s", results["recommendation"])
    log.info("  H-NLA1a (FVE >= 0.40): %s (actual=%.4f)",
             results["predicates"]["H_NLA1a (fve >= 0.40)"]["verdict"], final_fve)
    log.info("  H-NLA1b (FVE >= 0.30): %s (actual=%.4f)",
             results["predicates"]["H_NLA1b (fve >= 0.30)"]["verdict"], final_fve)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["full", "smoke"], default="smoke",
                        help="'smoke': Kaggle T4 ~3h (50K pairs); 'full': RunPod H100 ~20h (500K pairs)")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER,
                        help=f"Target layer index (default: {TARGET_LAYER})")
    parser.add_argument("--out-dir", type=Path, default=Path("data/nla_ar_checkpoints"),
                        help="Directory for checkpoints and logs")
    parser.add_argument("--acts-dir", type=Path, default=Path("data/nla_activations"),
                        help="Directory for activation cache")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to AR checkpoint to resume from")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip activation collection (use existing cache)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.mode == "smoke":
        collect_n = SMOKE_COLLECT
        train_cfg = TrainConfig(
            n_train=SMOKE_TRAIN, n_holdout=SMOKE_HOLDOUT,
            n_epochs=SMOKE_EPOCHS, batch_size=SMOKE_BATCH,
            device=args.device, out_dir=args.out_dir, resume_from=args.resume,
        )
        log.info("Mode: SMOKE TEST (%d pairs, %d epochs) — free on Kaggle T4 (~3h)",
                 SMOKE_TRAIN, SMOKE_EPOCHS)
    else:
        collect_n = FULL_COLLECT
        train_cfg = TrainConfig(
            n_train=FULL_TRAIN, n_holdout=FULL_HOLDOUT,
            n_epochs=FULL_EPOCHS, batch_size=FULL_BATCH,
            device=args.device, out_dir=args.out_dir, resume_from=args.resume,
        )
        log.info("Mode: FULL RUN (%d pairs, %d epoch) — RunPod H100 (~20h, ~$50)",
                 FULL_TRAIN, FULL_EPOCHS)

    # Step 1: collect activations
    if not args.skip_collection:
        collect_cfg = CollectionConfig(
            layer_idx=args.layer,
            n_passages=collect_n,
            out_dir=args.acts_dir,
            device=args.device,
        )
        activation_path = collect_activations(collect_cfg)
    else:
        # Find existing cache
        candidates = list(args.acts_dir.glob(f"layer{args.layer}_n*.h5")) + \
                     list(args.acts_dir.glob(f"layer{args.layer}_n*.pt"))
        if not candidates:
            log.error("--skip-collection but no activation cache found in %s", args.acts_dir)
            return 1
        activation_path = sorted(candidates)[-1]
        log.info("Using existing activation cache: %s", activation_path)

    # Step 2: train AR
    train_cfg.out_dir = args.out_dir
    results = train_ar(activation_path, train_cfg)

    print()
    print("=" * 60)
    print(f"NLA Stage 1 AR SFT — {args.mode.upper()} COMPLETE")
    print("=" * 60)
    print(f"  Final FVE:       {results['final_fve']:.4f}")
    print(f"  H-NLA1a (≥0.40): {results['predicates']['H_NLA1a (fve >= 0.40)']['verdict']}")
    print(f"  H-NLA1b (≥0.30): {results['predicates']['H_NLA1b (fve >= 0.30)']['verdict']}")
    print(f"  Recommendation:  {results['recommendation']}")
    print()
    if results["recommendation"] == "AUTHORIZE_STAGE2":
        print("  → H-NLA1a PASSED. Stage 2 (AV+AR RL, ~$2,000) is authorized.")
        print("    Register the AR checkpoint in prism.nla.registry and proceed.")
    elif results["recommendation"] == "REQUEST_OPERATOR_DECISION":
        print("  → FVE is marginal (0.30–0.40). Stage 2 may converge but with higher")
        print("    variance. Request operator decision before spending ~$2,000.")
    else:
        print("  → FVE below 0.30. ABORT NLA track. Wait for community Gemma-4 NLA.")
        print("    See docs/nla_training_cost_analysis_2026-05-11.md for alternatives.")
    print()
    print(f"  Artifacts: {args.out_dir}/stage1_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
