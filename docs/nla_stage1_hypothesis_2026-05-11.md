# NLA Stage 1 Hypothesis: AR SFT on Gemma-4-E2B — 2026-05-11

**Status:** PENDING — operator authorized Stage 1 (~$50), training not yet run.

---

## Context

The v46 cycle refuted H4e and confirmed that v42's 13.8% strict-refusal rate
is the current behavioral ceiling. The governance pipeline mitigates this via
Tools 5+6 (NLA audit + MPK provenance). Tool 5 currently uses MockNLA because
**no NLA has been trained for Gemma-4-E2B-it**.

The `nla_training_cost_analysis_2026-05-11.md` decision doc analyzed the full
training cost (~$2,050 to first running NLA) and recommended deferring. The
operator has authorized Stage 1 standalone ($50) as a pipeline-validation gate.

---

## Hypothesis H-NLA1

**Stage 1 can produce a warm-start Activation Reconstructor (AR) for
Gemma-4-E2B-it that achieves FVE ≥ 0.4 on held-out activations after
supervised fine-tuning (SFT) alone.**

### Falsifiable predicates

```
H-NLA1a: stage1_fve >= 0.40   (Anthropic SFT warm-start floor, from NLA paper)
H-NLA1b: stage1_fve >= 0.30   (minimal viability — Stage 2 might still converge)
```

**Decision tree:**
- FVE ≥ 0.40 → PASS. Authorize Stage 2 (~$2,000). AR is at the warm-start
  floor where Anthropic's RL training has historically succeeded.
- 0.30 ≤ FVE < 0.40 → MARGINAL. Request operator decision before Stage 2.
  Stage 2 may still converge but with higher variance.
- FVE < 0.30 → ABORT NLA track. The Gemma-4-E2B activation distribution at
  the target layer may not be amenable to NLA at this model scale with this
  data budget. Wait for community-trained Gemma-4 NLA instead.

---

## What Stage 1 is

The Activation Reconstructor (AR) maps natural-language text back into the
model's residual stream activation space. It is trained to minimize MSE between
the AR's predicted activation and the actual layer-l activation from
Gemma-4-E2B-it.

Training data: (text passage, layer-l residual stream activation) pairs
collected by running a pretraining-like corpus through Gemma-4-E2B-it with
activation hooks at the target layer.

The AR is a separate model (typically an MLP or small transformer) that learns
to invert: "given this text, predict what the target model's hidden state
looked like while processing it." High FVE means the AR's predictions capture
most of the variance in real activations — a necessary precondition for Stage 2
(where an Activation Verbalizer learns to describe activations in natural
language, and the AR is used as a reconstruction-quality reward signal).

Stage 1 does NOT produce interpretable explanations. It produces a
reconstruction capability that is the required prerequisite for Stage 2.

---

## Training plan

### Data collection

**Corpus:** `roneneldan/TinyStories` (HuggingFace, ~476 MB, permissive license)
- Rationale: short, clean, diverse narratives; no copyright risk; domain-neutral.
- Quantity: 500K passages (roughly 1B tokens equivalent for a 2B model).
- Alternative: `EleutherAI/pile-uncopyrighted` subset if more variety needed.

**Layer selection:** Layer 18 of 28 (Gemma-4-E2B-it has 28 transformer blocks).
- Layer 18 is mid-to-late (64% depth), where Anthropic's NLA paper reports
  the best FVE/interpretability trade-off for small models.
- Fallback: also collect layer 24 activations to test both depths.

**Activation collection procedure:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, h5py

model_id = "google/gemma-4-2b-it"   # Gemma-4-E2B-it on HuggingFace
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# Register hook at layer 18 residual stream output
acts = {}
model.model.layers[18].register_forward_hook(
    lambda m, i, o: acts.update({"layer18": o[0][:, -1, :].detach().cpu()})
)
# Run corpus passages; save (passage_text, activation) pairs to HDF5
```

**Storage:** ~500K vectors × 1536 dims × 2 bytes (bfloat16) = ~1.5 GB HDF5.

### AR architecture

Small MLP (4-layer, hidden_dim=2048) that takes the tokenized passage
(mean-pooled embedding from a small encoder, or directly tokenized)
and predicts the d_model=1536 target activation.

Reference: kitft repo `natural_language_autoencoders` AR architecture.
Specifically `models/ar_model.py` — use their default MLP config for
sub-7B target models.

### Training config

```
Hardware:   1× H100-80GB (RunPod, spot instance, ~$2.50/hr → ~$50 for 20h)
Epochs:     1 pass over 500K pairs
Batch size: 64
LR:         1e-4 (cosine decay, 2K warmup steps)
Loss:       MSE on bfloat16 activations (normalized by activation std)
Checkpoint: every 5K steps + final
Logging:    FVE on held-out 10K pairs every 5K steps
Duration:   ~20h wall clock at 500K pairs, batch=64 on 1×H100
```

**FVE formula:**
```
FVE = 1 - MSE(predicted, actual) / Var(actual)
```
where Var is computed on the same held-out 10K pairs. Values should
progress: ~0.15 at init → ~0.30 at 50K steps → ~0.38-0.42 at 500K steps
(based on Anthropic's reported curves for similarly-sized target models).

---

## Infrastructure

### Option A: RunPod (recommended)

```bash
# Provision
runpod_id=$(runpodctl create pod \
    --gpuType "NVIDIA H100 80GB HBM3" \
    --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --containerDiskInGb 100 \
    --volumeInGb 50 \
    --spotInstance true)

# Deploy training script
rsync -av experiments/nla_stage1_ar_sft.py root@$POD_IP:/workspace/
rsync -av data/nla_stage1_config.json root@$POD_IP:/workspace/

# Run
ssh root@$POD_IP "cd /workspace && python nla_stage1_ar_sft.py --config nla_stage1_config.json"
```

### Option B: Kaggle (free, but T4-constrained)

Kaggle T4 (16 GB VRAM) can collect activations and train the AR, but:
- Gemma-4-E2B-it in bfloat16 = ~4 GB VRAM. AR MLP adds ~0.5 GB. Fits on T4.
- BUT 500K pairs × 20 epochs takes ~40h on T4 (Kaggle limit: 12h/session).
- Workaround: 50K pairs, 3 epochs, ~3h on T4. FVE will be lower (~0.25-0.30),
  making this a smoke test rather than a full Stage 1 run.

**Recommendation:** Kaggle Smoke Test first (free, 3h, ~50K pairs) to validate
the pipeline end-to-end. If FVE > 0.20 on the smoke test, proceed to RunPod
for the full $50 Stage 1 run.

---

## Output artifacts

```
D:/kaggle/results/nla-stage1/
  ar_model_final.pt          ← AR checkpoint at end of training
  ar_model_step50000.pt      ← intermediate checkpoint
  fve_curve.json             ← FVE at each logging step (10K held-out pairs)
  training_log.jsonl         ← per-step loss + FVE
  config.json                ← exact training config (reproducibility)
  activation_sample_stats.json ← mean/std/norm statistics of collected acts
```

---

## How Stage 1 feeds back into the submission

If FVE ≥ 0.40:
1. The AR is the prerequisite for Stage 2 (AV SFT + RL, ~$2,000).
2. Even before Stage 2, the AR itself can be used for a weaker form of
   activation audit: "does the reconstruction error spike on concealed-
   compliance probes?" High reconstruction error = the model's activation
   at that prompt is unusual relative to the training distribution.
3. The `prism_integration/nla.py` adapter gets updated: AR checkpoint
   registers in `prism.nla.registry` as `haic/nla-gemma4-e2b-layer18-ar-sft`.
4. `audit_activation_explanation` in the notebook returns FVE from the real
   AR rather than MockNLA's deterministic 0.35-0.65 range.

---

## Files

- `experiments/nla_stage1_ar_sft.py` — training script (to be deployed on RunPod)
- `notebook/nla_stage1_kaggle_smoke_test.ipynb` — Kaggle smoke test notebook
- `data/nla_stage1_config.json` — training configuration
- `prism_integration/nla.py` — AR adapter (add real checkpoint after training)
- `D:/prism/src/prism/nla/registry.py` — add entry after training

---

*Hypothesis committed 2026-05-11. The FVE on the held-out 10K pairs at training
end is the single arbiter. Training metrics during the run are informative but
not decisive — FVE can plateau or regress near the end of SFT.*
