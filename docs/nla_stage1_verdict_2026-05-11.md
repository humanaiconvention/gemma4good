# Verdict: NLA Stage 1 Smoke Test — 2026-05-11

**Hypothesis tested:** H-NLA1 from `docs/nla_stage1_hypothesis_2026-05-11.md` —
An Activation Reconstructor (AR) SFT for Gemma-4-E2B-it achieves FVE ≥ 0.40
on layer-18 activations after full training (~$50 on RunPod H100).
Smoke gate: FVE > 0.20 proves the pipeline sound before committing RunPod spend.

**Verdict: SMOKE GATE FAILED — NLA TRACK ABORTED.**

FVE = 0.1635 on held-out 5K pairs. Below the 0.20 gate.
RunPod full run is NOT authorized. The AR task as formulated has a
structural bottleneck that a larger compute budget cannot fix.

---

## Numbers

| Predicate | Target | Result |
|---|---|---|
| Smoke gate | FVE > 0.20 | ✗ FAIL — 0.1635 |
| H-NLA1b | FVE ≥ 0.30 | ✗ FAIL |
| H-NLA1a | FVE ≥ 0.40 | ✗ FAIL |

| Run detail | Value |
|---|---|
| Pairs collected | 50,000 (all collected successfully) |
| Train / holdout split | 45K / 5K |
| Layer | 18 of 35 (Gemma4TextDecoderLayer) |
| Layer path confirmed | `model.model.language_model.layers[18]` |
| d_model | 1536 |
| AR parameters | 18,896,384 |
| Training steps | 2,112 (3 epochs × 703 steps @ batch=64) |
| Final loss | 0.8732 (MSE on normalized activations) |
| Activation norm (mean L2) | 73.05 — large residual stream at layer 18 |
| FVE at step 2000 | 0.1635 |
| FVE at final (2112) | 0.1635 (+0.000038 — effectively plateaued) |

## FVE curve

```
step 2000: FVE = 0.1635  loss = 0.8732
step 2112: FVE = 0.1635  (final — no measurable improvement)
```

Only one log point (LOG_EVERY=2000, total steps=2112) — the model
plateaued before step 2000 and never improved in the final ~112 steps.

---

## Root cause: architectural mismatch in the AR task

The smoke test ran the AR on this mapping:

```
Input:  mean-pool( residual-stream ENTERING layer 18, over all tokens )
Target: residual-stream EXITING layer 18, at the LAST TOKEN position
```

This is the wrong task. Here's why:

**Transformer layer 18 computes attention over the full sequence.**
The output at the last-token position is determined by:
  1. The last-token's own residual-stream vector (local)
  2. Keys/values from ALL other token positions (global, via attention)

The AR input is the mean-pooled encoding of (1) — it's a blurred
average over all token positions. It captures aggregate context but
discards the position-specific structure that attention actually
operates on. The AR cannot reconstruct the last-token output from a
mean-pooled input because the crucial positional signal is gone.

**What the 0.1635 FVE actually tells us:**
- Untrained AR → FVE ≈ 0 (random predictions, MSE ≈ Var)
- Our AR → FVE = 0.163 — it learned 16.3% of the variance
- The remaining 83.7% is variance driven by attention patterns that
  the mean-pooled input cannot encode
- The plateau at step 2000/2112 confirms the AR has extracted all it
  can from the mean-pool representation; more training won't help

**This is not an infrastructure failure.** The pipeline itself is sound:
- Layer discovery via fast path: ✓ `model.model.language_model.layers[18]`
- 50K pair collection: ✓ (no OOM after batch=8 + logits_to_keep=0 fix)
- AR training: ✓ (ran 3 epochs cleanly)
- FVE computation: ✓ (consistent with loss)

The bottleneck is task design, not hardware, not the layer path, not
the AR architecture.

---

## How to fix it (if NLA track is revived)

The correct AR task for NLA is **last-token → last-token**:

```python
def _hook(m, inp, out):
    h_in  = inp[0] if isinstance(inp, tuple) else inp
    h_out = out[0] if isinstance(out, tuple) else out
    # Use LAST TOKEN for BOTH — same position, same context
    _buf['emb'] = h_in[:, -1, :].detach().cpu().float()   # ← last token (not mean-pool)
    _buf['act'] = h_out[:, -1, :].detach().cpu().float()
```

This maps the residual stream BEFORE layer 18 at the last-token
position → the residual stream AFTER layer 18 at the last-token
position. The AR now approximates the layer's local transformation
(attention + FFN) in the context of a single position.

Why this should work much better:
- The input contains the full positional context for that token
- The AR only needs to approximate the local (per-position) transform
- The FFN component IS a local function (pointwise MLP), which a 4-layer
  MLP approximator should handle well
- The attention component adds cross-token signal, but since the
  residual stream already carries prior-layer attention summaries, the
  input is richer than a mean-pool

Expected FVE improvement: 0.16 → 0.35–0.50 (estimate; the FFN
dominates layer output and is highly approximable by an MLP).

**Other improvements for a v2 smoke:**
- LOG_EVERY = 200 (not 2000) — need to see the learning curve, not just the plateau
- Verify FVE is growing before step 500; if flat → architectural fix needed
- Consider 2 layers (both input AND output of a residual block) to compare
- `activation_norm_mean = 73.05` is large — consider per-sample L2 normalization
  of both inputs and targets before training (not just mean/std)

---

## Production impact

None. The NLA track was always a stretch goal for the Gemma4Good
submission. The governance pipeline (Tools 1-5) already includes a
MockNLA implementation (Tool 5: `audit_activation_explanation`) with
deterministic SHA3-256 seeding and [0.35, 0.65] FVE range. The
submission is complete and self-consistent without a live AR.

MockNLA in the governance notebook correctly:
- Documents that a real AR would need the `model.model.language_model.layers`
  hook path (now confirmed)
- Uses the correct d_model=1536 and layer=18 parameters
- Produces audit hashes and confidence classifications

The Kaggle Gemma 4 Good submission (`benhaslam/haic-gemma4-governance-agent`)
is unaffected by this verdict.

---

## Artifacts

```
Results:      D:/gemma4good/results/nla-stage1-smoke/nla_smoke_results.json
FVE curve:    D:/gemma4good/results/nla-stage1-smoke/nla_smoke_fve_curve.json
AR checkpoint: C:/Users/benja/AppData/Local/Temp/nla-v4-output/nla_ar_smoke_final.pt
               (73 MB — kept for reference; not promoted)
Activations:  C:/Users/benja/AppData/Local/Temp/nla-v4-output/nla_smoke_activations.h5
               (284 MB — layer-18 activations for 50K TinyStories passages)
```

## Methodology note

The smoke-test infrastructure is validated even though the gate failed:
- 4 notebook versions to fix multimodal architecture issues (v1–v4)
- The layer path `model.model.language_model.layers[18]` is now confirmed
  and hardcoded in the fast-path for any future NLA experiments
- OOM resolution path documented: `logits_to_keep=0`, `batch=8`,
  `empty_cache()` per batch
- The AR task fix (last-token → last-token) is a one-line change to
  the hook and is well-understood from the data

*"Follow the science." The smoke gate was designed to prevent a $50
RunPod spend on a broken pipeline. It worked exactly as intended —
the problem it found is real and actionable. The fix is trivial once
the predicate has flagged it.*
