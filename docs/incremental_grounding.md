# Incremental Grounding — Technical Design

**Module:** `tools/incremental_grounding.py`
**Tracker:** `viability/grounding_tracker.py`
**Status:** Implemented. Dry-run mode tested (51/51 tests pass). Live mode requires GPU.

---

## Overview

Incremental grounding extends the HAIC governance pipeline from inference-time enforcement to **training-time operationalization** of the Viability Condition. After each consented interview session, the system can optionally update a LoRA adapter on a local Gemma 4 E2B model, encoding the participant's lived experience directly into the model's weights.

Every step of this process — consent validation, SFT pair extraction, adapter state hashing, loss computation — is Merkle-committed. The receipt proves that the weight update was (a) consent-gated, (b) traceable to a specific interview session, and (c) verifiable by any third party.

### Design Principle: No Hallucinated Data

> Every number in a receipt comes from a real computation, or it is explicitly `None`.
> No simulated loss trajectories. No placeholder adapter hashes. No estimated metrics.

This is the Viability Condition applied to the pipeline itself: if we allow simulated training artifacts into the receipt chain, we are introducing uncorrected synthetic data — the exact failure mode the framework exists to prevent.

---

## Session-to-SFT Pipeline

### Input: HAIC Interview Session

A 7-turn HAIC grounding interview:

```
messages[0]: system   — interviewer prompt (base_interviewer.txt)
messages[1]: user     — T1: participant's lived experience
messages[2]: assistant — T2: [PIVOT: TYPE] pivot question
messages[3]: user     — T3: response to pivot
messages[4]: assistant — T4: texture follow-up
messages[5]: user     — T5: sensory response
messages[6]: assistant — T6: [COMPRESSION:] compression question
```

### Output: SFT Training Pairs

Three training windows per session:

| Window | Context | Target | Training Signal |
|---|---|---|---|
| T2 | T1 (user's experience) | Pivot question | Learn to ask grounding pivots |
| T4 | T1–T3 (experience + pivot response) | Texture follow-up | Learn to deepen grounding |
| T6 | T1–T5 (full dialog) | Compression question | Learn to compress and verify |

Each pair includes the full conversational context up to that point, formatted as:
```json
{
  "instruction": "<preceding user turn>",
  "response": "<assistant response>",
  "turn_index": 2,
  "context_messages": [{"role": "user", "content": "..."}]
}
```

### Consent Gate

`format_session_as_sft()` checks `consent["training_signal"]` before extracting any pairs. If not `"granted"`, returns an empty list. The gate is absolute: no partial extraction, no override.

---

## Two-Level Merkle Training Receipt

### Structure

```
Training Receipt Merkle Tree:
├── Leaf 1: SHA-256(session_receipt_root)     — links to session provenance
├── Leaf 2: SHA-256(adapter_state_before)     — adapter hash pre-update
├── Leaf 3: SHA-256(training_config)          — lr, steps, target_modules, etc.
├── Leaf 4: SHA-256(loss_trajectory)          — per-step loss values
├── Leaf 5: SHA-256(adapter_state_after)      — adapter hash post-update
└── Root:   training_receipt_root              — pairwise SHA-256 reduction
```

### None Handling

Any `None` leaf (e.g., adapter hashes in dry-run mode) is hashed as `SHA-256("null")`. This is explicit in the receipt — a verifier can see which fields were null and know that training did not execute.

### Linking to Session Receipt

The `session_receipt_root` (from `generate_receipt()`) is the first leaf. This creates a cryptographic chain:

```
Interview Session → Session Receipt → Training Receipt
     (data)            (Merkle root)     (weight update proof)
```

A verifier can prove: "This model update was (a) triggered by a consented session, (b) used the documented training configuration, and (c) produced specific weight changes, and (d) the session that generated the signal was itself Merkle-verified."

---

## Grounding Tracker

`viability/grounding_tracker.py` maintains an append-only log of sessions with cumulative C(t) computation.

### Key Properties

- **`cumulative_ceff()`** — total consented SFT pairs contributed. Only counts sessions where `training_signal == "granted"`.
- **`viability_trend()`** — per-session time series of cumulative C(t) and C(t)/E(t) ratio.
- **`monotonically_improving()`** — verifies C(t) is non-decreasing. Must always be True by construction (sessions only add, never subtract).

### E(t) Assumption

E(t) is held constant at the PRISM `quantization_hostility` value (default: 0.9146 for E2B). This is justified by the empirical finding that LoRA fine-tuning does not change activation geometry — confirmed across 4 independent adapters on 2 base models (v6/v7/v8 on Qwen3.5-2B, v1 on Gemma 4 E2B). See PRISM geometry audit results.

---

## LoRA Update Configuration

### Recommended Configuration (T4 GPU)

```python
GroundingConfig(
    mode="live",
    lr=5e-5,              # conservative; v1 used 1e-4 and hit NaN at step 80
    steps=5,              # 3-10 per session
    lora_r=8,             # low rank for minimal VRAM
    lora_alpha=16,        # alpha = 2*r
    target_modules=["q_proj", "v_proj"],
    max_seq_len=512,      # matches E2B sliding window
    compute_dtype="float16",  # T4 does NOT support bfloat16
    model_id="google/gemma-4-E2B-it",
)
```

### VRAM Budget (per T4 = 16 GB)

| Component | VRAM |
|---|---:|
| Base model (4-bit NF4) | ~1.0–1.2 GB |
| LoRA adapter (r=8) | ~5–15 MB |
| Activations (seq=512, batch=1) | ~1.5–2.0 GB |
| Gradients (LoRA params only) | ~10–30 MB |
| Optimizer (AdamW 8-bit paged) | ~20–60 MB |
| KV cache (post-update inference) | ~0.5–1.0 GB |
| CUDA overhead + safety margin | ~1.0 GB |
| **Total** | **~4.5–6.0 GB** |

Fits comfortably on a single T4 with ~10 GB headroom.

---

## Consent Flow for Training Events

The existing 5-layer consent model already includes `training_signal`:

```
ConsentGate:
  transcript:      granted/denied   ← can we store the text?
  felt_state:      granted/denied   ← can we store emotional signal?
  training_signal: granted/denied   ← can we update model weights?
  retention:       granted/denied   ← can we retain data after training?
```

If `training_signal = denied`, the session contributes to C(t) as an inference-time correction (the model *reasons* about the grounding signal) but does NOT trigger a LoRA update (the model's *weights* are unchanged).

### Revocability

Weight updates are not cleanly revocable — gradient updates are additive and entangled. Mitigation: maintain a versioned adapter checkpoint per session. The `adapter_hash_before` in the training receipt identifies which checkpoint to restore if consent is revoked post-training.

---

## Privacy Architecture

The incremental grounding system provides the strongest possible privacy guarantees for personalized AI:

1. **No data leaves the device** — training happens locally
2. **No centralized training corpus** — one user, one model
3. **No gradient aggregation** — unlike federated learning, no gradients are shared
4. **Adapter is user-specific** — the ~50–100 MB adapter file is meaningless without the base model + training history
5. **Adapter can be encrypted at rest** — standard filesystem encryption

---

## Function-Calling Integration

Tool #7 in the governance pipeline:

```python
# Schema
{
    "name": "run_grounding_update",
    "parameters": {
        "session_id": str,
        "session_messages": list,
        "consent": dict,
        "mode": "dry_run" | "live",
        "session_receipt_root": str  # optional
    }
}

# Response (dry_run)
{
    "training_executed": false,
    "consent_valid": true,
    "sft_pair_count": 3,
    "token_count": 142,
    "loss_before": null,     # not simulated — genuinely null
    "loss_after": null,
    "steps_executed": 0,
    "training_receipt": { ... }
}
```

Registered in `tools/haic_tools.py` alongside the 4 core governance tools. The governance agent can invoke it as the final step after a successful interview + receipt generation.
