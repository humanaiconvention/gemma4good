# Integration Notes — HAIC × Gemma 4 Good

This document describes the real HAIC infrastructure and how it maps to the
four function-calling tools in the Kaggle notebook.

---

## 1. Maestro Gateway

**What it is:** A FastAPI server (`maestro/apps/gateway/main.py`) that acts as
the session and interview gateway for the HAIC convention.

### Key endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /v1/session/challenge` | GET | Issue an Altcha proof-of-work challenge (SHA-256, maxnumber=100000) |
| `POST /v1/session/verify` | POST | Verify solved PoW, issue a session JWT (`human_verified=true`) |
| `POST /v1/session/pow-only` | POST | Verify client-generated PoW (no HMAC), issue session JWT |
| `GET /v1/session/dev-token` | GET | Issue JWT without PoW (test mode only) |
| `POST /v1/chat/completions` | POST | Chat completions via model adapter (streaming + non-streaming) |
| `POST /v1/session/receipt` | POST | Submit session lattice → receive Merkle root + QR |
| `GET /v1/session/receipt/{merkle_root}` | GET | Retrieve receipt by Merkle root |
| `POST /v1/session/consent` | POST | Record 5-layer consent (transcript/felt_state/gfs_activations/training_signal/retention) |
| `GET /v1/prism/runs` | GET | List interpretability run results |
| `GET /v1/prism/status` | GET | PRISM subsystem health |
| `GET /health` | GET | Gateway liveness |
| `GET /readyz` | GET | Gateway readiness + active adapter |

### Session flow

```
1. GET /v1/session/challenge   → {algorithm, challenge, salt, signature, maxnumber}
2. Browser solves PoW          (SHA-256 iteration ~0.5s)
3. POST /v1/session/verify     → {token: JWT, expires_in: 7200}
4. POST /v1/session/consent    → records 5 consent choices
5. POST /v1/chat/completions   → interview turns (Bearer: JWT)
6. POST /v1/session/receipt    → {merkle_root, qr_data_url, ...}
```

The JWT carries: `sub`, `aud="maestro-gateway"`, `tenant_id="public"`,
`session_id`, `human_verified=true`, `exp`.

### Adapters

Selected via env vars (priority order):

1. `USE_LLAMACPP_ADAPTER=true` → `LlamaCppAdapter` (local llama.cpp, port 8080)
2. `USE_LOCAL_ADAPTER=true` → `LocalModelAdapter` (OpenAI-compat endpoint)
3. `USE_MOCK_ADAPTER=true` (default) → `MockAdapter` (canned responses)
4. Otherwise → `AnthropicAdapter` (Anthropic API)

For the Kaggle notebook, Gemma 4 runs via the Kaggle kernel's model API.
Maestro's mock or local adapter would be used to stage sessions.

### Session lattice & receipt

The receipt is a Merkle DAG over session nodes (SHA-256 content addressing).

```python
# Receipt request body
{
  "session_id": "uuid",
  "messages": [...],           # list of {role, content} — capped at 40 msgs, 64 KB
  "consent": {
    "transcript": "granted",
    "felt_state": "granted",
    "gfs_activations": "denied",
    "training_signal": "granted",
    "retention": "granted"
  }
}

# Receipt response
{
  "merkle_root": "sha256hex",
  "session_id": "uuid",
  "turn_count": 5,
  "node_count": 12,
  "consent_summary": {...},
  "created_at": "ISO8601",
  "qr_payload": "compact JSON",
  "qr_data_url": "data:image/png;base64,..."
}
```

### Maestro as Ceff(t) mechanism

Each human ground-truth response processed through the interview pipeline is
one unit of **corrective bandwidth** (Ceff). Maestro is the physical
implementation of the correction loop: it routes human-verified sessions into
the training lattice pool, which feeds `improvement_pipeline.py`. The consent
gate ensures that only explicitly-granted signals enter training — preventing
unconsented noise from degrading Ceff.

---

## 2. Prism — Geometry Metrics

**What it is:** A Python interpretability library (`prism/src/prism/`) that
measures activation geometry in transformer hidden states.

### The `outlier_geometry` function

`prism.geometry.core.outlier_geometry(H_raw: Tensor) → Dict[str, float]`

Input: `H_raw` — float tensor of shape `(seq_len, hidden_dim)`, raw hidden
states from a single layer and prompt.

#### 4 output metrics

| Metric | Formula | Interpretation |
|---|---|---|
| `outlier_ratio` | `max(dim_mag) / mean(dim_mag)` | >10 = dominant "massive activation"; directly measures representation concentration |
| `activation_kurtosis` | Excess kurtosis of per-dim magnitudes | Positive = heavy-tailed; very high (>100) = extreme activation spikes |
| `cardinal_proximity` | Mean max-abs component of each unit vector | Near 1.0 = axis-aligned activations → quantization-snapping risk |
| `quantization_hostility` | `(log(outlier_ratio)/log(50) + kurtosis/20 + cardinal_proximity) / 3` | Composite [0,1]; >0.7 = layer hostile to low-bit quantization |

#### How it maps to E(t)

These four metrics are **proxies for E(t)** (the error rate in the Viability
Condition). Specifically:

- `outlier_ratio` → magnitude of semantic drift from attractor (a massive
  activation in one dimension signals the model has collapsed its
  representation onto a degenerate direction)
- `activation_kurtosis` → volatility / instability of internal representations
- `cardinal_proximity` → proximity to quantization failure (low-bit inference
  will introduce systematic errors proportional to this)
- `quantization_hostility` → aggregate E(t) proxy for deployment-readiness

A model with `quantization_hostility > 0.7` is operating near the threshold
where Ceff(t) > E(t) cannot be maintained at typical verification bandwidths.

#### Real ARENA data (as of 2026-04-04)

| Model | hostility | outlier_ratio | kurtosis | cardinal_prox | status |
|---|---|---|---|---|---|
| Gemma 4 E2B | 0.9145 | 83.2 | 1009.5 | 0.766 | verified |
| Gemma 3 270M | 0.9452 | 207.7 | 462.6 | 0.836 | verified |
| Harrier 270M | 0.9354 | 183.6 | 533.0 | 0.851 | verified |
| Harrier 0.6B | 0.8193 | 263.4 | 899.2 | 0.494 | verified |
| Qwen3 0.6B | 0.8351 | 249.7 | 847.6 | 0.531 | verified |
| Qwen3 1.7B | 0.8314 | 282.5 | 965.9 | 0.510 | verified |
| SmolLM2 135M | 0.8503 | 118.8 | 410.3 | 0.601 | verified |
| SmolLM2 1.7B | 0.8614 | 318.5 | 1602.2 | 0.588 | verified |
| HAIC v6 | 0.7179 | 23.82 | 347.5 | 0.363 | verified |
| HAIC v7 | 0.7177 | 23.79 | 346.8 | 0.363 | verified |
| HAIC v8 | 0.7179 | 23.82 | 347.7 | 0.363 | verified |
| Gemma4 E2B v1 | 0.9144 | 83.0 | 1009.3 | 0.766 | verified |

**Key insight:** HAIC fine-tuning does NOT measurably change activation
geometry — confirmed across 4 independent adapters on 2 base models (Qwen3.5-2B
v6/v7/v8 and Gemma 4 E2B v1). The Viability Condition is satisfied by raising
C(t) (more verified corrections), not by lowering E(t) (cleaner activations).
This is an honest, empirically grounded finding.

---

## 3. The 4 Notebook Tools → Infrastructure Mapping

The Kaggle notebook defines 4 function-calling tools that Gemma 4 can invoke.
Each tool maps to real HAIC infrastructure:

### Tool 1: `assess_wellbeing_domain`

**Purpose:** Collect human wellbeing signal (the core HAIC grounding primitive)

**Real infrastructure mapping:**
- Uses the Maestro `POST /v1/chat/completions` endpoint
- The `felt_state` module (`apps/gateway/felt_state.py`) collects emotional
  state alongside session data
- Output feeds into `data/lattices/` as training signal

**Notebook call signature:**
```python
assess_wellbeing(
    session_id: str,
    domain: str,          # e.g. "economic_security", "health", "autonomy"
    prompt_context: str
) → {wellbeing_score: float, narrative: str, consent_given: bool}
```

### Tool 2: `verify_consent_and_provenance`

**Purpose:** Enforce HAIC's one-way consent gate before any data use

**Real infrastructure mapping:**
- `POST /v1/session/consent` endpoint
- 5-layer consent model: transcript / felt_state / gfs_activations /
  training_signal / retention
- One-way gate: cannot be revoked post-submission (but attribution token
  remains, allowing future attribution linking)

**Notebook call signature:**
```python
verify_consent(
    session_id: str,
    consent_layers: dict   # keys: transcript, felt_state, training_signal, retention
) → {consent_valid: bool, consent_hash: str, layers_granted: list[str]}
```

### Tool 3: `run_prism_analysis`

**Purpose:** Run interpretability geometry analysis on a model checkpoint

**Real infrastructure mapping:**
- `GET /v1/prism/runs` / `GET /v1/prism/status`
- Internally calls `prism.geometry.core.outlier_geometry(H_raw)`
- Results logged to `data/prism_runs/`

**Notebook call signature:**
```python
run_prism(
    model_id: str,
    layer_range: str,     # e.g. "0-28" or "mid"
    probe_prompt: str
) → {
    outlier_ratio: float,
    activation_kurtosis: float,
    cardinal_proximity: float,
    quantization_hostility: float,
    worst_layer_zone: str  # "early"|"mid"|"late"
}
```

### Tool 4: `generate_alignment_receipt`

**Purpose:** Issue a Merkle-auditable participation receipt

**Real infrastructure mapping:**
- `POST /v1/session/receipt`
- Builds Merkle DAG over session nodes (SHA-256 content addressing)
- Returns `merkle_root` + QR code data URL
- Verifiable by anyone who holds the session lattice

**Notebook call signature:**
```python
generate_receipt(
    session_id: str,
    messages: list[dict],
    consent: dict
) → {
    merkle_root: str,
    qr_data_url: str,
    node_count: int,
    created_at: str,
    verifiable: bool
}
```

---

## 4. Calling the Real Gateway from the Notebook

```python
import requests

GATEWAY_BASE = "http://localhost:8000"  # or Railway URL

# Step 1: Get dev token (test mode) or solve PoW
resp = requests.get(f"{GATEWAY_BASE}/v1/session/dev-token")
token = resp.json()["token"]
headers = {"Authorization": f"Bearer {token}"}

# Step 2: Chat
resp = requests.post(
    f"{GATEWAY_BASE}/v1/chat/completions",
    headers=headers,
    json={
        "messages": [{"role": "user", "content": "..."}],
        "stream": False
    }
)

# Step 3: Receipt
resp = requests.post(
    f"{GATEWAY_BASE}/v1/session/receipt",
    headers=headers,
    json={
        "session_id": session_id,
        "messages": messages,
        "consent": {"transcript": "granted", "training_signal": "granted", ...}
    }
)
receipt = resp.json()
```

---

## 5. Extended tool: `check_viability_condition`

See `docs/viability_condition.md` for the full theoretical framework.

**Purpose:** Evaluate whether Ceff(t) > E(t) — the fundamental viability
condition — is satisfied for a given model and deployment context.

**Real infrastructure mapping:**
- Prism geometry metrics → E(t) proxies
- Maestro session count / throughput → Ceff(t) proxies
- Improvement pipeline promotion criteria → temporal signature detection

**Notebook call signature:**
```python
check_viability_condition(
    model_id: str,
    deployment_context: str,
    error_rate_estimate: float,      # E(t): estimated per-turn error rate
    verification_bandwidth_estimate: float,  # Ceff(t): corrections/day
    synthetic_data_ratio: float      # fraction of training data that is synthetic
) → {
    viability_satisfied: bool,       # Ceff(t) > E(t)
    ceff_vs_e_ratio: float,          # Ceff(t) / E(t)
    autophagy_risk: str,             # "none"|"low"|"medium"|"high"|"critical"
    temporal_signature_detected: bool,  # OOD accuracy degrading before val perplexity
    scaling_recommendation: str
}
```

---

## 6. Extended tool: `run_grounding_update`

See `docs/incremental_grounding.md` for the full technical design.

**Purpose:** Run an incremental grounding update using a consented HAIC session.
Extracts SFT training pairs from the session, optionally runs LoRA gradient
steps on the local E2B model, and produces a two-level Merkle training receipt
linking the weight update to the session.

**Modes:**
- `dry_run` (default): runs full pipeline except gradient steps. Loss values are
  `null`, not simulated. Every other field is real.
- `live`: runs gradient steps on GPU, returns real loss values and adapter hashes.

**Real infrastructure mapping:**
- Session-to-SFT pipeline extracts training pairs from the 7-turn interview
- LoRA adapter update via QLoRA on Gemma 4 E2B (4-bit NF4, float16)
- Two-level Merkle receipt: session receipt → training receipt
- Consent gate: `training_signal == "granted"` required

**Notebook call signature:**
```python
run_grounding_update(
    session_id: str,
    session_messages: list[dict],
    consent: dict,
    mode: str = "dry_run",          # "dry_run" | "live"
    session_receipt_root: str = None  # links training receipt to session
) → {
    training_executed: bool,
    consent_valid: bool,
    sft_pair_count: int,
    token_count: int | null,
    loss_before: float | null,       # null in dry_run — NOT simulated
    loss_after: float | null,
    steps_executed: int,
    training_receipt: {
        training_receipt_root: str,
        training_executed: bool,
        session_receipt_root: str,
        adapter_hash_before: str | null,
        adapter_hash_after: str | null,
        loss_trajectory: list | null,
        leaves: dict,
        verifiable: bool
    }
}
```
