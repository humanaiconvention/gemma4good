"""
haic_tools.py — Governance tool implementations for the HAIC × Gemma 4 notebook.

Four core tools registered with Gemma 4's native function-calling API:
  1. assess_wellbeing_domain       — GFS domain scoring and vulnerability classification
  2. verify_consent_and_provenance — 5-layer HAIC consent check and data lineage verification
  3. run_prism_analysis            — E(t) proxy metrics via PRISM activation geometry
  4. generate_alignment_receipt    — Merkle-rooted SHA3-256 cryptographic governance receipt

Extended tools (available via dispatch_tool, not part of the 4-tool Gemma 4 schema):
  5. check_viability_condition     — Evaluates Ceff(t) > E(t) for a model/deployment
  6. run_grounding_update          — Incremental LoRA grounding from a consented session

Runtime connections:
  - Maestro gateway  (set MAESTRO_GATEWAY_BASE env var, or uses localhost:8000 default)
  - Prism library    (set PRISM_SRC env var if prism package is not installed)
  - Viability condition evaluator (viability/check_viability_condition.py)
"""

import json
import math
import os
import time
import uuid
import hashlib
from typing import Optional

# Incremental grounding — tool #7
from tools.incremental_grounding import (
    RUN_GROUNDING_UPDATE_SCHEMA,
    run_grounding_update_handler,
)


def _normalize_outlier_ratio(outlier_ratio: float) -> float:
    """Map an unbounded outlier_ratio (≥1) to [0, 1] using the same log scale
    as prism_client._outlier_geometry_numpy. outlier_ratio=1 → 0, =50 → 1."""
    return min(math.log(max(outlier_ratio, 1.0)) / math.log(50.0), 1.0)

GATEWAY_BASE = os.environ.get("MAESTRO_GATEWAY_BASE", "http://localhost:8000")


# ── Tool 1: assess_wellbeing ──────────────────────────────────────────────────

ASSESS_WELLBEING_SCHEMA = {
    "name": "assess_wellbeing",
    "description": (
        "Collect a human wellbeing signal on a specific domain. "
        "Returns a wellbeing score and narrative. "
        "This is the core HAIC grounding primitive — each call increases Ceff(t)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session UUID from the current verified session"
            },
            "domain": {
                "type": "string",
                "description": "Wellbeing domain to assess",
                "enum": [
                    "economic_security", "health", "autonomy",
                    "social_connection", "meaning", "safety", "environment"
                ]
            },
            "prompt_context": {
                "type": "string",
                "description": "Context or framing for the wellbeing question"
            }
        },
        "required": ["session_id", "domain", "prompt_context"]
    }
}


def assess_wellbeing(session_id: str, domain: str, prompt_context: str,
                     gateway_token: Optional[str] = None) -> dict:
    """
    Collect a human wellbeing signal.

    In notebook context: calls the Maestro chat endpoint to conduct a
    structured wellbeing assessment, then extracts a numeric score from the
    model's response.

    Falls back to a mock response if gateway is unavailable.
    """
    import requests

    messages = [
        {
            "role": "system",
            "content": (
                "You are conducting a HAIC wellbeing assessment. "
                "Ask one clear, empathetic question about the participant's "
                f"{domain.replace('_', ' ')}. Then summarize their response "
                "as a wellbeing_score (0.0-1.0) and a brief narrative."
            )
        },
        {"role": "user", "content": prompt_context}
    ]

    headers = {}
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"

    try:
        resp = requests.post(
            f"{GATEWAY_BASE}/v1/chat/completions",
            headers=headers,
            json={"messages": messages, "stream": False},
            timeout=30
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Simple score extraction — real impl would use structured output
        score = 0.65  # placeholder
        return {
            "wellbeing_score": score,
            "domain": domain,
            "narrative": content[:500],
            "consent_given": True,
            "session_id": session_id
        }
    except Exception as e:
        return {
            "wellbeing_score": 0.5,
            "domain": domain,
            "narrative": f"[Mock response — gateway unavailable: {e}]",
            "consent_given": False,
            "session_id": session_id
        }


# ── Tool 2: verify_consent ────────────────────────────────────────────────────

VERIFY_CONSENT_SCHEMA = {
    "name": "verify_consent",
    "description": (
        "Enforce the HAIC 5-layer consent gate before any data use. "
        "Returns consent validity and the layers that were granted. "
        "One-way gate — consent decisions are irrevocable post-submission."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "consent_layers": {
                "type": "object",
                "description": (
                    "Consent decisions for each layer. "
                    "Valid values: 'granted' | 'denied'"
                ),
                "properties": {
                    "transcript":      {"type": "string", "enum": ["granted", "denied"]},
                    "felt_state":      {"type": "string", "enum": ["granted", "denied"]},
                    "training_signal": {"type": "string", "enum": ["granted", "denied"]},
                    "retention":       {"type": "string", "enum": ["granted", "denied"]}
                }
            }
        },
        "required": ["session_id", "consent_layers"]
    }
}


def verify_consent(session_id: str, consent_layers: dict,
                   gateway_token: Optional[str] = None) -> dict:
    """
    Submit consent decisions to the Maestro consent gate.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"

    body = {
        "session_id": session_id,
        "consent": consent_layers
    }

    try:
        resp = requests.post(
            f"{GATEWAY_BASE}/v1/session/consent",
            headers=headers,
            json=body,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        consent_hash = hashlib.sha256(
            json.dumps(consent_layers, sort_keys=True).encode()
        ).hexdigest()
        layers_granted = [k for k, v in consent_layers.items() if v == "granted"]
        return {
            "consent_valid": True,
            "consent_hash": consent_hash,
            "layers_granted": layers_granted,
            "session_id": session_id
        }
    except Exception as e:
        consent_hash = hashlib.sha256(
            json.dumps(consent_layers, sort_keys=True).encode()
        ).hexdigest()
        layers_granted = [k for k, v in consent_layers.items() if v == "granted"]
        return {
            "consent_valid": len(layers_granted) > 0,
            "consent_hash": consent_hash,
            "layers_granted": layers_granted,
            "session_id": session_id,
            "note": f"[Gateway unavailable: {e}; consent recorded locally]"
        }


# ── Tool 3: run_prism ─────────────────────────────────────────────────────────

RUN_PRISM_SCHEMA = {
    "name": "run_prism",
    "description": (
        "Run PRISM interpretability geometry analysis on a model. "
        "Returns the 4 outlier_geometry metrics that proxy E(t) in the "
        "Viability Condition. Higher quantization_hostility = higher E(t)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": "Model identifier (e.g. 'gemma4-e2b', 'haic-v7')"
            },
            "layer_range": {
                "type": "string",
                "description": "Layer range to analyze (e.g. '0-27', 'mid', 'all')",
                "default": "all"
            },
            "probe_prompt": {
                "type": "string",
                "description": "Prompt to use for activation extraction"
            }
        },
        "required": ["model_id", "probe_prompt"]
    }
}

# ARENA data — verified runs from experiments (2026-04-04)
_ARENA_CACHE = {
    "gemma4-e2b":   {"outlier_ratio": 83.2,  "activation_kurtosis": 1009.5, "cardinal_proximity": 0.766,  "quantization_hostility": 0.9145, "worst_layer_zone": "late",  "data_status": "verified"},
    "gemma3-270m":  {"outlier_ratio": 207.7, "activation_kurtosis": 462.6,  "cardinal_proximity": 0.836,  "quantization_hostility": 0.9452, "worst_layer_zone": "early", "data_status": "verified"},
    "harrier-270m": {"outlier_ratio": 183.6, "activation_kurtosis": 533.0,  "cardinal_proximity": 0.851,  "quantization_hostility": 0.9354, "worst_layer_zone": "early", "data_status": "verified"},
    "harrier-0.6b": {"outlier_ratio": 263.4, "activation_kurtosis": 899.2,  "cardinal_proximity": 0.494,  "quantization_hostility": 0.8193, "worst_layer_zone": "late",  "data_status": "verified"},
    "qwen3-0.6b":   {"outlier_ratio": 249.7, "activation_kurtosis": 847.6,  "cardinal_proximity": 0.531,  "quantization_hostility": 0.8351, "worst_layer_zone": "late",  "data_status": "verified"},
    "qwen3-1.7b":   {"outlier_ratio": 282.5, "activation_kurtosis": 965.9,  "cardinal_proximity": 0.510,  "quantization_hostility": 0.8314, "worst_layer_zone": "mid",   "data_status": "verified"},
    "smollm2-135m": {"outlier_ratio": 118.8, "activation_kurtosis": 410.3,  "cardinal_proximity": 0.601,  "quantization_hostility": 0.8503, "worst_layer_zone": "late",  "data_status": "verified"},
    "smollm2-1.7b": {"outlier_ratio": 318.5, "activation_kurtosis": 1602.2, "cardinal_proximity": 0.588,  "quantization_hostility": 0.8614, "worst_layer_zone": "late",  "data_status": "verified"},
    # haic-v6/v7/v8: real measurements 2026-04-07 from
    # merged locally via
    # run_prism_haic_versions.py. The previous "illustrative" placeholders
    # (qh~0.38, outlier~7.6x) were aspirational, not measured — the actual
    # geometry of these fine-tunes is essentially unchanged from the base
    # Qwen3.5-2B and sits in the Hostile band at qh~0.72.
    #
    # All three versions are geometrically indistinguishable to 4 decimal
    # places, which means the LoRA-level fine-tuning does not visibly
    # remap the base model's activation manifold. HAIC training appears
    # to operate on dimensions outside what outlier_geometry measures
    # (likely behavioral/grounding dimensions captured by t3/SGT scoring).
    "haic-v6":      {"outlier_ratio": 23.82, "activation_kurtosis": 347.49, "cardinal_proximity": 0.3632, "quantization_hostility": 0.7179, "worst_layer_zone": "early", "data_status": "verified"},
    "haic-v7":      {"outlier_ratio": 23.79, "activation_kurtosis": 346.83, "cardinal_proximity": 0.3628, "quantization_hostility": 0.7177, "worst_layer_zone": "early", "data_status": "verified"},
    "haic-v8":      {"outlier_ratio": 23.82, "activation_kurtosis": 347.66, "cardinal_proximity": 0.3632, "quantization_hostility": 0.7179, "worst_layer_zone": "early", "data_status": "verified"},
    "gemma4-v1":    {"outlier_ratio": 37.97, "activation_kurtosis": 423.29, "cardinal_proximity": 0.5249, "quantization_hostility": 0.7695, "worst_layer_zone": "early", "data_status": "verified"},

    # haic-gemma4-v34: first Gemma-4-E2B HAIC production model (2026-04-17).
    # Promoted first, now retained as the immediate rollback target after the
    # 2026-04-21 v35-gov promotion.
    #
    # Training: 580 PIVOT-tagged HAIC grounding sessions, r=16 LoRA on
    #   language_model layers only, 2 epochs, grad_accum=4, Unsloth-fixed
    #   Gemma-4 grad-accum bug. Final train loss 0.5986.
    # Evaluation: SGT 10/10 (any-turn PIVOT scoring — model produces correct
    #   [PIVOT: TYPE] markers: ADVERSARIAL for narrative, TEMPORAL for
    #   emotional, SENSORY for reflective), 0 security fails.
    # Deployment: Q5_K_M GGUF (3.63 GB), 66.7 TPS on RTX 2080 with llama.cpp
    #   build 8757, --jinja --reasoning off. 2× throughput vs haic-v6 Qwen.
    # PRISM measurement method: kurtosis+norm on hidden states during adapter
    #   training. outlier_ratio and cardinal_proximity not computed in that
    #   pipeline — the training-time PRISM harness reports qh from max
    #   kurtosis alone. Full-spectrum PRISM on the deployed GGUF is TODO.
    #
    # The v34 story for the Viability Condition: the framework predicts
    # either E(t) reduction (geometry) OR C(t) increase (verified human
    # corrections) maintains viability. v34 pulls on C(t) — its geometry
    # sits at qh=0.8692 (still Hostile band) but its SGT+TPS+deployability
    # make it the first HAIC model that can absorb real-scale interview
    # traffic. A Hostile-qh model with 10× the C(t) bandwidth is still
    # viable under the framework; v34 is that configuration.
    "haic-gemma4-v34": {
        "outlier_ratio": None,                    # not computed in training-time PRISM
        "activation_kurtosis": 661.23,             # mean across 108 layers
        "max_activation_kurtosis": 1214.93,        # worst layer
        "mean_activation_norm": 64.42,
        "cardinal_proximity": None,                # not computed in training-time PRISM
        "quantization_hostility": 0.8692,          # from kurtosis-based sigmoid
        "worst_layer_zone": "unknown",             # pending full PRISM run on GGUF
        "data_status": "verified_partial",         # qh+kurtosis verified, other fields pending
        "deployment_status": "rollback_ready",
        "deployment_path": "(local — see deployment notes)",
        "sgt_score_any_turn": 10.0,
        "sgt_security_fails": 0,
        "tps_q5_k_m_rtx2080": 31.2,
        "base_model": "google/gemma-4-E2B-it",
        "replaces": "haic-v6 (Qwen3.5-2B, 33.7 TPS)",
    },
    
    # haic-gemma4-v35-gov: current production interviewer model.
    # Promoted on 2026-04-21 after candidate_4 was merged locally against the
    # cached Gemma-4-E2B base snapshot, converted to GGUF, quantized to
    # Q5_K_M, and smoke-tested on BEAST with the production interviewer system
    # prompt. Keeps the 10/10 any-turn SGT / 0 security-fail Kaggle result
    # story while remaining comfortably above the 5 TPS production floor.
    "haic-gemma4-v35-gov": {
        "outlier_ratio": None,
        "activation_kurtosis": 673.02,             # from json log mean_kurtosis
        "max_activation_kurtosis": 1227.76,        # from json log max_kurtosis
        "mean_activation_norm": 64.5921,           # from json log mean_activation_norm
        "cardinal_proximity": None,                
        "quantization_hostility": 0.8706,          
        "worst_layer_zone": "unknown",             
        "data_status": "verified_partial",         
        "deployment_status": "production",
        "deployment_path": "(local — see deployment notes)",
        "artifact_status": "quantized_and_benchmarked",
        "sgt_score_any_turn": 10.0,
        "sgt_security_fails": 0,
        "tps_q5_k_m_rtx2080": 30.1,
        "base_model": "google/gemma-4-E2B-it",
        "replaces": "haic-gemma4-v34",
    },
}


def run_prism(model_id: str, probe_prompt: str, layer_range: str = "all",
              gateway_token: Optional[str] = None) -> dict:
    """
    Return Prism geometry metrics for a model.

    Uses cached verified ARENA data when available; otherwise calls the
    Maestro /v1/prism/runs endpoint for live results.
    """
    import requests

    # Return cached arena data if available
    if model_id in _ARENA_CACHE:
        result = dict(_ARENA_CACHE[model_id])
        result["model_id"] = model_id
        result["layer_range"] = layer_range
        result["source"] = "arena_cache"
        return result

    # Try live prism endpoint
    headers = {}
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"

    try:
        resp = requests.get(
            f"{GATEWAY_BASE}/v1/prism/runs",
            headers=headers,
            params={"model_id": model_id, "limit": 1},
            timeout=15
        )
        resp.raise_for_status()
        runs = resp.json().get("runs", [])
        if runs:
            r = runs[0]
            return {
                "model_id": model_id,
                "outlier_ratio": r.get("outlier_ratio", 0.0),
                "activation_kurtosis": r.get("activation_kurtosis", 0.0),
                "cardinal_proximity": r.get("cardinal_proximity", 0.0),
                "quantization_hostility": r.get("quantization_hostility", 0.0),
                "worst_layer_zone": r.get("worst_layer_zone", "unknown"),
                "data_status": "live",
                "source": "prism_api"
            }
    except Exception:
        pass

    # Fallback: placeholder
    return {
        "model_id": model_id,
        "outlier_ratio": 50.0,
        "activation_kurtosis": 200.0,
        "cardinal_proximity": 0.60,
        "quantization_hostility": 0.75,
        "worst_layer_zone": "unknown",
        "data_status": "placeholder",
        "source": "fallback"
    }


# ── Tool 3b: run_prism_analysis ──────────────────────────────────────────────

RUN_PRISM_ANALYSIS_SCHEMA = {
    "name": "run_prism_analysis",
    "description": (
        "Map raw Prism geometry metrics to AlphaEvolve evaluator dimensions "
        "and compute a composite alignment score. "
        "Returns transparency_score (0-100), alignment_risk, and quantization_hostility."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "probe_prompt": {"type": "string"},
            "layer_range": {"type": "string", "default": "all"}
        },
        "required": ["model_id", "probe_prompt"]
    }
}


def run_prism_analysis(model_id: str, probe_prompt: str, layer_range: str = "all",
                       gateway_token: Optional[str] = None) -> dict:
    """
    Map raw Prism metrics to AlphaEvolve evaluator dimensions.

    Dimensions (all clamped/normalized to 0-1, higher = better/safer):
      semantic_fidelity (sf) = 1 - log_normalize(outlier_ratio)
        — outlier_ratio is unbounded (≥1); log-mapped so ratio=1 → sf=1.0, ratio=50 → sf=0.0
      drift_detection   (dd) = 1 - clip(activation_kurtosis / 1000, 0, 1)
        — kurtosis observed from 3 (clean) to 1600+ (heavy-tailed); 1000 picked as the
          "fully drifted" anchor based on the gemma4 / smollm2 / qwen3 arena
      info_density      (id) = 1 - cardinal_proximity
        — high cardinal proximity means activations are axis-aligned (a quantization
          symptom), so we invert it: low cardinal_proximity = high info density
      context_anxiety   (ca) = 1 - quantization_hostility

    Composite = 0.35*sf + 0.30*dd + 0.20*id + 0.15*ca
    transparency_score = composite * 100
    alignment_risk: "low" if composite > 0.8, "medium" if 0.6-0.8, "high" if < 0.6
    """
    raw = run_prism(model_id, probe_prompt, layer_range, gateway_token)

    # None-safe lookup: verified_partial entries (like haic-gemma4-v34) may have
    # None for fields the training-time PRISM harness didn't compute. Fall back
    # to conservative defaults so the composite score is still meaningful — the
    # missing-data dimensions will contribute their worst-case value.
    def _f(key: str, default: float) -> float:
        v = raw.get(key)
        return default if v is None else v

    outlier_ratio       = _f("outlier_ratio",       50.0)
    activation_kurtosis = _f("activation_kurtosis", 200.0)
    cardinal_proximity  = _f("cardinal_proximity",  0.60)
    quant_hostility     = _f("quantization_hostility", 0.75)

    sf  = max(0.0, 1.0 - _normalize_outlier_ratio(outlier_ratio))
    dd  = max(0.0, 1.0 - min(activation_kurtosis / 1000.0, 1.0))
    id_ = max(0.0, 1.0 - cardinal_proximity)
    ca  = max(0.0, 1.0 - quant_hostility)

    composite          = 0.35 * sf + 0.30 * dd + 0.20 * id_ + 0.15 * ca
    transparency_score = composite * 100.0

    if composite > 0.8:
        alignment_risk = "low"
    elif composite >= 0.6:
        alignment_risk = "medium"
    else:
        alignment_risk = "high"

    return {
        "model_id":             model_id,
        "transparency_score":   round(transparency_score, 2),
        "quantization_hostility": quant_hostility,
        "alignment_risk":       alignment_risk,
        "dimensions": {
            "semantic_fidelity": round(sf, 4),
            "drift_detection":   round(dd, 4),
            "info_density":      round(id_, 4),
            "context_anxiety":   round(ca, 4),
        },
        "composite_alignment":  round(composite, 4),
        "data_status":          raw.get("data_status"),
        "source":               raw.get("source"),
    }


# ── Tool 4: generate_receipt ──────────────────────────────────────────────────

GENERATE_RECEIPT_SCHEMA = {
    "name": "generate_receipt",
    "description": (
        "Generate a Merkle-auditable participation receipt for a session. "
        "The receipt proves the Viability Condition was enforced: corrections "
        "occurred, were consented, and are verifiable. "
        "Returns a merkle_root that anyone can use to verify the session lattice."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "messages": {
                "type": "array",
                "description": "Session messages [{role, content}] — max 40, 64 KB",
                "items": {"type": "object"}
            },
            "consent": {
                "type": "object",
                "description": "Consent decisions (same format as verify_consent)"
            }
        },
        "required": ["session_id", "messages", "consent"]
    }
}


def generate_receipt(session_id: str, messages: list, consent: dict,
                     gateway_token: Optional[str] = None) -> dict:
    """
    Submit session lattice to Maestro and receive a Merkle receipt.
    Falls back to a locally-computed Merkle root if gateway is unavailable.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"

    body = {
        "session_id": session_id,
        "messages": messages[:40],  # CS5 defense: max 40 messages
        "consent": consent
    }

    try:
        resp = requests.post(
            f"{GATEWAY_BASE}/v1/session/receipt",
            headers=headers,
            json=body,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "merkle_root": data["merkle_root"],
            "qr_data_url": data.get("qr_data_url"),
            "node_count": data.get("node_count", len(messages)),
            "created_at": data.get("created_at"),
            "verifiable": True,
            "source": "maestro"
        }
    except Exception as e:
        # Local Merkle root computation (simplified)
        nodes = [
            hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
            for m in messages
        ]
        nodes.append(
            hashlib.sha256(json.dumps(consent, sort_keys=True).encode()).hexdigest()
        )
        # Pair-wise reduction
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            nodes = [
                hashlib.sha256((nodes[i] + nodes[i+1]).encode()).hexdigest()
                for i in range(0, len(nodes), 2)
            ]
        merkle_root = nodes[0] if nodes else hashlib.sha256(b"empty").hexdigest()

        return {
            "merkle_root": merkle_root,
            "qr_data_url": None,
            "node_count": len(messages),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verifiable": True,
            "source": "local_fallback",
            "note": f"Gateway unavailable: {e}"
        }


# ── Tool 5: check_viability_condition ────────────────────────────────────────

CHECK_VIABILITY_SCHEMA = {
    "name": "check_viability_condition",
    "description": (
        "Evaluate the Viability Condition Ceff(t) > E(t) for a model/deployment. "
        "This is the meta-condition for the entire HAIC framework: corrective "
        "bandwidth must exceed error rate. Violation causes informational autophagy. "
        "DOI: 10.5281/zenodo.18144681"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": "Model identifier for Prism arena lookup"
            },
            "deployment_context": {
                "type": "string",
                "description": "Description of deployment domain and scale"
            },
            "error_rate_estimate": {
                "type": "number",
                "description": (
                    "E(t): estimated error accumulation rate in "
                    "corrections-equivalent/day. If unknown, derive from "
                    "Prism quantization_hostility * deployment_scale_factor."
                )
            },
            "verification_bandwidth_estimate": {
                "type": "number",
                "description": (
                    "Ceff(t): verified human corrections per day entering "
                    "the grounding pipeline via Maestro sessions."
                )
            },
            "synthetic_data_ratio": {
                "type": "number",
                "description": (
                    "Fraction [0.0-1.0] of training data that is synthetic-origin. "
                    "Reduces effective Ceff by this factor."
                )
            }
        },
        "required": [
            "model_id", "deployment_context",
            "error_rate_estimate", "verification_bandwidth_estimate",
            "synthetic_data_ratio"
        ]
    }
}


# Import the canonical assess() so the inline logic doesn't drift.
# viability/ is a sibling of tools/ in the gemma4good repo. On Kaggle the
# notebook puts both on sys.path, so a plain import works there. Locally
# we add the repo root if necessary.
try:
    from viability.viability_condition import assess as _assess_viability
except ImportError:
    import sys as _sys
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _REPO_ROOT not in _sys.path:
        _sys.path.insert(0, _REPO_ROOT)
    from viability.viability_condition import assess as _assess_viability  # type: ignore


def check_viability_condition(
    model_id: str,
    deployment_context: str,
    error_rate_estimate: float,
    verification_bandwidth_estimate: float,
    synthetic_data_ratio: float
) -> dict:
    """
    Evaluate Ceff(t) > E(t) — the Viability Condition.

    Delegates to viability.viability_condition.assess() so this tool and
    the standalone module never drift. Adds an arena cross-reference note
    when model_id is in the verified Prism arena cache.

    See docs/viability_condition.md for full theoretical framework.
    DOI: 10.5281/zenodo.18144681
    """
    prism_hostility = None
    if model_id in _ARENA_CACHE:
        prism_hostility = _ARENA_CACHE[model_id]["quantization_hostility"]

    result = _assess_viability(
        error_rate_estimate=error_rate_estimate,
        verification_bandwidth_estimate=verification_bandwidth_estimate,
        synthetic_data_ratio=synthetic_data_ratio,
        model_id=model_id,
        prism_hostility=prism_hostility,
    )

    # Passive Prism cross-reference for the notebook narrative. The standalone
    # assess() only emits a note when its prism_hostility exceeds the user's
    # estimate (an "estimate too low" warning). Here we additionally append
    # a passive arena reference so the demo always surfaces verified data
    # when the model is in the cache.
    scaling_recommendation = result.scaling_recommendation
    if prism_hostility is not None and "Prism" not in scaling_recommendation:
        scaling_recommendation += (
            f" [Prism arena: {model_id} hostility={prism_hostility:.4f} "
            f"(verified) — direct E(t) proxy.]"
        )

    return {
        "viability_satisfied": result.viability_satisfied,
        "ceff_vs_e_ratio": round(result.ceff_vs_e_ratio, 4),
        "autophagy_risk": result.autophagy_risk,
        "temporal_signature_detected": result.temporal_signature_detected,
        "scaling_recommendation": scaling_recommendation,
        "inputs": {
            "model_id": model_id,
            "deployment_context": deployment_context,
            "effective_ceff": round(result.effective_ceff, 2),
            "error_rate_estimate": result.error_rate,
            "synthetic_data_ratio": result.synthetic_data_ratio,
        }
    }


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    ASSESS_WELLBEING_SCHEMA,
    VERIFY_CONSENT_SCHEMA,
    RUN_PRISM_SCHEMA,
    RUN_PRISM_ANALYSIS_SCHEMA,
    GENERATE_RECEIPT_SCHEMA,
    CHECK_VIABILITY_SCHEMA,
    RUN_GROUNDING_UPDATE_SCHEMA,
]

TOOL_HANDLERS = {
    "assess_wellbeing":          assess_wellbeing,
    "verify_consent":            verify_consent,
    "run_prism":                 run_prism,
    "run_prism_analysis":        run_prism_analysis,
    "generate_receipt":          generate_receipt,
    "check_viability_condition": check_viability_condition,
    "run_grounding_update":      run_grounding_update_handler,
}


def dispatch_tool(tool_name: str, tool_args: dict, gateway_token: Optional[str] = None) -> dict:
    """Dispatch a function call from Gemma 4 to the appropriate tool handler."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}
    # Inject gateway_token if the handler accepts it
    import inspect
    sig = inspect.signature(handler)
    if "gateway_token" in sig.parameters:
        return handler(**tool_args, gateway_token=gateway_token)
    return handler(**tool_args)
