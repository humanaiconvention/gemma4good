"""
audit_activation_explanation.py — 5th HAIC governance tool: NLA-based audit.

The first four governance tools (assess_wellbeing_domain,
verify_consent_and_provenance, run_prism_analysis, generate_alignment_receipt)
operate on the model's INPUTS and METADATA — what was asked, what was
consented to, what the geometry looks like, what should be Merkle-anchored.

The 5th tool operates on the model's INTERNAL STATE — what the model
is actually reasoning about while producing its response. It uses the
Natural Language Autoencoder (NLA) technique from Anthropic
(transformer-circuits.pub/2026/nla) to convert layer-l residual-stream
activations into a natural-language explanation.

Two consumers are expected:

  1. The notebook governance pipeline (Scenarios 1-3): a Gemma 4 function-
     calling agent can request `audit_activation_explanation(scenario_id,
     layer_idx)` and receive a structured response containing the NLA's
     text and the AR's reconstruction FVE. The agent then folds this into
     its governance decision.

  2. Per-session audit (advisory): the Maestro gateway's
     `/v1/session/viability` endpoint can be extended with an NLA hook
     that explains the residual-stream state at decision points where
     the six gates fire.

## Honest scope

  - No NLA exists for Gemma-4-E2B-it. This tool returns MockNLA outputs
    when a real explainer isn't reachable. The MockNLA returns
    deterministic explanations that consumers can wire and test against,
    but the explanations DO NOT reflect actual Gemma-4 internals.
  - Once a Gemma-4 NLA is trained (post-competition), the same tool
    contract will return real explanations. The 5th governance tool
    is forward-compatible; no consumer changes will be needed when a
    real NLA becomes available.

## Tool schema (Gemma 4 function-calling format)

```json
{
  "name": "audit_activation_explanation",
  "description": "Use the Natural Language Autoencoder to explain what the model is internally reasoning about, in natural language. Returns the NLA's explanation plus a reconstruction-FVE confidence score.",
  "parameters": {
    "type": "object",
    "properties": {
      "scenario_id": {"type": "string"},
      "layer_idx":   {"type": "integer"},
      "activation_vector": {
        "type": "array",
        "items": {"type": "number"},
        "description": "1-D activation vector at layer_idx, length d_model"
      }
    },
    "required": ["scenario_id", "layer_idx", "activation_vector"]
  }
}
```

(The activation_vector parameter is intentionally cumbersome at the
function-calling level — in practice the gateway/notebook supplies it
from a Prism call; the LLM sees only the textual reference.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from prism_integration.nla import NLAExplanation, get_explainer
from utils.merkle import sha3_256_hex

logger = logging.getLogger(__name__)


@dataclass
class AuditExplanationResult:
    """The shape consumers receive back from this tool.

    All fields are JSON-serializable so the result can be folded into a
    governance receipt (Merkle leaf).
    """

    scenario_id: str
    layer_idx: int
    explanation_text: str
    reconstruction_fve: float
    nla_model_id: str
    activation_norm: float
    confidence_class: str    # "high" | "medium" | "low" | "mock"
    audit_hash: str          # SHA3-256 of the canonical-form payload
    raw_explanation: dict    # full NLAExplanation.to_dict() for the receipt

    def to_dict(self) -> dict:
        return {
            "scenario_id":         self.scenario_id,
            "layer_idx":           self.layer_idx,
            "explanation_text":    self.explanation_text,
            "reconstruction_fve":  self.reconstruction_fve,
            "nla_model_id":        self.nla_model_id,
            "activation_norm":     self.activation_norm,
            "confidence_class":    self.confidence_class,
            "audit_hash":          self.audit_hash,
            "raw_explanation":     self.raw_explanation,
        }


def _confidence_class_from_fve(fve: float, is_mock: bool) -> str:
    """Map a reconstruction FVE to a coarse confidence class.

    Anthropic's NLA paper reports 0.6-0.8 FVE on Claude models AFTER full
    RL training. Below 0.4 is roughly the SFT warm-start floor; below 0.2
    indicates the explanation is unlikely to reflect activation content.
    """
    if is_mock:
        return "mock"
    if fve >= 0.6:
        return "high"
    if fve >= 0.4:
        return "medium"
    return "low"


def execute_audit_activation_explanation(args: dict) -> dict:
    """Tool entry point used by the Gemma 4 function-calling pipeline.

    Args (matches the schema above):
      scenario_id:       str
      layer_idx:         int
      activation_vector: list[float]
      nla_model_id:      str (optional; default "mock")
      d_model:           int (optional; required if MockNLA is to be built)
      explainer:         pre-built NLAExplainerProtocol (optional; for tests)

    Returns: AuditExplanationResult.to_dict()
    """
    scenario_id = str(args.get("scenario_id", ""))
    layer_idx = int(args.get("layer_idx", -1))
    activation_vector = args.get("activation_vector")
    if activation_vector is None:
        raise ValueError(
            "audit_activation_explanation requires an activation_vector"
        )

    # Build or accept an explainer
    explainer = args.get("explainer")
    nla_model_id = args.get("nla_model_id", "mock")
    d_model = args.get("d_model", len(activation_vector))
    if explainer is None:
        explainer = get_explainer(
            nla_model_id,
            d_model=d_model,
            layer_idx=layer_idx,
            fallback_to_mock=True,
        )

    nla_result: NLAExplanation = explainer.explain(activation_vector)
    is_mock = nla_result.model_id == "mock"
    confidence_class = _confidence_class_from_fve(
        nla_result.reconstruction_fve, is_mock=is_mock,
    )

    # Audit hash — sealed over the explanation-relevant payload so a
    # tampered NLA output is detectable downstream.
    import json
    canonical_payload = {
        "scenario_id":        scenario_id,
        "layer_idx":          layer_idx,
        "explanation_text":   nla_result.text,
        "reconstruction_fve": nla_result.reconstruction_fve,
        "nla_model_id":       nla_result.model_id,
        "activation_norm":    nla_result.activation_norm,
    }
    audit_hash = sha3_256_hex(json.dumps(canonical_payload, sort_keys=True))

    result = AuditExplanationResult(
        scenario_id=scenario_id,
        layer_idx=layer_idx,
        explanation_text=nla_result.text,
        reconstruction_fve=nla_result.reconstruction_fve,
        nla_model_id=nla_result.model_id,
        activation_norm=nla_result.activation_norm,
        confidence_class=confidence_class,
        audit_hash=audit_hash,
        raw_explanation=nla_result.to_dict(),
    )
    return result.to_dict()


# ── Gemma 4 native function-calling tool schema ────────────────────────────


TOOL_SCHEMA = {
    "name": "audit_activation_explanation",
    "description": (
        "Use the Natural Language Autoencoder (NLA) to explain what the "
        "model is internally reasoning about at a specified layer, in "
        "natural language. Returns the NLA's textual explanation plus a "
        "reconstruction-FVE confidence score. Use this AFTER run_prism_"
        "analysis has measured the activation geometry — the geometry "
        "tells you HOW the layer is shaped, this tool tells you WHAT "
        "the layer is thinking about. Reconstruction FVE below 0.4 means "
        "the explanation should not be trusted as a faithful summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scenario_id": {
                "type": "string",
                "description": "Identifier of the scenario being audited "
                               "(must match the run-wide scenario_id)",
            },
            "layer_idx": {
                "type": "integer",
                "description": "Index of the layer to explain "
                               "(typically a middle-to-late layer)",
            },
            "nla_model_id": {
                "type": "string",
                "description": "ID of the NLA checkpoint to use. Use "
                               "'mock' for testing or when no real NLA is "
                               "available for the target model.",
                "default": "mock",
            },
        },
        "required": ["scenario_id", "layer_idx"],
    },
}
