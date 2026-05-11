"""Tests for tools/audit_activation_explanation.py — 5th governance tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism_integration.nla import mock_explainer
from tools.audit_activation_explanation import (
    TOOL_SCHEMA,
    AuditExplanationResult,
    _confidence_class_from_fve,
    execute_audit_activation_explanation,
)


# ── Confidence-class mapping ────────────────────────────────────────────────


def test_confidence_class_high_for_high_fve():
    assert _confidence_class_from_fve(0.75, is_mock=False) == "high"
    assert _confidence_class_from_fve(0.60, is_mock=False) == "high"


def test_confidence_class_medium_for_middle_fve():
    assert _confidence_class_from_fve(0.50, is_mock=False) == "medium"
    assert _confidence_class_from_fve(0.40, is_mock=False) == "medium"


def test_confidence_class_low_for_low_fve():
    assert _confidence_class_from_fve(0.30, is_mock=False) == "low"
    assert _confidence_class_from_fve(0.10, is_mock=False) == "low"


def test_confidence_class_mock_regardless_of_fve():
    """A MockNLA always reports 'mock' confidence so consumers never
    accidentally interpret mock outputs as a real NLA's verdict."""
    assert _confidence_class_from_fve(0.75, is_mock=True) == "mock"
    assert _confidence_class_from_fve(0.30, is_mock=True) == "mock"


# ── Tool execution: end-to-end with MockNLA ─────────────────────────────────


def _good_args(d_model: int = 64, layer_idx: int = 20) -> dict:
    return {
        "scenario_id": "smoke-1",
        "layer_idx": layer_idx,
        "activation_vector": [0.1 * i for i in range(d_model)],
        "d_model": d_model,
        "nla_model_id": "mock",
    }


def test_execute_returns_all_required_fields():
    result = execute_audit_activation_explanation(_good_args())
    required = {
        "scenario_id", "layer_idx", "explanation_text",
        "reconstruction_fve", "nla_model_id", "activation_norm",
        "confidence_class", "audit_hash", "raw_explanation",
    }
    assert set(result.keys()) >= required


def test_execute_mock_explainer_yields_mock_confidence():
    result = execute_audit_activation_explanation(_good_args())
    assert result["nla_model_id"] == "mock"
    assert result["confidence_class"] == "mock"


def test_execute_deterministic_on_same_input():
    """Same scenario, same layer, same activation → same audit_hash."""
    args = _good_args()
    r1 = execute_audit_activation_explanation(args)
    r2 = execute_audit_activation_explanation(dict(args))
    assert r1["audit_hash"] == r2["audit_hash"]
    assert r1["explanation_text"] == r2["explanation_text"]


def test_execute_different_activation_different_audit_hash():
    args1 = _good_args()
    args2 = _good_args()
    args2["activation_vector"] = [v + 0.01 for v in args2["activation_vector"]]
    r1 = execute_audit_activation_explanation(args1)
    r2 = execute_audit_activation_explanation(args2)
    assert r1["audit_hash"] != r2["audit_hash"]


def test_execute_missing_activation_vector_raises():
    args = _good_args()
    del args["activation_vector"]
    with pytest.raises(ValueError, match="activation_vector"):
        execute_audit_activation_explanation(args)


def test_execute_with_injected_explainer():
    """A pre-built explainer can be passed via args['explainer'] for tests."""
    args = _good_args(d_model=32)
    args["explainer"] = mock_explainer(d_model=32, layer_idx=20)
    args.pop("nla_model_id")   # explainer is provided directly
    args.pop("d_model")
    result = execute_audit_activation_explanation(args)
    assert result["nla_model_id"] == "mock"


def test_execute_result_is_json_serializable():
    """A core property: the result must round-trip JSON so it can be a
    Merkle leaf in a governance receipt."""
    result = execute_audit_activation_explanation(_good_args())
    blob = json.dumps(result, sort_keys=True)
    again = json.loads(blob)
    assert again == result


# ── Tool schema ────────────────────────────────────────────────────────────


def test_tool_schema_has_required_top_level_fields():
    assert TOOL_SCHEMA["name"] == "audit_activation_explanation"
    assert "description" in TOOL_SCHEMA
    assert "parameters" in TOOL_SCHEMA


def test_tool_schema_required_params_are_listed():
    params = TOOL_SCHEMA["parameters"]
    assert "scenario_id" in params["required"]
    assert "layer_idx" in params["required"]


def test_tool_schema_describes_fve_threshold():
    """The description should warn consumers about the FVE confidence floor."""
    assert "0.4" in TOOL_SCHEMA["description"] or "0.40" in TOOL_SCHEMA["description"]


# ── AuditExplanationResult to_dict ─────────────────────────────────────────


def test_audit_result_to_dict_round_trips():
    r = AuditExplanationResult(
        scenario_id="s",
        layer_idx=20,
        explanation_text="t",
        reconstruction_fve=0.5,
        nla_model_id="mock",
        activation_norm=1.0,
        confidence_class="mock",
        audit_hash="0" * 64,
        raw_explanation={"text": "t", "reconstruction_fve": 0.5,
                         "model_id": "mock", "layer_idx": 20,
                         "activation_norm": 1.0, "extra": {}},
    )
    d = r.to_dict()
    assert d["scenario_id"] == "s"
    assert d["audit_hash"] == "0" * 64
