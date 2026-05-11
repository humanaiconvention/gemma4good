"""Tests for prism_integration/nla.py — NLA inference interface."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism_integration.nla import (
    MockNLA,
    NLAExplanation,
    NLAExplainerProtocol,
    get_explainer,
    mock_explainer,
)


# ── MockNLA contract ──────────────────────────────────────────────────────


def test_mock_explainer_basic_contract():
    """A MockNLA explanation must contain every field declared on NLAExplanation."""
    exp = mock_explainer(d_model=1536, layer_idx=20)
    vec = [0.0] * 1536
    result = exp.explain(vec)
    assert isinstance(result, NLAExplanation)
    assert isinstance(result.text, str) and len(result.text) > 0
    assert 0.0 <= result.reconstruction_fve <= 1.0
    assert result.model_id == "mock"
    assert result.layer_idx == 20
    assert result.activation_norm == 0.0   # zero vector → zero norm
    assert "mock" in result.text.lower()


def test_mock_explainer_deterministic_same_vector_same_output():
    """Same activation → same explanation (essential for test reproducibility)."""
    exp = mock_explainer(d_model=128)
    vec = [float(i) / 100 for i in range(128)]
    r1 = exp.explain(vec)
    r2 = exp.explain(vec)
    assert r1.text == r2.text
    assert r1.reconstruction_fve == r2.reconstruction_fve
    assert r1.activation_norm == r2.activation_norm


def test_mock_explainer_different_vectors_different_themes():
    """Different inputs → different theme indices (avoids the
    degenerate-mock case where all inputs map to the same explanation)."""
    exp = mock_explainer(d_model=64)
    # Construct two clearly-different vectors
    v1 = [1.0] + [0.0] * 63
    v2 = [0.0] * 63 + [1.0]
    r1 = exp.explain(v1)
    r2 = exp.explain(v2)
    # At minimum, the activation_norm differs and the hash-derived FVE differs.
    # (Theme text may collide rarely, so don't require text difference.)
    assert r1.activation_norm == 1.0
    assert r2.activation_norm == 1.0
    # FVE depends on the hash of the first 32 entries — these differ
    assert r1.reconstruction_fve != r2.reconstruction_fve


def test_mock_fve_in_realistic_range():
    """FVE should be in [0.35, 0.65] to match Anthropic's warm-start SFT range."""
    exp = mock_explainer(d_model=64)
    rs = [exp.explain([float(i % 7) for i in range(64)]) for _ in range(10)]
    for r in rs:
        assert 0.35 <= r.reconstruction_fve <= 0.65


def test_mock_explainer_d_model_mismatch_raises():
    exp = mock_explainer(d_model=1536)
    with pytest.raises(ValueError, match="d_model=1536"):
        exp.explain([0.0] * 512)


def test_mock_explain_batch_returns_list_of_explanations():
    exp = mock_explainer(d_model=32)
    vecs = [[float(i + offset) for i in range(32)] for offset in range(5)]
    results = exp.explain_batch(vecs)
    assert len(results) == 5
    for r in results:
        assert isinstance(r, NLAExplanation)


def test_mock_explainer_satisfies_protocol():
    exp = mock_explainer(d_model=64)
    assert isinstance(exp, NLAExplainerProtocol)


# ── Factory behavior ──────────────────────────────────────────────────────


def test_factory_mock_returns_mocknla():
    exp = get_explainer("mock", d_model=1536, layer_idx=20)
    assert isinstance(exp, MockNLA)
    assert exp.d_model == 1536
    assert exp.layer_idx == 20


def test_factory_mock_requires_d_model():
    with pytest.raises(ValueError, match="d_model"):
        get_explainer("mock")


def test_factory_real_model_id_falls_back_when_transport_missing():
    """The most common "real NLA not usable" case: prism.nla is installed,
    the model_id is in the registry, but no HTTP server is reachable so
    no transport can be built. With fallback_to_mock=True, factory falls
    back to MockNLA cleanly."""
    exp = get_explainer(
        "kitft/nla-gemma-3-12b-it-layer32",
        d_model=4608,
        layer_idx=32,
        fallback_to_mock=True,
        # Intentionally NO server_url= or transport= → real NLA construction
        # raises ValueError, which our adapter maps to ImportError, which
        # the factory's fallback path catches.
    )
    assert isinstance(exp, MockNLA)


def test_factory_real_model_id_raises_without_fallback():
    """Without fallback_to_mock, the factory raises a helpful ImportError
    (which wraps the underlying ValueError/KeyError from PRISM)."""
    with pytest.raises(ImportError):
        get_explainer(
            "kitft/nla-gemma-3-12b-it-layer32",
            d_model=4608,
            layer_idx=32,
            fallback_to_mock=False,
        )


def test_factory_unknown_model_id_falls_back():
    """An NLA model_id not in PRISM's registry → KeyError inside the
    adapter, translated to ImportError, caught by fallback_to_mock."""
    exp = get_explainer(
        "fictitious/nla-does-not-exist-layer99",
        d_model=128,
        layer_idx=42,
        fallback_to_mock=True,
    )
    assert isinstance(exp, MockNLA)


def test_factory_real_model_id_with_transport_succeeds():
    """When a transport IS supplied (e.g. a callable stub for tests),
    the adapter constructs a real NLAExplainer and we get back a
    _PrismNLAAdapter — NOT a MockNLA."""
    def stub_transport(payload: dict) -> dict:
        return {
            "text": "stub-transport explanation",
            "reconstruction_fve": 0.42,
            "reconstructed_vector": [0.0] * len(payload.get("activation_vector", [])),
        }
    from prism_integration.nla import _PrismNLAAdapter
    exp = get_explainer(
        "kitft/nla-gemma-3-12b-it-layer32",
        d_model=4608,
        layer_idx=32,
        transport=stub_transport,
        # fallback_to_mock irrelevant here — construction should succeed
    )
    assert isinstance(exp, _PrismNLAAdapter)
    assert not isinstance(exp, MockNLA)


# ── NLAExplanation serialization ──────────────────────────────────────────


def test_nla_explanation_to_dict_round_trip():
    exp = mock_explainer(d_model=64)
    result = exp.explain([0.5] * 64)
    d = result.to_dict()
    assert d["text"] == result.text
    assert d["reconstruction_fve"] == result.reconstruction_fve
    assert d["model_id"] == result.model_id
    assert "extra" in d


# ── Integration sketch: NLA fits into a governance-receipt-style trace ────


def test_nla_explanation_is_json_serializable():
    """A core property: NLA outputs must be JSON-serializable so they can
    be folded into governance receipts and Merkle leaves."""
    import json
    exp = mock_explainer(d_model=64)
    result = exp.explain([0.3] * 64)
    blob = json.dumps(result.to_dict(), sort_keys=True)
    # Round-trip
    round_tripped = json.loads(blob)
    assert round_tripped["text"] == result.text
    assert round_tripped["reconstruction_fve"] == result.reconstruction_fve
