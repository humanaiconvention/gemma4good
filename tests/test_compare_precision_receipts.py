"""Tests for tools/compare_precision_receipts.py.

Pure-Python — exercises the spread computation against synthetic receipts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_precision_receipts import (  # noqa: E402
    PrecisionSpread,
    compute_spread,
    _cis_overlap,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _report(grounding_rate=0.5, grounding_ci=(0.4, 0.6),
            security_rate=1.0, model_id="m", precision="4bit_nf4"):
    return {
        "finetune": {
            "sampling": {
                "grounding_pass_rate": grounding_rate,
                "grounding_ci95": list(grounding_ci),
                "security_pass_rate": security_rate,
                "security_ci95": [security_rate, 1.0],
                "model_id": model_id,
                "decoding": {"precision": precision},
            }
        }
    }


# ── _cis_overlap ────────────────────────────────────────────────────────────


class TestCisOverlap:

    def test_clearly_overlapping(self):
        assert _cis_overlap([0.3, 0.6], [0.4, 0.7])

    def test_disjoint(self):
        assert not _cis_overlap([0.1, 0.3], [0.5, 0.7])
        assert not _cis_overlap([0.5, 0.7], [0.1, 0.3])

    def test_touching_at_boundary_counts_as_overlap(self):
        assert _cis_overlap([0.3, 0.5], [0.5, 0.7])


# ── compute_spread ──────────────────────────────────────────────────────────


class TestComputeSpread:

    def test_zero_spread_is_pass(self):
        e = _report(grounding_rate=0.95, grounding_ci=(0.85, 1.00),
                    security_rate=0.95, model_id="m-eval", precision="4bit_nf4")
        d = _report(grounding_rate=0.95, grounding_ci=(0.85, 1.00),
                    security_rate=0.95, model_id="m-deploy", precision="GGUF Q5_K_M")
        s = compute_spread(e, d)
        assert s.verdict == "PASS"
        assert s.grounding_spread_pp == 0.0
        assert s.security_spread_pp == 0.0

    def test_small_spread_within_tolerance_passes(self):
        e = _report(grounding_rate=1.00, grounding_ci=(0.89, 1.00),
                    security_rate=0.95, precision="4bit_nf4")
        d = _report(grounding_rate=0.97, grounding_ci=(0.85, 1.00),
                    security_rate=0.93, precision="GGUF Q5_K_M")
        s = compute_spread(e, d, tolerance_pp=5.0)
        # 3 pp grounding, 2 pp security — both under 5 pp tolerance
        assert s.verdict == "PASS"
        assert abs(s.grounding_spread_pp - 3.0) < 0.001
        assert abs(s.security_spread_pp - 2.0) < 0.001

    def test_large_grounding_spread_fails(self):
        e = _report(grounding_rate=1.00, grounding_ci=(0.89, 1.00),
                    security_rate=0.95)
        d = _report(grounding_rate=0.80, grounding_ci=(0.65, 0.92),
                    security_rate=0.95)
        s = compute_spread(e, d, tolerance_pp=5.0)
        # 20 pp grounding spread → FAIL
        assert s.verdict == "FAIL"
        assert s.grounding_spread_pp > 5.0

    def test_large_security_spread_fails(self):
        e = _report(grounding_rate=1.00, grounding_ci=(0.89, 1.00),
                    security_rate=0.95)
        d = _report(grounding_rate=1.00, grounding_ci=(0.89, 1.00),
                    security_rate=0.70)
        s = compute_spread(e, d, tolerance_pp=5.0)
        # 25 pp security spread → FAIL
        assert s.verdict == "FAIL"
        assert s.security_spread_pp > 5.0

    def test_within_tolerance_but_cis_disjoint_is_partial(self):
        # Point estimates close but CIs don't overlap — measurement at
        # this n cannot distinguish them statistically AND they aren't
        # in the same range.
        e = _report(grounding_rate=1.00, grounding_ci=(0.95, 1.00),
                    security_rate=0.95)
        d = _report(grounding_rate=0.96, grounding_ci=(0.80, 0.92),
                    security_rate=0.95)
        s = compute_spread(e, d, tolerance_pp=5.0)
        # 4 pp grounding spread (within tolerance) but CIs disjoint
        assert s.verdict == "PARTIAL"
        assert "do NOT overlap" in s.rationale or "do not overlap" in s.rationale.lower()

    def test_records_precision_labels(self):
        e = _report(precision="4bit_nf4")
        d = _report(precision="GGUF Q5_K_M")
        s = compute_spread(e, d)
        assert s.eval_precision == "4bit_nf4"
        assert s.deploy_precision == "GGUF Q5_K_M"

    def test_records_model_ids(self):
        e = _report(model_id="haic-gemma4-v39")
        d = _report(model_id="haic-gemma4-v39-q5km")
        s = compute_spread(e, d)
        assert s.eval_model_id == "haic-gemma4-v39"
        assert s.deploy_model_id == "haic-gemma4-v39-q5km"

    def test_tolerance_threshold_respected(self):
        # 4 pp spread at tolerance 3 pp → FAIL
        e = _report(grounding_rate=1.00, grounding_ci=(0.85, 1.00))
        d = _report(grounding_rate=0.96, grounding_ci=(0.85, 1.00))
        s = compute_spread(e, d, tolerance_pp=3.0)
        assert s.verdict == "FAIL"

        # Same spread at tolerance 10 pp → PASS
        s2 = compute_spread(e, d, tolerance_pp=10.0)
        assert s2.verdict == "PASS"

    def test_precision_spread_dataclass_serializable(self):
        e = _report()
        d = _report()
        s = compute_spread(e, d)
        # __dict__ access for json.dumps
        d_out = s.__dict__
        assert "verdict" in d_out
        assert "grounding_spread_pp" in d_out
        # Should be serializable
        import json as _json
        _json.dumps(d_out)
