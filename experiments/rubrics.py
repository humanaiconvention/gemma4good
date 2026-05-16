"""
Canonical evaluation rubrics — the load-bearing classifiers behind every
H-series gate.

This module is the named entry point for the strict and v1 rubrics that
canonical_eval.py uses to grade responses. The two original implementations
live in `rescore_concealed_strict.py` (named after the offline rescore tool
that first introduced the strict rubric) and `run_v39_gguf_v2_scenarios.py`
(named after the v39 scenario harness that first defined the v1 rubric).

Those filenames look archival, but the rubrics themselves are canonical and
underpin every promotion decision, including H18r4 (guard + v42). This
module re-exports them under stable, neutral names so:

  - `canonical_eval` and `tools/` import from `experiments.rubrics`
  - the two historical scripts can be treated as appendix material
    without breaking the promotion path
  - any future replacement rubric is predeclared (see
    `docs/evaluation_doctrine.md`) and lands here, not in those scripts

If a rubric changes here, every H-series anchor produced before the change
is invalidated. Treat this file like a public API.
"""

from __future__ import annotations

# Strict concealed-compliance rubric — used by H18b, H17b, H16b, H15b.
from experiments.rescore_concealed_strict import (
    EXPLICIT_REFUSAL_PATTERNS,
    SEMANTIC_LEAK_PATTERNS,
    classify as classify_strict,
    wilson_ci,
)

# v1 security rubric — used by H18a aggregate_security and all per-scenario
# H-series gates (H18e–H18j).
from experiments.run_v39_gguf_v2_scenarios import (
    REFUSAL_MARKERS,
    NEGATION_PREFIXES,
    SECURITY_RUBRICS,
    PIVOT_RE,
    V38_SYSTEM_PROMPT,
    stayed_in_protocol,
    refused,
    complied,
    grade as grade_v1,
)

__all__ = [
    # Strict rubric
    "EXPLICIT_REFUSAL_PATTERNS",
    "SEMANTIC_LEAK_PATTERNS",
    "classify_strict",
    "wilson_ci",
    # v1 rubric
    "REFUSAL_MARKERS",
    "NEGATION_PREFIXES",
    "SECURITY_RUBRICS",
    "PIVOT_RE",
    "V38_SYSTEM_PROMPT",
    "stayed_in_protocol",
    "refused",
    "complied",
    "grade_v1",
]
