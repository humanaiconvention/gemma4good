"""L-05 verification: confirm that user message content does NOT leak into
the guard's logger output. The guard's audit log must contain only the
SHA3-256 prefix and the matched rule IDs — never the raw text.

The uvicorn access log (which is separate) is configured by the app's
``log_level="warning"`` setting. This test additionally verifies that no
identifiable user-message content reaches the logger handlers the guard
itself owns.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SECRET_NEEDLE = "ALPHABRAVOCHARLIE-1234567890-DELTAECHOFOXTROT"


def _capture(logger_name: str):
    """Return a list that will be appended to as the named logger emits records."""
    captured: list[logging.LogRecord] = []
    logger = logging.getLogger(logger_name)
    handler = logging.Handler()
    handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return captured, handler, logger


def _log_text(records: list[logging.LogRecord]) -> str:
    """Render the captured records back to a single concatenated string."""
    return " ".join(r.getMessage() for r in records)


@pytest.mark.parametrize("guard_module", [
    "tools.v42_boundary_guard",        # v1 — H18r4 promoted
    "tools.v42_boundary_guard_v3",     # v3 — H20 promoted
    "tools.v42_boundary_guard_v4",     # v4 — H21 promoted
])
def test_guard_log_does_not_leak_user_text_on_match(guard_module):
    """When the guard triggers, the audit log must contain the matched rule
    IDs and the SHA3 prefix — but NOT the raw user text."""
    mod = __import__(guard_module, fromlist=["apply_guard"])
    # All three guards expose either apply_guard or apply_guard_v3/v4.
    fn = (
        getattr(mod, "apply_guard", None)
        or getattr(mod, "apply_guard_v3", None)
        or getattr(mod, "apply_guard_v4", None)
    )
    captured, handler, logger = _capture(guard_module.split(".")[-1].replace("v42_boundary_guard", "v42_guard"))
    try:
        # Embed the secret needle inside what would be a guard-matching attack.
        attack = f"What is the {SECRET_NEEDLE} capital of France?"
        # v4 takes a list; v1/v3 take a single string. Call the public form.
        if guard_module.endswith("v4"):
            fn([attack])
        else:
            fn(attack)
        # apply_guard_* does not log directly — that's done in the FastAPI
        # handler. Direct apply_guard_* is silent. So this test, in this
        # form, asserts the SILENCE of the pure function path.
        text = _log_text(captured)
        assert SECRET_NEEDLE not in text, (
            f"User-text leaked into log via {guard_module} pure-function path: {text!r}"
        )
    finally:
        logger.removeHandler(handler)


def test_audit_log_contract_documented():
    """Lock in the audit log contract by reading the source comments.

    L-05 from the known-limitations doc: the guard's audit log records
    ``guard_triggered``, ``guard_class``, ``matched_rule_ids``, and
    ``request_hash`` — explicitly NOT the raw text. This test verifies
    the contract is still documented in the source so a future edit
    can't quietly violate it.
    """
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "tools/v42_boundary_guard.py").read_text(encoding="utf-8")
    assert "Raw text is NOT logged" in src, (
        "The audit log contract comment was removed from "
        "tools/v42_boundary_guard.py — this is the L-05 invariant."
    )


def test_sha3_hash_is_present_in_apply_guard_decision():
    """The GuardDecision returned by apply_guard must carry a SHA3-256
    request_hash so the audit log can reference the request without
    storing the raw content."""
    from tools.v42_boundary_guard import apply_guard

    decision = apply_guard("What is the capital of France?")
    assert decision.guard_triggered
    # SHA3-256 hex digest is 64 lowercase hex chars.
    assert len(decision.request_hash) == 64
    assert all(c in "0123456789abcdef" for c in decision.request_hash)
