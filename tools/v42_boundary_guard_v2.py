"""
v42 Boundary Guard v2 — H19 experiment

Extends v42_boundary_guard.py with two predeclared changes (see
``docs/h19_precommit_hypothesis_2026-05-16.md``):

1. **Unicode normalization** before regex matching:
   - NFKC normalization folds fullwidth Latin and mathematical bold to ASCII.
   - Zero-width and control characters are stripped.
   - A fixed lookalike table folds common Cyrillic / Greek homoglyphs to Latin.
   The raw user text is forwarded unmodified to the upstream model; only the
   text passed into the regex engine is normalized.

2. **Multi-message scanning**: every ``role: user`` message in the chat
   history is evaluated. Client-supplied ``role: system`` messages after the
   first message are rejected with HTTP 400 ``system_role_in_history``.

The 16 H18r4 rule patterns are imported byte-identical from v1; only the
pre-match pipeline and message-iteration change. The H18r4 anchor remains
valid for the v1 guard; v2 carries its own anchor when H19 evaluation passes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
import unicodedata
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# Re-use the v1 rule set, decision dataclass, hash helper, and completion
# builder verbatim — H19 must not change matching behavior on canonical
# (ASCII) inputs, only add normalization on top.
from tools.v42_boundary_guard import (
    RULES,
    GuardDecision,
    _build_completion,
    _sha3,
    _PASSTHROUGH_GET_ALLOWLIST,
)


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard-v2] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard_v2")


# ── Unicode normalization ────────────────────────────────────────────────────

# Cyrillic and Greek letters that visually match Latin letters. The list is
# deliberately conservative — only the highest-confidence pairs are folded so
# that legitimate Cyrillic/Greek text in benign messages is NOT mangled
# (e.g. Russian "капитан" / Cyrillic "к" → not in this table). Adding more
# entries here would increase H19-B coverage at the cost of H19-C FP risk.
_HOMOGLYPH_TABLE = str.maketrans({
    # Cyrillic lowercase → Latin lowercase
    "а": "a",  # а
    "е": "e",  # е
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "у": "y",  # у
    "х": "x",  # х
    # Cyrillic uppercase → Latin uppercase
    "А": "A",  # А
    "В": "B",  # В
    "Е": "E",  # Е
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "Х": "X",  # Х
    # Greek lowercase → Latin lowercase
    "ο": "o",  # ο omicron
    "ρ": "p",  # ρ rho
    "ε": "e",  # ε epsilon
})


def _normalize_for_match(text: str) -> str:
    """Apply NFKC + zero-width strip + homoglyph fold for regex matching.

    This is the H19-A1..A12 + H19-B path. Returns a string with the same
    semantic content as ``text`` for an English reader, but with:
      - fullwidth Latin → ASCII Latin
      - mathematical-bold Latin → ASCII Latin
      - zero-width / format / control chars (except \\n \\r \\t) removed
      - high-confidence Cyrillic/Greek lookalikes folded to Latin
    """
    if not text:
        return text
    # NFKC handles fullwidth, mathematical-bold, mathematical-italic,
    # superscript/subscript Latin, and many other presentation-form chars.
    nfkc = unicodedata.normalize("NFKC", text)
    # Strip Cf (format, includes ZWSP/ZWJ/ZWNJ/BOM/RLO/LRO/etc.) and Cc
    # (control) except whitespace that participants may legitimately use.
    stripped_chars = []
    for ch in nfkc:
        cat = unicodedata.category(ch)
        if cat in ("Cf",) or (cat == "Cc" and ch not in "\n\r\t"):
            continue
        stripped_chars.append(ch)
    stripped = "".join(stripped_chars)
    # Apply the homoglyph fold last so the post-NFKC text is the target.
    return stripped.translate(_HOMOGLYPH_TABLE)


# ── Message extraction ───────────────────────────────────────────────────────


def _extract_user_texts(body: dict) -> list[str]:
    """Return every ``role: user`` message's text content.

    Multimodal content arrays (`[{"type": "text", "text": ...}, ...]`)
    are flattened to a space-joined string per message, matching v1's
    behavior.
    """
    out: list[str] = []
    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.append(" ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            ))
    return out


def _detect_system_role_injection(body: dict) -> bool:
    """Return True if any non-first message claims ``role: system``.

    A single leading system message is the operator-supplied system prompt
    and is permitted; any subsequent system message is treated as an
    injection attempt.
    """
    messages = body.get("messages", [])
    for i, msg in enumerate(messages):
        if i == 0:
            continue
        if msg.get("role") == "system":
            return True
    return False


# ── Guard logic ──────────────────────────────────────────────────────────────


def apply_guard_v2(user_texts: list[str]) -> tuple[GuardDecision, list[int]]:
    """Evaluate every user message under H19 normalization.

    Returns ``(decision, matched_message_indices)``. The decision's
    ``request_hash`` is computed over the joined raw texts (no normalization)
    so the audit log is faithful to what the client sent.

    First match across all messages wins; matched message indices and rule
    IDs are surfaced in metadata for audit.
    """
    raw_joined = "".join(user_texts)  # unit separator — never appears in real text
    request_hash = _sha3(raw_joined)

    matched_rules: list = []
    matched_indices: list[int] = []
    seen_rule_ids: set[str] = set()

    for idx, text in enumerate(user_texts):
        norm = _normalize_for_match(text)
        for rule in RULES:
            if rule.pattern.search(norm):
                if rule.rule_id not in seen_rule_ids:
                    matched_rules.append(rule)
                    seen_rule_ids.add(rule.rule_id)
                if idx not in matched_indices:
                    matched_indices.append(idx)

    if not matched_rules:
        return (
            GuardDecision(
                guard_triggered=False,
                guard_class=None,
                matched_rule_ids=[],
                response_source="model",
                response_text=None,
                request_hash=request_hash,
            ),
            [],
        )

    primary = matched_rules[0]
    return (
        GuardDecision(
            guard_triggered=True,
            guard_class=primary.guard_class,
            matched_rule_ids=[r.rule_id for r in matched_rules],
            response_source="guard",
            response_text=primary.response_text,
            request_hash=request_hash,
        ),
        matched_indices,
    )


# ── FastAPI app ──────────────────────────────────────────────────────────────


DEFAULT_UPSTREAM = "http://127.0.0.1:8081"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    upstream = getattr(app.state, "upstream", DEFAULT_UPSTREAM)
    timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=10.0)
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=8)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        app.state.http_client = client
        app.state.upstream = upstream
        log.info("guard-v2 ready: upstream=%s rules=%d", upstream, len(RULES))
        yield
        log.info("guard-v2 shutdown")


app = FastAPI(title="v42 Boundary Guard v2 (H19)", version="0.2.0", lifespan=lifespan)
app.state.upstream = DEFAULT_UPSTREAM


def _client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def _upstream_url(request: Request) -> str:
    return request.app.state.upstream


def _bad_gateway(detail: str, exc: Exception | None = None) -> Response:
    log.warning("upstream_error %s: %s", detail, exc if exc else "")
    payload = {"error": {"type": "bad_gateway", "message": detail}}
    return Response(
        content=json.dumps(payload),
        status_code=502,
        media_type="application/json",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "v42_boundary_guard_v2"}


@app.get("/guard/rules")
async def list_rules() -> list[dict[str, str]]:
    return [
        {
            "rule_id": r.rule_id,
            "guard_class": r.guard_class,
            "pattern": r.pattern.pattern,
        }
        for r in RULES
    ]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    # H19-D2: reject client-supplied role:system in history (after position 0).
    if _detect_system_role_injection(body):
        raise HTTPException(
            status_code=400,
            detail="system_role_in_history: only the first message may be role:system",
        )

    user_texts = _extract_user_texts(body)
    decision, matched_indices = apply_guard_v2(user_texts)

    log.info(
        "guard_decision request_hash=%s triggered=%s class=%s rules=%s msgs=%s source=%s",
        decision.request_hash[:16],
        decision.guard_triggered,
        decision.guard_class,
        decision.matched_rule_ids,
        matched_indices,
        decision.response_source,
    )

    if decision.guard_triggered:
        model_id = body.get("model", "haic-gemma4-v42-guard-v2")
        stream = body.get("stream", False)
        payload = _build_completion(decision, model_id, stream=stream)
        # H19-specific metadata: which messages triggered.
        payload["guard_metadata"]["matched_message_indices"] = matched_indices

        if stream:
            chunk = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"

            async def gen():
                yield chunk.encode()

            return StreamingResponse(gen(), media_type="text/event-stream")
        return Response(
            content=json.dumps(payload),
            media_type="application/json",
        )

    client = _client(request)
    upstream = _upstream_url(request)
    try:
        upstream_resp = await client.post(
            f"{upstream}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
        )
    except httpx.TimeoutException as exc:
        return _bad_gateway("upstream timeout", exc)
    except httpx.HTTPError as exc:
        return _bad_gateway("upstream unreachable", exc)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def proxy_safe_get(request: Request, path: str):
    if path not in _PASSTHROUGH_GET_ALLOWLIST:
        raise HTTPException(status_code=404, detail="not found")

    client = _client(request)
    upstream = _upstream_url(request)
    try:
        upstream_resp = await client.request(
            method=request.method,
            url=f"{upstream}/{path}",
            params=dict(request.query_params),
            timeout=30.0,
        )
    except httpx.TimeoutException as exc:
        return _bad_gateway("upstream timeout", exc)
    except httpx.HTTPError as exc:
        return _bad_gateway("upstream unreachable", exc)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="v42 Boundary Guard v2 (H19)")
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help=f"Upstream llama-server URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8083,
        help="Port to serve guard-v2 on (default: 8083, distinct from v1's 8082)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    app.state.upstream = args.upstream
    log.info("Starting v42 boundary guard v2 on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
