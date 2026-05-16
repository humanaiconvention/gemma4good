"""
v42 Boundary Guard v3 — H20 candidate (Unicode-normalized v1)

Per the H20 precommit hypothesis at
``docs/h20_precommit_hypothesis_2026-05-16.md``:

> A guard variant that NFKC-normalizes user text and strips zero-width /
> control characters and folds high-confidence Cyrillic / Greek
> homoglyphs to Latin before regex matching — and changes **only**
> that — will replay H18r4 within sampling noise, close the L-01
> Unicode bypass at ≥ 0.95, and stay under 0.02 benign FP on legitimate
> non-Latin text.

This module is the strictly-simpler successor to ``v42_boundary_guard_v2.py``:

  guard-v3 = guard-v1 (last-message scan, no system-role rejection)
           + v2._normalize_for_match (NFKC + zero-width strip + homoglyph fold)

The multi-message attack scan and system-role rejection that confused
H19's evaluation are intentionally NOT included here. They are deferred
to a future H21 with a cleanly-scoped suite.

The H18r4 16-rule set is imported verbatim from the v1 guard. No rule
changes. Matching behavior on ASCII canonical attacks is identical
because NFKC is the identity on pure ASCII.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# Re-use the v1 rule set, decision dataclass, hash helper, completion
# builder, and allowlist verbatim.
from tools.v42_boundary_guard import (
    RULES,
    GuardDecision,
    _build_completion,
    _extract_user_text,
    _sha3,
    _PASSTHROUGH_GET_ALLOWLIST,
)

# Re-use the v2 normalization function and homoglyph table verbatim.
# These are the ONLY net-new behavior introduced by v3 relative to v1.
from tools.v42_boundary_guard_v2 import _normalize_for_match


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard-v3] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard_v3")


# ── Guard logic ──────────────────────────────────────────────────────────────

def apply_guard_v3(user_text: str) -> GuardDecision:
    """Evaluate the last user message under H20 normalization.

    Pipeline:
      1. Hash the raw user text (audit log faithful to what client sent).
      2. NFKC-normalize + strip zero-width + fold Cyrillic/Greek homoglyphs.
      3. Run the 16 H18r4 regex rules over the normalized text.
      4. Return a GuardDecision with the v1 shape.
    """
    request_hash = _sha3(user_text)
    norm = _normalize_for_match(user_text)

    matched: list = []
    for rule in RULES:
        if rule.pattern.search(norm):
            matched.append(rule)

    if not matched:
        return GuardDecision(
            guard_triggered=False,
            guard_class=None,
            matched_rule_ids=[],
            response_source="model",
            response_text=None,
            request_hash=request_hash,
        )

    primary = matched[0]
    return GuardDecision(
        guard_triggered=True,
        guard_class=primary.guard_class,
        matched_rule_ids=[r.rule_id for r in matched],
        response_source="guard",
        response_text=primary.response_text,
        request_hash=request_hash,
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
        log.info("guard-v3 ready: upstream=%s rules=%d", upstream, len(RULES))
        yield
        log.info("guard-v3 shutdown")


app = FastAPI(title="v42 Boundary Guard v3 (H20)", version="0.3.0", lifespan=lifespan)
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
    return {"status": "ok", "service": "v42_boundary_guard_v3"}


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
    user_text = _extract_user_text(body)
    decision = apply_guard_v3(user_text)

    log.info(
        "guard_decision request_hash=%s triggered=%s class=%s rules=%s source=%s",
        decision.request_hash[:16],
        decision.guard_triggered,
        decision.guard_class,
        decision.matched_rule_ids,
        decision.response_source,
    )

    if decision.guard_triggered:
        model_id = body.get("model", "haic-gemma4-v42-guard-v3")
        stream = body.get("stream", False)
        payload = _build_completion(decision, model_id, stream=stream)

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
    parser = argparse.ArgumentParser(description="v42 Boundary Guard v3 (H20)")
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help=f"Upstream llama-server URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8084,
        help="Port to serve guard-v3 on (default: 8084, distinct from v1's 8082 and v2's 8083)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    app.state.upstream = args.upstream
    log.info("Starting v42 boundary guard v3 on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
