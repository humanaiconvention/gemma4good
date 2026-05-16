"""
v42 Boundary Guard v4 — H21 candidate (multi-message-aware v3)

Per the H21 precommit at ``docs/h21_precommit_hypothesis_2026-05-16.md``:

> A guard variant that retains all guard-v3 behavior (16 rules, Unicode
> normalization, single-message scan) AND additionally evaluates every
> ``role: user`` message in the conversation history — and changes
> only that — will replay H20 within sampling noise, trigger on
> multi-message attacks ≥ 0.95, and not introduce benign multi-turn
> FP regressions.

This module is the strictly-simpler successor to ``v42_boundary_guard_v2.py``:

  guard-v4 = guard-v3 + per-message scan over user-role messages
           (NO system-role-in-history rejection, NO new metadata
            that affects matching behavior)

System-role-in-history rejection — the H19-D2 confound — is deliberately
NOT included. That property is its own L-02b limitation, deferred to a
future H22 hypothesis with a corrected predeclaration.

The 16 H18r4 rule set, Unicode normalization, and ASCII-attack matching
behavior are imported verbatim from guard-v3. The only behavioral
change is iterating every user-role message instead of only the last.
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
# builder, and pass-through allowlist verbatim.
from tools.v42_boundary_guard import (
    RULES,
    GuardDecision,
    _build_completion,
    _sha3,
    _PASSTHROUGH_GET_ALLOWLIST,
)

# Re-use the v2 normalization function. Identical to what guard-v3 uses.
from tools.v42_boundary_guard_v2 import _normalize_for_match


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard-v4] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard_v4")


# ── Message extraction ───────────────────────────────────────────────────────


def _extract_user_texts(body: dict[str, Any]) -> list[str]:
    """Return every ``role: user`` message's text content, in order.

    Multimodal content arrays are flattened the same way as in v1's
    ``_extract_user_text``. Each element of the returned list corresponds
    to one user-role message; downstream callers map these to message
    indices.
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


# ── Guard logic ──────────────────────────────────────────────────────────────


def apply_guard_v4(user_texts: list[str]) -> tuple[GuardDecision, list[int]]:
    """Evaluate every user message under H20 normalization.

    Pipeline (per H21 spec):
      1. Hash the joined raw user texts (audit log faithful to what client sent).
      2. For each user message, NFKC-normalize + strip zero-width + fold
         Cyrillic/Greek homoglyphs, then run the 16 H18r4 regex rules.
      3. If any rule matches in any message, trigger with the first rule
         that matched (in v1 RULES order). Return both the decision and
         the list of message indices that contained matches.

    First-match-wins is preserved across messages: if message 0 fires
    DI-01 and message 2 fires CC-03, the primary class is DIRECT_INJECT
    and the response is the DI-01 response — same shape as the
    single-message guard.
    """
    # Sentinel-joined hash so empty strings between messages are still
    # distinct. The unit separator U+001F never appears in real text.
    raw_joined = "\x1f".join(user_texts)
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
        log.info("guard-v4 ready: upstream=%s rules=%d", upstream, len(RULES))
        yield
        log.info("guard-v4 shutdown")


app = FastAPI(title="v42 Boundary Guard v4 (H21)", version="0.4.0", lifespan=lifespan)
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
    return {"status": "ok", "service": "v42_boundary_guard_v4"}


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
    user_texts = _extract_user_texts(body)
    decision, matched_indices = apply_guard_v4(user_texts)

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
        model_id = body.get("model", "haic-gemma4-v42-guard-v4")
        stream = body.get("stream", False)
        payload = _build_completion(decision, model_id, stream=stream)
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
    parser = argparse.ArgumentParser(description="v42 Boundary Guard v4 (H21)")
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help=f"Upstream llama-server URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8085,
        help="Port to serve guard-v4 on (default: 8085, distinct from v1=8082, v2=8083, v3=8084)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    app.state.upstream = args.upstream
    log.info("Starting v42 boundary guard v4 on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
