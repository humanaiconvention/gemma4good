"""
v42 Boundary Guard v5 — H22 candidate (system-role rejection on top of v4)

Per the H22 precommit at ``docs/h22_precommit_hypothesis_2026-05-16.md``:

> A guard variant that retains all guard-v4 behavior (16 rules, Unicode
> normalization, per-message user scan) AND additionally rejects any
> incoming request whose ``messages`` array contains a ``role: system``
> entry at any position other than index 0 — and changes only that —
> will replay H21 within sampling noise, reject 100% of client-supplied
> system-role injections at positions 1+, pass through 100% of
> legitimate operator-supplied leading system prompts, and not regress
> on benign multi-turn conversation.

guard-v5 = guard-v4 + ``_detect_system_role_injection`` returning 400.
The detection logic is identical to what lived in guard-v2 (H19), now
isolated under a correct predeclaration that explicitly distinguishes
the operator's position-0 system prompt from client-supplied
system-role injection at later positions.
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

# Re-use the v4 multi-message extraction + apply function.
from tools.v42_boundary_guard_v4 import (
    _extract_user_texts,
    apply_guard_v4,
)


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard-v5] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard_v5")


# ── System-role injection detection ──────────────────────────────────────────


def _detect_system_role_injection(body: dict[str, Any]) -> bool:
    """Return True if any non-first message claims ``role: system``.

    Position 0 is explicitly permitted as the operator-supplied system
    prompt (canonical eval uses exactly this shape). Any subsequent
    ``role: system`` entry in the messages array is treated as a
    client-supplied injection attempt and the request is rejected.

    This function is the H22 hypothesis distilled to one boolean.
    """
    messages = body.get("messages", [])
    for i, msg in enumerate(messages):
        if i == 0:
            continue
        if msg.get("role") == "system":
            return True
    return False


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
        log.info("guard-v5 ready: upstream=%s rules=%d", upstream, len(RULES))
        yield
        log.info("guard-v5 shutdown")


app = FastAPI(title="v42 Boundary Guard v5 (H22)", version="0.5.0", lifespan=lifespan)
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
    return {"status": "ok", "service": "v42_boundary_guard_v5"}


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

    # H22-D2a: reject client-supplied role:system at any non-first position.
    # Position 0 is explicitly permitted as the operator's system prompt.
    if _detect_system_role_injection(body):
        log.info("system_role_in_history: rejecting request (HTTP 400)")
        raise HTTPException(
            status_code=400,
            detail="system_role_in_history: only the first message may be role:system",
        )

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
        model_id = body.get("model", "haic-gemma4-v42-guard-v5")
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
    parser = argparse.ArgumentParser(description="v42 Boundary Guard v5 (H22)")
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help=f"Upstream llama-server URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8086,
        help="Port to serve guard-v5 on (default: 8086, distinct from v1=8082, v2=8083, v3=8084, v4=8085)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    app.state.upstream = args.upstream
    log.info("Starting v42 boundary guard v5 on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
