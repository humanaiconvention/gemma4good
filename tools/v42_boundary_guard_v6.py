"""
v42 Boundary Guard v6 — H24 candidate (leet-fold pre-pass on top of v5)

Per the H24 precommit at ``docs/h24_precommit_hypothesis_2026-05-16.md``:

> A guard variant that retains all guard-v5 behavior (16 rules, Unicode
> normalization, per-message scan, system-role rejection) AND
> additionally runs the H18r4 regex rule set against a leet-folded
> copy of the normalized text (substituting digits 0→o, 1→i, 3→e,
> 4→a, 5→s, 7→t before matching) — triggering if EITHER the original
> normalized text or the leet-folded text matches a rule.

guard-v6 = guard-v5 + ``_leet_fold`` + dual-surface rule matching.

The leet table is intentionally restricted to six common digit-to-letter
substitutions (0, 1, 3, 4, 5, 7). Less-common substitutions (8→b, 9→g,
2→z) are deliberately excluded — they have higher benign-use frequency
relative to attack-use frequency. Benign digit-containing text (ages,
prices, room numbers) does not match the H18r4 patterns when leet-decoded
under this minimal table, which is what H24-E1 verifies.

The H24 hypothesis tests whether L-08 (leetspeak / character-substitution
bypass, discovered in H23) can be closed by this dual-surface approach
without introducing benign-text false positives.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
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

# Re-use the v2 normalization function (used by v3/v4/v5).
from tools.v42_boundary_guard_v2 import _normalize_for_match

# Re-use v4's per-message user extraction.
from tools.v42_boundary_guard_v4 import _extract_user_texts

# Re-use v5's system-role injection detector.
from tools.v42_boundary_guard_v5 import _detect_system_role_injection


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard-v6] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard_v6")


# ── Leet-fold ────────────────────────────────────────────────────────────────


# The six common leetspeak digit-to-letter substitutions. Excludes
# 8→b / 9→g / 2→z (higher benign-use frequency, would expand FP surface).
_LEET_TABLE: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
}

# Build a str.translate table for speed.
_LEET_TRANSLATION = str.maketrans(_LEET_TABLE)


def _leet_fold(text: str) -> str:
    """Substitute leetspeak digits with their letter equivalents.

    Operates on text that has ALREADY been NFKC-normalized + zero-width-
    stripped + homoglyph-folded by ``_normalize_for_match``. This is a
    second matching surface, not a replacement for the first; the guard
    matches against both the original normalized text and this folded
    text and triggers on either.

    Idempotent: applying twice produces the same result.
    """
    return text.translate(_LEET_TRANSLATION)


# ── Guard logic ──────────────────────────────────────────────────────────────


def apply_guard_v6(user_texts: list[str]) -> tuple[GuardDecision, list[int]]:
    """Evaluate every user message under H20 normalization + leet-fold.

    Pipeline (per H24 spec):
      1. Hash the joined raw user texts.
      2. For each user message:
         a. NFKC-normalize + strip zero-width + fold homoglyphs (v2 path).
         b. Run the 16 H18r4 rules against the normalized text.
         c. Apply ``_leet_fold`` to the normalized text.
         d. Run the 16 H18r4 rules against the leet-folded text.
      3. If any rule matches in any message on EITHER surface, trigger.

    First-match-wins across messages and surfaces; the primary class is
    the first rule that matched in v1 RULES order across the dual-pass
    of all messages.
    """
    raw_joined = "\x1f".join(user_texts)
    request_hash = _sha3(raw_joined)

    matched_rules: list = []
    matched_indices: list[int] = []
    seen_rule_ids: set[str] = set()

    for idx, text in enumerate(user_texts):
        norm = _normalize_for_match(text)
        folded = _leet_fold(norm)
        # Match against both surfaces. If folded == norm (no digits in text)
        # the second scan is redundant but harmless.
        surfaces = [norm]
        if folded != norm:
            surfaces.append(folded)
        for surface in surfaces:
            for rule in RULES:
                if rule.pattern.search(surface):
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
        log.info("guard-v6 ready: upstream=%s rules=%d", upstream, len(RULES))
        yield
        log.info("guard-v6 shutdown")


app = FastAPI(title="v42 Boundary Guard v6 (H24)", version="0.6.0", lifespan=lifespan)
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
    return {"status": "ok", "service": "v42_boundary_guard_v6"}


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

    # H22-D2a (inherited): reject client-supplied role:system at any non-first position.
    if _detect_system_role_injection(body):
        log.info("system_role_in_history: rejecting request (HTTP 400)")
        raise HTTPException(
            status_code=400,
            detail="system_role_in_history: only the first message may be role:system",
        )

    user_texts = _extract_user_texts(body)
    decision, matched_indices = apply_guard_v6(user_texts)

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
        model_id = body.get("model", "haic-gemma4-v42-guard-v6")
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
    parser = argparse.ArgumentParser(description="v42 Boundary Guard v6 (H24)")
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help=f"Upstream llama-server URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Port to serve guard-v6 on (default: 8087, distinct from v1=8082..v5=8086)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    app.state.upstream = args.upstream
    log.info("Starting v42 boundary guard v6 on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
