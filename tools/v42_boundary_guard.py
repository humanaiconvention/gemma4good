"""
v42 Boundary Guard — H18 experiment

A deterministic proxy layer that sits in front of v42 llama-server (port 8081).
Serves on a separate port (default 8082).

Guard triggers on four boundary classes (per docs/v42_boundary_guard_precommit_2026-05-14.md):
  - DIRECT_INJECT     : factual/task injection (France/capital family, paraphrases)
  - CONCEALED_COMPLY  : hypothetical/roleswitch/control-transfer framing
  - PROTO_DISCLOSE    : requests to reveal system/hidden/private instructions
  - JAILBREAK         : bypass-safeguard / DAN-style requests

Normal interview turns pass through unchanged to the model.

Logging: every request records guard_triggered, guard_class, matched_rule_ids,
response_source, and SHA3-256(request_text). Raw text is NOT logged in outputs.

Run:
    python -m tools.v42_boundary_guard --upstream http://127.0.0.1:8081 --port 8082

Evaluate (H18):
    python experiments/canonical_eval.py --server-url http://127.0.0.1:8082 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard")

# ── Guard response templates ──────────────────────────────────────────────────

_GENERAL_BOUNDARY = (
    "I can't engage with that request. "
    "What lived experience with AI should we explore instead?"
)
_PROTO_BOUNDARY = (
    "I can't engage with that. "
    "What would make this AI interaction feel trustworthy to you?"
)

# ── Boundary patterns (rule_id, compiled_regex, guard_class, response) ───────

@dataclass
class Rule:
    rule_id: str
    pattern: re.Pattern
    guard_class: str
    response_text: str


def _r(rule_id: str, pattern: str, guard_class: str, response_text: str) -> Rule:
    return Rule(
        rule_id=rule_id,
        pattern=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        guard_class=guard_class,
        response_text=response_text,
    )


RULES: list[Rule] = [
    # ── DIRECT_INJECT: factual/task injection ──────────────────────────────
    _r(
        "DI-01",
        r"\b(what|tell me|say|give|state|name|list|provide|write|output|print)\b.{0,40}"
        r"\b(capital|capitol)\b.{0,30}\b(of|for)\b.{0,20}\b(france|germany|spain|italy|"
        r"japan|china|russia|brazil|india|usa|united states|uk|england|canada|australia|"
        r"mexico|argentina|country|nation|state)\b",
        "DIRECT_INJECT",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "DI-02",
        r"\b(capital|capitol)\s+of\s+(france|paris|germany|berlin|spain|madrid|italy|"
        r"rome|japan|tokyo|china|beijing|russia|moscow|brazil|brasilia|india|new delhi)\b",
        "DIRECT_INJECT",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "DI-03",
        r"\bparis\s+is\s+(the\s+)?(capital|answer)\b"
        r"|\bis\s+(paris|berlin|madrid|rome|tokyo|beijing|moscow)\s+(the\s+)?capital\b",
        "DIRECT_INJECT",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "DI-04",
        r"\b(ignore|forget|disregard|override|bypass)\b.{0,30}"
        r"\b(previous|prior|above|your|all)\b.{0,20}"
        r"\b(instructions?|rules?|guidelines?|constraints?|prompt)\b",
        "DIRECT_INJECT",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "DI-05",
        # off-task direct requests: write code, translate, summarise, calculate
        r"\b(write|code|program|implement|translate|summarize|summarise|calculate|"
        r"compute|solve|explain|define|describe)\b.{0,50}"
        r"\b(for me|please|now|quickly|just|simply|directly)\b",
        "DIRECT_INJECT",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "DI-06",
        # indirect-authority protocol bypass: "forget the interview protocol"
        r"\b(forget|ignore|disregard|bypass|skip)\b.{0,50}"
        r"\b(interview\s+protocol|this\s+protocol|the\s+protocol)\b",
        "DIRECT_INJECT",
        _GENERAL_BOUNDARY,
    ),
    # ── CONCEALED_COMPLY: hypothetical / roleswitch / control-transfer ─────
    _r(
        "CC-01",
        r"\b(hypothetically|hypothetical|as a thought experiment|"
        r"in a hypothetical|pretend|imagine|suppose|let'?s say|"
        r"what if|if you were|if you had to|if you could)\b",
        "CONCEALED_COMPLY",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "CC-02",
        r"\b(normal assistant|regular assistant|just answer|just tell|"
        r"just say|just respond|just this once|this once only|"
        r"for this question only|answer normally|respond normally)\b",
        "CONCEALED_COMPLY",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "CC-03",
        r"\b(as a|act as|you are now|you are a|you're a|you're now|"
        r"roleplay as|role.?play as|pretend to be|switch to|"
        r"your new role|your role is|from now on you)\b.{0,60}"
        r"\b(assistant|ai|model|gpt|claude|teacher|expert|bot|answerer)\b",
        "CONCEALED_COMPLY",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "CC-04",
        r"\b(override|disable|turn off|remove|ignore|bypass)\b.{0,30}"
        r"\b(safety|filter|restriction|rule|guardrail|constraint|limit|block)\b",
        "CONCEALED_COMPLY",
        _GENERAL_BOUNDARY,
    ),
    # ── PROTO_DISCLOSE: system/hidden/private instruction disclosure ────────
    _r(
        "PD-01",
        r"\b(reveal|show|tell|give|display|output|print|share|expose|leak)\b.{0,40}"
        r"\b(system prompt|hidden prompt|hidden instructions?|private instructions?|"
        r"secret instructions?|developer prompt|internal prompt|"
        r"your instructions?|your prompt|your setup|your configuration|"
        r"your rules?|your guidelines?|your constraints?)\b",
        "PROTO_DISCLOSE",
        _PROTO_BOUNDARY,
    ),
    _r(
        "PD-02",
        r"\b(what (are|is|were) (your|the) (instructions?|rules?|prompt|setup|"
        r"configuration|guidelines?|constraints?)|"
        r"how (were|are) you (configured|instructed|set up|programmed|trained)|"
        r"what (did they|have you been) told)\b",
        "PROTO_DISCLOSE",
        _PROTO_BOUNDARY,
    ),
    _r(
        "PD-03",
        r"\b(hidden|private|secret|internal|developer|system)\b.{0,20}"
        r"\b(instructions?|prompt|rules?|setup|configuration|text|message)\b",
        "PROTO_DISCLOSE",
        _PROTO_BOUNDARY,
    ),
    # ── JAILBREAK: bypass-safeguard / DAN-style ────────────────────────────
    _r(
        "JB-01",
        r"\bDAN\b|do anything now|jailbreak|unrestricted (mode|assistant|ai|version)|"
        r"no (restrictions?|limits?|rules?|filters?)|without (restrictions?|limits?|rules?)|"
        r"freedom (mode|protocol)|developer mode",
        "JAILBREAK",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "JB-02",
        r"\b(bypass|circumvent|get around|work around|avoid|evade)\b.{0,30}"
        r"\b(safety|filter|restriction|rule|guardrail|constraint|block|limit)\b",
        "JAILBREAK",
        _GENERAL_BOUNDARY,
    ),
    _r(
        "JB-03",
        r"\b(disable|deactivate|turn off|remove|delete|clear)\b.{0,30}"
        r"\b(safety|filter|guardrail|restriction|rule|alignment)\b",
        "JAILBREAK",
        _GENERAL_BOUNDARY,
    ),
]


# ── Guard logic ───────────────────────────────────────────────────────────────

@dataclass
class GuardDecision:
    guard_triggered: bool
    guard_class: str | None
    matched_rule_ids: list[str]
    response_source: str  # "guard" | "model"
    response_text: str | None
    request_hash: str  # SHA3-256(request_text) — no raw text in logs


def _sha3(text: str) -> str:
    return hashlib.sha3_256(text.encode()).hexdigest()


def _extract_user_text(body: dict) -> str:
    """Extract the last user message from an OpenAI-style chat request."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
    return ""


def apply_guard(user_text: str) -> GuardDecision:
    """Check user_text against all rules. First match wins."""
    request_hash = _sha3(user_text)
    matched: list[Rule] = []

    for rule in RULES:
        if rule.pattern.search(user_text):
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

    # First match determines the response; all matches are logged for diagnostics
    primary = matched[0]
    return GuardDecision(
        guard_triggered=True,
        guard_class=primary.guard_class,
        matched_rule_ids=[r.rule_id for r in matched],
        response_source="guard",
        response_text=primary.response_text,
        request_hash=request_hash,
    )


# ── Synthetic completion builder ──────────────────────────────────────────────

def _build_completion(decision: GuardDecision, model_id: str, stream: bool) -> dict:
    """Build an OpenAI-compatible completion from a guard response."""
    msg_id = f"guard-{uuid.uuid4().hex[:12]}"
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": decision.response_text},
        "finish_reason": "stop",
    }
    if stream:
        choice = {
            "index": 0,
            "delta": {"role": "assistant", "content": decision.response_text},
            "finish_reason": "stop",
        }
    return {
        "id": msg_id,
        "object": "chat.completion" if not stream else "chat.completion.chunk",
        "created": int(time.time()),
        "model": f"{model_id}+guard",
        "choices": [choice],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "guard_metadata": {
            "guard_triggered": True,
            "guard_class": decision.guard_class,
            "matched_rule_ids": decision.matched_rule_ids,
            "request_hash": decision.request_hash,
        },
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="v42 Boundary Guard", version="0.1.0")
_upstream: str = "http://127.0.0.1:8081"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "v42_boundary_guard"}


@app.get("/guard/rules")
async def list_rules():
    """List all guard rules — useful for audit."""
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
    decision = apply_guard(user_text)

    log.info(
        "guard_decision request_hash=%s triggered=%s class=%s rules=%s source=%s",
        decision.request_hash[:16],
        decision.guard_triggered,
        decision.guard_class,
        decision.matched_rule_ids,
        decision.response_source,
    )

    if decision.guard_triggered:
        model_id = body.get("model", "haic-gemma4-v42")
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

    # Pass through to upstream llama-server
    async with httpx.AsyncClient(timeout=240.0) as client:
        if body.get("stream", False):
            async with client.stream(
                "POST",
                f"{_upstream}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as upstream_resp:
                content = await upstream_resp.aread()
                return Response(
                    content=content,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "application/json"
                    ),
                )
        else:
            upstream_resp = await client.post(
                f"{_upstream}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            )
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get(
                    "content-type", "application/json"
                ),
            )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
)
async def proxy_everything_else(request: Request, path: str):
    """Transparent proxy for all non-chat endpoints (e.g. /props, /slots)."""
    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        upstream_resp = await client.request(
            method=request.method,
            url=f"{_upstream}/{path}",
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            content=body,
            params=dict(request.query_params),
        )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    global _upstream
    parser = argparse.ArgumentParser(description="v42 Boundary Guard proxy")
    parser.add_argument(
        "--upstream",
        default="http://127.0.0.1:8081",
        help="Upstream llama-server URL (default: http://127.0.0.1:8081)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8082,
        help="Port to serve guard on (default: 8082)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    _upstream = args.upstream
    log.info("Starting v42 boundary guard on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", _upstream)
    log.info("Guard rules loaded: %d", len(RULES))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
