"""
maestro_gateway/app.py — Reference Maestro gateway implementation.

Provides the endpoints maestro_client.py expects:
  GET  /health
  GET  /v1/session/dev-token         (test mode only)
  POST /v1/session/consent
  POST /v1/session/receipt
  POST /v1/chat/completions          (echo stub — wire to real LLM in prod)
  GET  /v1/prism/runs                (in-memory log)

Receipts are computed using the SAME utils/merkle.py the client uses for its
local fallback, so on-chain anchors verify regardless of which side built them.

Env vars:
  MAESTRO_LAUNCH_MODE   "test" enables /v1/session/dev-token (default "test")
  MAESTRO_DEV_TOKEN     fixed token returned by /v1/session/dev-token
  MAESTRO_RATE_LIMIT_S  min seconds between receipt submissions per token
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow `python -m maestro_gateway.app` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field

from utils.merkle import sha3_256_hex, merkle_root, hash_items_to_leaves


# ── config ───────────────────────────────────────────────────────────────────
MODE         = os.environ.get("MAESTRO_LAUNCH_MODE", "test")
DEV_TOKEN    = os.environ.get("MAESTRO_DEV_TOKEN", f"dev-{uuid.uuid4().hex[:16]}")
RATE_LIMIT_S = float(os.environ.get("MAESTRO_RATE_LIMIT_S", "0.5"))

app = FastAPI(
    title="Maestro Gateway (reference)",
    version="0.1.0",
    description="HAIC governance gateway — receipts, consent, PRISM logs.",
)

# In-memory state. Production should swap for Postgres + Redis.
_consents: dict[str, dict] = {}
_receipts: dict[str, dict] = {}
_prism_runs: list[dict] = []
_last_receipt_at: dict[str, float] = defaultdict(float)


# ── auth ─────────────────────────────────────────────────────────────────────
def auth(authorization: str | None = Header(default=None)) -> str:
    """Bearer-token auth. In test mode any token starting with 'dev-' works."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(None, 1)[1].strip()
    if MODE == "test":
        if token == DEV_TOKEN or token.startswith("dev-"):
            return token
    # Production would validate against a session store + PoW.
    raise HTTPException(status_code=401, detail="invalid token")


# ── schemas ──────────────────────────────────────────────────────────────────
class ConsentBody(BaseModel):
    session_id: str
    consent: dict[str, Any]

class ReceiptBody(BaseModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list, max_length=40)
    consent: dict[str, Any]

class ChatBody(BaseModel):
    messages: list[dict[str, Any]]
    stream: bool = False


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": MODE,
        "uptime_s": int(time.time() - _BOOT_AT),
        "receipts": len(_receipts),
    }


@app.get("/v1/session/dev-token")
def dev_token() -> dict:
    if MODE != "test":
        raise HTTPException(status_code=403, detail="dev-token disabled outside test mode")
    return {"token": DEV_TOKEN, "expires_in": 86400}


@app.post("/v1/session/consent")
def submit_consent(body: ConsentBody, _tok: str = Depends(auth)) -> dict:
    consent_hash = sha3_256_hex(json.dumps(body.consent, sort_keys=True))
    _consents[body.session_id] = {
        "consent": body.consent,
        "consent_hash": consent_hash,
        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return {
        "session_id": body.session_id,
        "consent_hash": consent_hash,
        "layers_granted": [k for k, v in body.consent.items() if v == "granted"],
    }


@app.post("/v1/session/receipt")
def submit_receipt(body: ReceiptBody, tok: str = Depends(auth)) -> dict:
    # Rate limit per token (CS5 defense).
    now = time.time()
    if now - _last_receipt_at[tok] < RATE_LIMIT_S:
        raise HTTPException(status_code=429, detail="rate limit")
    _last_receipt_at[tok] = now

    leaves = hash_items_to_leaves(body.messages)
    leaves.append(sha3_256_hex(json.dumps(body.consent, sort_keys=True)))
    root = merkle_root(leaves)

    receipt = {
        "merkle_root": root,
        "session_id": body.session_id,
        "node_count": len(body.messages) + 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qr_data_url": None,
        "source": "gateway",
    }
    _receipts[body.session_id] = receipt
    return receipt


@app.post("/v1/chat/completions")
def chat(body: ChatBody, _tok: str = Depends(auth)) -> dict:
    """Echo stub. Wire to vLLM / Gemini / Ollama in production."""
    if not body.messages:
        raise HTTPException(status_code=400, detail="empty messages")
    last = body.messages[-1].get("content", "")
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "maestro-stub",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"[echo] {last}"},
            "finish_reason": "stop",
        }],
    }


@app.get("/v1/prism/runs")
def prism_runs(model_id: str | None = None, limit: int = 10) -> dict:
    runs = _prism_runs
    if model_id:
        runs = [r for r in runs if r.get("model_id") == model_id]
    return {"runs": runs[-limit:]}


_BOOT_AT = time.time()
