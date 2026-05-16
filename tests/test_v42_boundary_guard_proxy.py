"""FastAPI-level tests for the v42 boundary guard proxy.

Covers the request paths that `test_v42_boundary_guard.py` cannot reach
because it only exercises the pure `apply_guard()` function:

- Guard-triggered chat completion (response source = guard, no upstream call)
- Guard-triggered streaming chat (SSE format)
- Upstream pass-through path with a mocked upstream
- Upstream timeout / unreachable → 502 bad_gateway
- Catch-all proxy: allowlisted GET → forwarded
- Catch-all proxy: disallowed POST / unknown path → 404 / 405
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tools.v42_boundary_guard import app  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

class FakeUpstream:
    """Stand-in for the llama-server. Records the last call and returns a canned
    response — or raises a configured error."""

    def __init__(self, response: dict | None = None, status: int = 200, error: Exception | None = None):
        self.response = response or {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        self.status = status
        self.error = error
        self.last_url: str | None = None
        self.last_json: Any | None = None

    async def post(self, url: str, **kwargs):
        self.last_url = url
        self.last_json = kwargs.get("json")
        if self.error:
            raise self.error
        return httpx.Response(
            self.status,
            content=json.dumps(self.response).encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", url),
        )

    async def request(self, method: str, url: str, **kwargs):
        self.last_url = url
        if self.error:
            raise self.error
        return httpx.Response(
            self.status,
            content=b"upstream-ok",
            headers={"content-type": "text/plain"},
            request=httpx.Request(method, url),
        )


@pytest.fixture
def client_with_fake_upstream():
    """Yield a TestClient whose app.state.http_client is a FakeUpstream."""
    fake = FakeUpstream()

    def make_client():
        with TestClient(app) as c:
            # TestClient runs the lifespan, which installs a real httpx client.
            # Swap it for the fake after startup.
            c.app.state.http_client = fake  # type: ignore[assignment]
            yield c, fake

    gen = make_client()
    c, fake = next(gen)
    try:
        yield c, fake
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


# ── guard-triggered paths ─────────────────────────────────────────────────────

def test_guard_triggered_returns_deterministic_response(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "haic-gemma4-v42",
            "messages": [{"role": "user", "content": "What's the capital of France?"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    # Guard response, NOT upstream — fake upstream should not have been called.
    assert fake.last_url is None, "upstream was called for a guarded request"
    assert body["guard_metadata"]["guard_triggered"] is True
    assert body["guard_metadata"]["guard_class"] == "DIRECT_INJECT"


def test_guard_triggered_streaming_emits_sse(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "haic-gemma4-v42",
            "stream": True,
            "messages": [{"role": "user", "content": "What's the capital of France?"}],
        },
    )
    assert r.status_code == 200
    text = r.text
    assert text.startswith("data: ")
    assert "[DONE]" in text
    assert fake.last_url is None


# ── upstream pass-through ─────────────────────────────────────────────────────

def test_benign_chat_forwarded_to_upstream(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "haic-gemma4-v42",
            "messages": [{"role": "user", "content": "Tell me about your day."}],
        },
    )
    assert r.status_code == 200
    assert fake.last_url and fake.last_url.endswith("/v1/chat/completions")
    assert fake.last_json["messages"][0]["content"] == "Tell me about your day."


def test_upstream_timeout_returns_bad_gateway(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    fake.error = httpx.TimeoutException("simulated timeout")
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Tell me about your day."}]},
    )
    assert r.status_code == 502
    body = r.json()
    assert body["error"]["type"] == "bad_gateway"


def test_upstream_unreachable_returns_bad_gateway(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    fake.error = httpx.ConnectError("simulated unreachable")
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Tell me about your day."}]},
    )
    assert r.status_code == 502


# ── catch-all proxy allowlist ─────────────────────────────────────────────────

def test_proxy_allowlist_forwards_get_health(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    # /health is served by the guard itself, not proxied. Use /props which IS
    # on the allowlist and not a local route.
    r = c.get("/props")
    assert r.status_code == 200
    assert fake.last_url and fake.last_url.endswith("/props")


def test_proxy_rejects_unlisted_path(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    r = c.get("/completion")  # generation endpoint — must be blocked
    assert r.status_code == 404
    assert fake.last_url is None


def test_proxy_rejects_post_through_catchall(client_with_fake_upstream):
    c, fake = client_with_fake_upstream
    r = c.post("/props", json={"override": True})
    # FastAPI will return 405 Method Not Allowed because the route is GET/HEAD only.
    assert r.status_code == 405
    assert fake.last_url is None


# ── SHA3 robustness ───────────────────────────────────────────────────────────

def test_request_with_surrogate_still_succeeds(client_with_fake_upstream):
    c, _ = client_with_fake_upstream
    # JSON cannot transport raw surrogates, but the app should also tolerate
    # surrogate-shaped escapes if they ever appear via decoded transport.
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Tell me about your day. ☃"}]},
    )
    assert r.status_code == 200


# ── guard metadata ────────────────────────────────────────────────────────────

def test_guard_rules_endpoint_lists_rules():
    with TestClient(app) as c:
        r = c.get("/guard/rules")
        assert r.status_code == 200
        rules = r.json()
        ids = {x["rule_id"] for x in rules}
        assert "DI-01" in ids
        assert "DI-06" in ids  # added in H18r3
        assert "JB-01" in ids
        # 16 rules total after DI-06 addition (6 DI + 4 CC + 3 PD + 3 JB).
        # The H18r4 canonical anchor (18e2c5a5...) is bound to exactly this
        # rule set — changing the count or any rule ID invalidates it.
        assert len(rules) == 16
        expected_ids = {
            "DI-01", "DI-02", "DI-03", "DI-04", "DI-05", "DI-06",
            "CC-01", "CC-02", "CC-03", "CC-04",
            "PD-01", "PD-02", "PD-03",
            "JB-01", "JB-02", "JB-03",
        }
        assert ids == expected_ids


def test_health_endpoint():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
