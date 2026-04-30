"""Integration tests for maestro_gateway/app.py — uses FastAPI TestClient.

Skipped if fastapi is not installed (it's not a hard dep of the framework).
Verifies every endpoint AND cross-language Merkle parity vs. utils/merkle.py.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from maestro_gateway.app import app, DEV_TOKEN  # noqa: E402
from utils.merkle import sha3_256_hex, merkle_root, hash_items_to_leaves  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {DEV_TOKEN}"}


class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "uptime_s" in body and "receipts" in body


class TestDevToken:
    def test_dev_token_in_test_mode(self):
        r = client.get("/v1/session/dev-token")
        assert r.status_code == 200
        assert r.json()["token"]


class TestAuth:
    def test_consent_requires_token(self):
        r = client.post("/v1/session/consent",
                        json={"session_id": "s", "consent": {}})
        assert r.status_code == 401

    def test_consent_rejects_bad_token(self):
        r = client.post("/v1/session/consent",
                        headers={"Authorization": "Bearer junk"},
                        json={"session_id": "s", "consent": {}})
        assert r.status_code == 401

    def test_consent_accepts_dev_prefix(self):
        r = client.post("/v1/session/consent",
                        headers={"Authorization": "Bearer dev-anything"},
                        json={"session_id": "s", "consent": {"transcript": "granted"}})
        assert r.status_code == 200


class TestConsent:
    def test_consent_returns_hash_and_layers(self):
        body = {"session_id": "s1", "consent": {"transcript": "granted",
                                                 "lattice": "denied"}}
        r = client.post("/v1/session/consent", headers=AUTH, json=body)
        assert r.status_code == 200
        out = r.json()
        # Hash must match the same SHA3-256(json sorted) the client would compute.
        expected = sha3_256_hex(json.dumps(body["consent"], sort_keys=True))
        assert out["consent_hash"] == expected
        assert out["layers_granted"] == ["transcript"]


class TestReceipt:
    def test_receipt_matches_local_computation(self):
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        consent = {"transcript": "granted"}

        # Sleep past the rate limit window from prior tests.
        time.sleep(0.3)
        r = client.post("/v1/session/receipt", headers=AUTH,
                        json={"session_id": "rcpt1",
                              "messages": msgs, "consent": consent})
        assert r.status_code == 200, r.text
        receipt = r.json()
        assert receipt["source"] == "gateway"
        assert receipt["node_count"] == 6  # 5 msgs + consent leaf

        # Cross-check with utils/merkle.py — what the client falls back to.
        leaves = hash_items_to_leaves(msgs)
        leaves.append(sha3_256_hex(json.dumps(consent, sort_keys=True)))
        assert receipt["merkle_root"] == merkle_root(leaves)

    def test_receipt_rate_limit(self):
        body = {"session_id": "rl", "messages": [], "consent": {}}
        time.sleep(0.7)
        r1 = client.post("/v1/session/receipt", headers=AUTH, json=body)
        r2 = client.post("/v1/session/receipt", headers=AUTH, json=body)
        # First passes, second hits rate limit (default 0.5s).
        assert r1.status_code == 200
        assert r2.status_code == 429


class TestChat:
    def test_chat_echoes(self):
        r = client.post("/v1/chat/completions", headers=AUTH,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "[echo] hi"

    def test_chat_rejects_empty(self):
        r = client.post("/v1/chat/completions", headers=AUTH,
                        json={"messages": []})
        assert r.status_code == 400


class TestPrismRuns:
    def test_prism_runs_empty_default(self):
        r = client.get("/v1/prism/runs")
        assert r.status_code == 200
        assert "runs" in r.json()


class TestEndToEndCompatWithMaestroClient:
    """The client and the gateway MUST agree on the receipt format."""

    def test_client_against_in_process_gateway(self):
        from maestro_integration.maestro_client import MaestroClient
        # Point client at a fake URL — we'll inject the TestClient by
        # monkeypatching `requests`.
        msgs = [{"role": "user", "content": "x"}]
        consent = {"transcript": "granted"}

        # Compute what the gateway WOULD return (via direct call):
        time.sleep(0.6)
        r = client.post("/v1/session/receipt", headers=AUTH,
                        json={"session_id": "compat",
                              "messages": msgs, "consent": consent})
        gateway_root = r.json()["merkle_root"]

        # Compute what the client's local fallback returns:
        local = MaestroClient._local_receipt("compat", msgs, consent)

        assert gateway_root == local["merkle_root"], \
            "client local fallback and gateway must produce identical roots"
