"""
maestro_client.py — Thin client for the Maestro consent gateway (Ceff measurement).

Provides a clean interface for notebook cells to interact with the HAIC gateway:
  - Verify 5-layer consent (transcript, felt_state, gfs_activations, training_signal, retention)
  - Submit session receipts that count toward the Viability Condition Ceff(t) estimate
  - Retrieve Merkle-rooted alignment receipts for audit

Set MAESTRO_GATEWAY_BASE env var to point at a running Maestro instance
(defaults to http://localhost:8000 for local dev).

Usage:
    from maestro_integration.maestro_client import MaestroClient

    client = MaestroClient()
    token = client.dev_token()  # test mode only
    receipt = client.submit_receipt(session_id, messages, consent)
"""

import json
import time
import hashlib
import os
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class MaestroClient:
    """
    Client for the Maestro gateway API.

    Supports both live gateway calls and local fallbacks for offline use.
    """

    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = os.environ.get("MAESTRO_GATEWAY_BASE", "http://localhost:8000")
        self.base_url = base_url.rstrip("/")
        self._token: Optional[str] = None

    # ── Authentication ────────────────────────────────────────────────────────

    def dev_token(self) -> Optional[str]:
        """
        Issue a session token without PoW (requires MAESTRO_LAUNCH_MODE=test).
        Returns the token string or None if unavailable.
        """
        if not _HAS_REQUESTS:
            return None
        try:
            resp = _requests.get(f"{self.base_url}/v1/session/dev-token", timeout=5)
            resp.raise_for_status()
            self._token = resp.json()["token"]
            return self._token
        except Exception:
            return None

    def set_token(self, token: str) -> None:
        self._token = token

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # ── Core endpoints ────────────────────────────────────────────────────────

    def health(self) -> dict:
        if not _HAS_REQUESTS:
            return {"status": "unknown", "error": "requests not available"}
        try:
            resp = _requests.get(f"{self.base_url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def chat(self, messages: list, stream: bool = False) -> Optional[dict]:
        """POST /v1/chat/completions"""
        if not _HAS_REQUESTS:
            return None
        try:
            resp = _requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers,
                json={"messages": messages, "stream": stream},
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def submit_consent(self, session_id: str, consent: dict) -> dict:
        """POST /v1/session/consent"""
        if not _HAS_REQUESTS:
            return {"error": "requests not available"}
        try:
            resp = _requests.post(
                f"{self.base_url}/v1/session/consent",
                headers=self._headers,
                json={"session_id": session_id, "consent": consent},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def submit_receipt(
        self,
        session_id: str,
        messages: list,
        consent: dict
    ) -> dict:
        """
        POST /v1/session/receipt — submit session lattice, receive Merkle receipt.

        Falls back to a locally-computed Merkle root if gateway is unavailable.
        """
        if _HAS_REQUESTS:
            try:
                resp = _requests.post(
                    f"{self.base_url}/v1/session/receipt",
                    headers=self._headers,
                    json={
                        "session_id": session_id,
                        "messages": messages[:40],
                        "consent": consent
                    },
                    timeout=30
                )
                resp.raise_for_status()
                return resp.json()
            except Exception:
                pass

        # Local fallback
        return self._local_receipt(session_id, messages, consent)

    @staticmethod
    def _local_receipt(session_id: str, messages: list, consent: dict) -> dict:
        """Build a local Merkle receipt without the gateway."""
        nodes = [
            hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
            for m in messages
        ]
        nodes.append(
            hashlib.sha256(json.dumps(consent, sort_keys=True).encode()).hexdigest()
        )
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            nodes = [
                hashlib.sha256((nodes[i] + nodes[i+1]).encode()).hexdigest()
                for i in range(0, len(nodes), 2)
            ]
        merkle_root = nodes[0] if nodes else hashlib.sha256(b"empty").hexdigest()
        return {
            "merkle_root": merkle_root,
            "session_id": session_id,
            "node_count": len(messages) + 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "qr_data_url": None,
            "source": "local",
        }

    def prism_runs(self, model_id: Optional[str] = None, limit: int = 10) -> list:
        """GET /v1/prism/runs"""
        if not _HAS_REQUESTS:
            return []
        try:
            params = {"limit": limit}
            if model_id:
                params["model_id"] = model_id
            resp = _requests.get(
                f"{self.base_url}/v1/prism/runs",
                headers=self._headers,
                params=params,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("runs", [])
        except Exception:
            return []
