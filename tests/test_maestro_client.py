"""
tests/test_maestro_client.py — Unit tests for maestro_integration/maestro_client.py.

Covers:
  - _local_receipt() Merkle generation and determinism
  - Error handling when requests is unavailable
  - submit_receipt fallback logic
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from maestro_integration.maestro_client import MaestroClient


class TestMaestroClientLocalReceipt:
    def test_local_receipt_determinism(self):
        """_local_receipt must be deterministic for identical inputs."""
        messages = [{"role": "user", "content": "hello"}]
        consent = {"transcript": "granted"}
        
        client = MaestroClient("test-gateway")
        r1 = client._local_receipt("sess-1", messages, consent)
        r2 = client._local_receipt("sess-1", messages, consent)
        
        assert r1["merkle_root"] == r2["merkle_root"]
        assert r1["session_id"] == "sess-1"
        assert r1["source"] == "local"

    def test_local_receipt_cross_module_consistency(self):
        """Cross-module Merkle consistency is handled in test_merkle.py, 
        but verify basic output structure here."""
        messages = [{"role": "user", "content": "hello"}]
        consent = {"transcript": "granted"}
        
        client = MaestroClient("test-gateway")
        r = client._local_receipt("sess-1", messages, consent)
        
        assert "merkle_root" in r
        assert "node_count" in r
        assert r["node_count"] == 2 # 1 msg + 1 consent
        assert "created_at" in r


class TestMaestroClientWithoutRequests:
    """Test the module's behavior when 'requests' is unavailable."""

    @pytest.fixture
    def no_requests_client(self):
        # We can simulate _HAS_REQUESTS = False by mocking the class attribute
        # or temporarily patching the global variable
        with patch("maestro_integration.maestro_client._HAS_REQUESTS", False):
            yield MaestroClient("test-gateway")

    def test_health_unavailable(self, no_requests_client):
        res = no_requests_client.health()
        assert res["status"] == "unknown"
        assert "requests not available" in res["error"]

    def test_dev_token_unavailable(self, no_requests_client):
        assert no_requests_client.dev_token() is None

    def test_chat_unavailable(self, no_requests_client):
        assert no_requests_client.chat("test", [{"role": "user", "content": "hi"}]) is None

    def test_submit_receipt_unavailable_uses_fallback(self, no_requests_client):
        res = no_requests_client.submit_receipt("sess-1", [], {})
        assert res["source"] == "local"


class TestMaestroClientWithRequests:
    """Test the module's behavior when 'requests' is available but fails."""

    @patch("maestro_integration.maestro_client._HAS_REQUESTS", True)
    @patch("maestro_integration.maestro_client._requests")
    def test_health_connection_error(self, mock_requests):
        mock_requests.get.side_effect = Exception("Connection refused")
        client = MaestroClient("test-gateway")
        res = client.health()
        assert res["status"] == "unreachable"
        assert "Connection refused" in res["error"]

    @patch("maestro_integration.maestro_client._HAS_REQUESTS", True)
    @patch("maestro_integration.maestro_client._requests")
    def test_submit_receipt_connection_error(self, mock_requests):
        mock_requests.post.side_effect = Exception("Connection refused")
        client = MaestroClient("test-gateway")
        
        # Connection error should trigger local fallback
        res = client.submit_receipt("sess-1", [], {})
        assert res["source"] == "local"
