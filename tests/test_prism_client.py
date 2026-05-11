"""
tests/test_prism_client.py — Unit tests for prism_integration/prism_client.py.

Covers:
  - _outlier_geometry_numpy() with various input shapes and edge cases.
  - hostility_to_error_rate() scaling.
  - compute_outlier_geometry() fallback behavior.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from prism_integration.prism_client import (
    _outlier_geometry_numpy,
    hostility_to_error_rate,
    compute_outlier_geometry
)

class TestOutlierGeometryNumpy:
    def test_identity_matrix(self):
        """Identity matrix has uniform magnitudes and axis-aligned vectors."""
        H = np.eye(5)
        res = _outlier_geometry_numpy(H)
        assert res["outlier_ratio"] == pytest.approx(1.0, abs=1e-4)
        assert res["activation_kurtosis"] < 1.0  # Actually negative/small
        assert res["cardinal_proximity"] == pytest.approx(1.0, abs=1e-4)
        assert 0.0 <= res["quantization_hostility"] <= 1.0

    def test_single_spike(self):
        """A single feature with huge activation should spike outlier_ratio."""
        H = np.ones((10, 5))
        H[:, 2] = 1000.0  # Feature 2 is a massive outlier
        res = _outlier_geometry_numpy(H)
        
        # mean is roughly (1+1+1000+1+1)/5 = 200.8
        # max is 1000
        # ratio is roughly 1000 / 200.8 = ~4.98
        assert res["outlier_ratio"] > 4.5
        assert res["activation_kurtosis"] > 0.0

    def test_zero_matrix(self):
        """Zero matrix should not crash (divide by zero)."""
        H = np.zeros((5, 5))
        res = _outlier_geometry_numpy(H)
        # fallback to 1e-12 handles divide by zero, ratio = 0 / 1e-12 = 0
        assert res["outlier_ratio"] == 0.0
        assert res["activation_kurtosis"] == 0.0
        assert res["cardinal_proximity"] == 0.0
        assert res["quantization_hostility"] == 0.0

    def test_1d_input(self):
        """1D input should be reshaped automatically."""
        H = np.array([1.0, 2.0, 3.0])
        res = _outlier_geometry_numpy(H)
        assert res["outlier_ratio"] > 0

    def test_empty_input(self):
        """Empty input should return defaults gracefully."""
        H = np.zeros((0, 5))
        res = _outlier_geometry_numpy(H)
        assert res["outlier_ratio"] == 1.0
        assert res["quantization_hostility"] == 0.0

class TestHostilityToErrorRate:
    def test_passthrough(self):
        assert hostility_to_error_rate(0.5, 1.0) == 0.5
        
    def test_multiplier(self):
        assert hostility_to_error_rate(0.5, 10.0) == 5.0

class TestComputeOutlierGeometry:
    @patch.dict('sys.modules', {'prism.geometry.core': None})
    def test_fallback_when_prism_missing(self):
        """When prism module doesn't exist, compute_outlier_geometry uses numpy fallback."""
        H = np.eye(3)
        # With prism missing, it will raise ImportError and use numpy fallback
        res = compute_outlier_geometry(H)
        assert "outlier_ratio" in res
        assert "quantization_hostility" in res

    @patch.dict('sys.modules', {})
    def test_uses_prism_if_available(self):
        """When prism is installed, it uses the module instead of the fallback."""
        mock_prism = MagicMock()
        mock_outlier = MagicMock(return_value={"outlier_ratio": 99.0})
        mock_prism.geometry.core.outlier_geometry = mock_outlier
        
        with patch.dict('sys.modules', {'prism': mock_prism, 'prism.geometry': mock_prism, 'prism.geometry.core': mock_prism.geometry.core}):
            res = compute_outlier_geometry(np.eye(3))
            assert res["outlier_ratio"] == 99.0
            mock_outlier.assert_called_once()
