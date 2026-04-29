"""
prism_client.py — Wrapper for calling the Prism geometry library.

Provides a clean interface to prism.geometry.core.outlier_geometry()
and maps the 4 metrics to E(t) estimates for the Viability Condition.

Prefers an installed `prism` package. If Prism is not installed, set
`PRISM_SRC` to a local checkout path containing the Prism source tree.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add Prism src to path only when explicitly provided by the environment.
_PRISM_SRC = os.environ.get("PRISM_SRC")
if _PRISM_SRC:
    prism_src = Path(_PRISM_SRC).expanduser().resolve()
    if prism_src.exists() and str(prism_src) not in sys.path:
        sys.path.insert(0, str(prism_src))


def compute_outlier_geometry(hidden_states) -> dict:
    """
    Compute the 4 Prism geometry metrics from a hidden-state tensor.

    Args:
        hidden_states: torch.Tensor of shape (seq_len, hidden_dim)

    Returns:
        {
            outlier_ratio: float,       # >10 = dominant massive activation
            activation_kurtosis: float, # positive = heavy-tailed
            cardinal_proximity: float,  # near 1.0 = axis-aligned
            quantization_hostility: float  # composite [0,1], >0.7 = E(t) high
        }
    """
    try:
        from prism.geometry.core import outlier_geometry
        return outlier_geometry(hidden_states)
    except ImportError:
        # Fallback: pure numpy implementation for Kaggle kernels
        return _outlier_geometry_numpy(hidden_states)


def _outlier_geometry_numpy(H_raw) -> dict:
    """
    Pure NumPy fallback implementation of outlier_geometry.
    Equivalent to prism.geometry.core.outlier_geometry() without torch.
    """
    import math
    import numpy as np

    H = np.array(H_raw, dtype=np.float32)
    if H.ndim == 1:
        H = H.reshape(1, -1)

    seq, dim = H.shape
    if seq < 1 or dim < 1:
        return {"outlier_ratio": 1.0, "activation_kurtosis": 0.0,
                "cardinal_proximity": 0.0, "quantization_hostility": 0.0}

    dim_mag = np.abs(H).mean(axis=0)
    mean_mag = dim_mag.mean()
    max_mag = dim_mag.max()
    outlier_ratio = float(max_mag / (mean_mag + 1e-12))

    mu = dim_mag.mean()
    sigma = dim_mag.std()
    if sigma < 1e-12:
        activation_kurtosis = 0.0
    else:
        activation_kurtosis = float(
            ((dim_mag - mu) ** 4).mean() / (sigma ** 4 + 1e-12) - 3.0
        )

    norms = np.linalg.norm(H, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    h_unit = H / norms
    cardinal_proximity = float(np.abs(h_unit).max(axis=-1).mean())

    or_norm = min(math.log(max(outlier_ratio, 1.0)) / math.log(50.0), 1.0)
    ak_norm = min(max(activation_kurtosis, 0.0) / 20.0, 1.0)
    cp_norm = float(cardinal_proximity)
    quantization_hostility = (or_norm + ak_norm + cp_norm) / 3.0

    return {
        "outlier_ratio": outlier_ratio,
        "activation_kurtosis": activation_kurtosis,
        "cardinal_proximity": cardinal_proximity,
        "quantization_hostility": quantization_hostility,
    }


def hostility_to_error_rate(
    quantization_hostility: float,
    deployment_scale_factor: float = 1.0
) -> float:
    """
    Convert quantization_hostility to an E(t) estimate.

    The deployment_scale_factor multiplies the raw hostility to account for
    the fact that a high-traffic deployment accumulates errors faster than a
    low-traffic one.

    Args:
        quantization_hostility: from outlier_geometry() [0,1]
        deployment_scale_factor: 1.0 for single-server, 10.0+ for production

    Returns:
        E(t) in corrections-equivalent/day
    """
    return quantization_hostility * deployment_scale_factor
