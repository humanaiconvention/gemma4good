"""
viability_condition.py — Standalone implementation of the Viability Condition.

Ceff(t) > E(t): corrective bandwidth must exceed error rate.

See docs/viability_condition.md for theoretical framework.
DOI: 10.5281/zenodo.18144681

This module is importable independently of the rest of the HAIC stack.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ViabilityAssessment:
    """Result of a Viability Condition evaluation."""
    viability_satisfied: bool
    ceff_vs_e_ratio: float
    autophagy_risk: str   # "none" | "low" | "medium" | "high" | "critical"
    temporal_signature_detected: bool
    scaling_recommendation: str
    effective_ceff: float
    error_rate: float
    synthetic_data_ratio: float
    model_id: Optional[str] = None
    prism_hostility: Optional[float] = None

    def __repr__(self) -> str:
        status = "VIABLE" if self.viability_satisfied else f"VIOLATED [{self.autophagy_risk.upper()} RISK]"
        return (
            f"ViabilityAssessment({status}, "
            f"Ceff/E={self.ceff_vs_e_ratio:.3f}, "
            f"autophagy_risk={self.autophagy_risk!r})"
        )


def assess(
    error_rate_estimate: float,
    verification_bandwidth_estimate: float,
    synthetic_data_ratio: float,
    model_id: Optional[str] = None,
    prism_hostility: Optional[float] = None,
) -> ViabilityAssessment:
    """
    Evaluate the Viability Condition Ceff(t) > E(t).

    Args:
        error_rate_estimate: E(t) in corrections-equivalent/day.
            Can be derived from Prism's quantization_hostility:
            E(t) = quantization_hostility * deployment_scale_factor
        verification_bandwidth_estimate: Ceff(t) in verified corrections/day.
            Each Maestro interview session with training_signal=granted counts as 1.
        synthetic_data_ratio: Fraction [0,1] of training data that is synthetic.
            Reduces effective Ceff by this factor (contaminated corrections are
            lower value than ground-truth corrections).
        model_id: Optional model identifier for reporting.
        prism_hostility: Optional quantization_hostility from Prism (cross-check
            against error_rate_estimate).

    Returns:
        ViabilityAssessment with full diagnosis.
    """
    effective_ceff = verification_bandwidth_estimate * (1.0 - synthetic_data_ratio)
    ratio = effective_ceff / max(error_rate_estimate, 1e-9)
    viability_satisfied = ratio > 1.0

    if ratio > 2.0:
        autophagy_risk = "none"
    elif ratio > 1.0:
        autophagy_risk = "low"
    elif ratio > 0.7:
        autophagy_risk = "medium"
    elif ratio > 0.3:
        autophagy_risk = "high"
    else:
        autophagy_risk = "critical"

    # Temporal signature: OOD accuracy degrading before val perplexity rises.
    # Approximated as: condition violated AND high synthetic ratio.
    temporal_signature_detected = (not viability_satisfied) and (synthetic_data_ratio > 0.3)

    # Cross-check Prism hostility
    if prism_hostility is not None and prism_hostility > error_rate_estimate:
        note = (
            f" NOTE: Prism hostility ({prism_hostility:.3f}) exceeds your "
            f"error_rate_estimate ({error_rate_estimate:.3f}). "
            f"Consider using hostility as a lower bound for E(t)."
        )
    else:
        note = ""

    if viability_satisfied and ratio > 2.0:
        rec = (
            f"Viable. Ceff/E={ratio:.2f}. "
            f"Safe to scale synthetic data up to {ratio:.1f}x before "
            f"verification infrastructure must also scale." + note
        )
    elif viability_satisfied:
        rec = (
            f"Marginally viable. Ceff/E={ratio:.2f}. "
            f"Do not increase synthetic_data_ratio without proportionally "
            f"increasing Maestro session throughput." + note
        )
    elif autophagy_risk == "medium":
        rec = (
            f"Condition violated. Ceff/E={ratio:.2f}. "
            f"Reduce synthetic_data_ratio or increase verified session count. "
            f"Monitor OOD accuracy as leading indicator (temporal signature)." + note
        )
    else:
        rec = (
            f"CRITICAL. Ceff/E={ratio:.2f}. "
            f"Informational autophagy likely. "
            f"Freeze synthetic data ingestion and audit grounding pipeline." + note
        )

    return ViabilityAssessment(
        viability_satisfied=viability_satisfied,
        ceff_vs_e_ratio=round(ratio, 6),
        autophagy_risk=autophagy_risk,
        temporal_signature_detected=temporal_signature_detected,
        scaling_recommendation=rec,
        effective_ceff=round(effective_ceff, 4),
        error_rate=error_rate_estimate,
        synthetic_data_ratio=synthetic_data_ratio,
        model_id=model_id,
        prism_hostility=prism_hostility,
    )


def from_prism_metrics(
    outlier_ratio: float,
    activation_kurtosis: float,
    cardinal_proximity: float,
    quantization_hostility: float,
    sessions_per_day: float,
    avg_turns_per_session: float = 6.0,
    consent_grant_rate: float = 0.85,
    synthetic_data_ratio: float = 0.0,
    deployment_scale_factor: float = 1.0,
    model_id: Optional[str] = None,
) -> ViabilityAssessment:
    """
    Convenience constructor that derives E(t) from Prism geometry metrics
    and Ceff(t) from Maestro session throughput.

    E(t) = quantization_hostility * deployment_scale_factor
    Ceff(t) = sessions_per_day * avg_turns * consent_grant_rate

    Args:
        outlier_ratio, activation_kurtosis, cardinal_proximity,
        quantization_hostility: from prism.geometry.core.outlier_geometry()
        sessions_per_day: Maestro verified sessions per day
        avg_turns_per_session: average interview turns (default 6)
        consent_grant_rate: fraction of sessions with training_signal=granted
        synthetic_data_ratio: fraction of training data that is synthetic
        deployment_scale_factor: multiplier for E(t) based on deployment scale
            (e.g. 1.0 = single-server; 10.0 = high-traffic production)
        model_id: optional model identifier
    """
    e_t = quantization_hostility * deployment_scale_factor
    ceff_t = sessions_per_day * avg_turns_per_session * consent_grant_rate

    return assess(
        error_rate_estimate=e_t,
        verification_bandwidth_estimate=ceff_t,
        synthetic_data_ratio=synthetic_data_ratio,
        model_id=model_id,
        prism_hostility=quantization_hostility,
    )
