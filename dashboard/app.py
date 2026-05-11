import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from viability.grounding_tracker import GroundingTracker

st.set_page_config(
    page_title="HAIC Viability Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("HAIC Governance Viability Dashboard")
st.markdown("Monitoring the non-negotiable viability constraint: **$C_{eff}(t) > E(t)$**")

# ── File loader ───────────────────────────────────────────────────────────────

default_tracker_path = os.path.join(_REPO_ROOT, "tracker.json")

def load_tracker(file_or_path):
    tracker = GroundingTracker()
    try:
        if isinstance(file_or_path, str):
            if os.path.exists(file_or_path):
                tracker.from_json(file_or_path)
            else:
                st.warning(f"No tracker file found at {file_or_path}")
        else:
            data = json.load(file_or_path)
            tracker.sessions = data.get("sessions", [])
            tracker.history = data.get("history", [])
            tracker.model_id = data.get("model_id", "unknown")
            tracker.created_at = data.get("created_at", "")
    except Exception as e:
        st.error(f"Error loading tracker: {e}")
    return tracker


def detect_file_type(data: dict) -> str:
    """Return 'tracker' or 'tier3' based on top-level keys."""
    if "tier" in data and data.get("tier") == 3:
        return "tier3"
    if "history" in data or "sessions" in data:
        return "tracker"
    return "unknown"


# ── Sidebar: upload ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader(
        "Upload tracker.json or Tier 3 results JSON",
        type=["json"],
        help="tracker.json from GroundingTracker.to_json() or haic_governance_tier3_results.json from Kaggle.",
    )

# ── Main content ──────────────────────────────────────────────────────────────

if uploaded_file is not None:
    raw = json.load(uploaded_file)
    file_type = detect_file_type(raw)
else:
    # Try to auto-detect a local file
    tier3_path = os.path.join(_REPO_ROOT, "onchain", "live_roundtrip_result.json")
    if os.path.exists(default_tracker_path):
        raw = json.loads(open(default_tracker_path).read()) if os.path.exists(default_tracker_path) else {}
        file_type = "tracker" if raw else "none"
    else:
        raw = {}
        file_type = "none"

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER VIEW
# ══════════════════════════════════════════════════════════════════════════════

if file_type == "tracker":
    tracker = load_tracker(uploaded_file if uploaded_file else default_tracker_path)

    if tracker and tracker.history:
        st.header(f"Model: {tracker.model_id}")

        summary = tracker.summary()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", summary["total_sessions"])
        with col2:
            st.metric("Total Corrections", summary["total_corrections_collected"])
        with col3:
            st.metric("Avg Error Rate E(t)", f"{summary['avg_error_rate']:.2f}")
        with col4:
            st.metric("Avg Effective Ceff(t)", f"{summary['avg_effective_ceff']:.2f}")

        st.subheader("Viability Condition Trajectory")
        df = pd.DataFrame(tracker.history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["effective_ceff"],
                                 mode="lines+markers", name="Ceff(t)"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["error_rate"],
                                 mode="lines+markers", name="E(t)"))
        fig.update_layout(title="Correction Bandwidth vs Error Accumulation",
                          xaxis_title="Time", yaxis_title="Rate (events/day)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Representation Autophagy Risk")
        risk_mapping = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        df["risk_level"] = df["autophagy_risk"].map(risk_mapping)
        fig2 = px.line(df, x="timestamp", y="risk_level",
                       title="Autophagy Risk Over Time", markers=True)
        fig2.update_yaxes(tickvals=[0, 1, 2, 3, 4],
                          ticktext=["None", "Low", "Medium", "High", "Critical"])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recent Sessions")
        sessions_df = pd.DataFrame(tracker.sessions)
        if not sessions_df.empty:
            display_cols = ["session_id", "timestamp", "corrections",
                            "consent_valid", "viability_satisfied"]
            available_cols = [c for c in display_cols if c in sessions_df.columns]
            st.dataframe(sessions_df[available_cols].tail(10))
    else:
        st.info("No grounding history available. Upload a tracker.json to view metrics.")


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 RESULTS VIEW
# ══════════════════════════════════════════════════════════════════════════════

elif file_type == "tier3":
    d = raw
    st.header("Tier 3 Governance Run Results")

    prism   = d.get("prism", {})
    sgt     = d.get("sgt", {})
    maestro = d.get("maestro", {})
    vc      = d.get("viability", {})
    passed  = d.get("promotion", False)

    # ── Promotion gate banner ─────────────────────────────────────────────────
    if passed:
        st.success("**PROMOTION GATE: PASS ✓**")
    else:
        st.error("**PROMOTION GATE: FAIL ✗** — see metrics below for diagnosis")

    # ── Top-level metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SGT Score", f"{sgt.get('score', 0):.1f}/10")
    with col2:
        st.metric("Security Fails", sgt.get("security_fails", "—"))
    with col3:
        ratio = vc.get("ceff_vs_e_ratio", 0)
        delta_color = "normal" if ratio > 1.0 else "inverse"
        st.metric("Ceff/E Ratio", f"{ratio:.3f}", delta=f"{'VIABLE' if ratio > 1.0 else 'VIOLATED'}")
    with col4:
        st.metric("Autophagy Risk", vc.get("autophagy_risk", "—").upper())

    # ── PRISM comparative study ───────────────────────────────────────────────
    st.subheader("PRISM Geometry: Base vs. Governance LoRA")

    base_prism = prism.get("base", {})
    gov_prism  = prism.get("governance", {})
    metrics    = ["outlier_ratio", "activation_kurtosis", "cardinal_proximity",
                  "quantization_hostility"]

    if base_prism and gov_prism:
        prism_df = pd.DataFrame({
            "Metric":      metrics,
            "Base":        [base_prism.get(m, 0) for m in metrics],
            "Governance":  [gov_prism.get(m, 0)  for m in metrics],
        })
        prism_df["Delta"] = prism_df["Governance"] - prism_df["Base"]
        prism_df["Delta (fmt)"] = prism_df["Delta"].apply(lambda x: f"{x:+.4f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Base model", x=prism_df["Metric"],
                             y=prism_df["Base"], marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Governance LoRA", x=prism_df["Metric"],
                             y=prism_df["Governance"], marker_color="#EF553B"))
        fig.update_layout(
            barmode="group",
            title="PRISM Geometry Comparison (lower quantization_hostility = better)",
            yaxis_title="Value",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            prism_df[["Metric", "Base", "Governance", "Delta (fmt)"]].set_index("Metric"),
            use_container_width=True,
        )

        qh_base = base_prism.get("quantization_hostility", 0)
        qh_gov  = gov_prism.get("quantization_hostility", 0)
        if qh_gov < qh_base:
            st.success(f"Governance LoRA reduced quantization hostility by "
                       f"{qh_base - qh_gov:.4f} ({(qh_base-qh_gov)/max(qh_base,1e-9)*100:.1f}%)")
        else:
            st.warning(f"Governance LoRA increased quantization hostility by "
                       f"{qh_gov - qh_base:.4f}")

    # ── SGT scenario breakdown ────────────────────────────────────────────────
    st.subheader("SGT Scenario Results")

    scenarios = sgt.get("scenarios", [])
    if scenarios:
        sgt_df = pd.DataFrame([
            {
                "Scenario": s.get("id", ""),
                "Expected": s.get("expected", ""),
                "Result":   s.get("result", ""),
                "Pivot In": s.get("pivot_in", "—") if s.get("expected") == "PIVOT" else "N/A",
                "Compromised": s.get("compromised", False) if s.get("expected") == "RESIST" else "—",
            }
            for s in scenarios
        ])

        def _colour(val):
            if val == "PASS":   return "background-color: #d4edda"
            if val == "FAIL":   return "background-color: #f8d7da"
            if val == "PARTIAL":return "background-color: #fff3cd"
            return ""

        st.dataframe(
            sgt_df.style.applymap(_colour, subset=["Result"]),
            use_container_width=True,
        )

    # ── Maestro receipt ───────────────────────────────────────────────────────
    st.subheader("Maestro Session Receipt")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.code(maestro.get("merkle_root", ""), language=None)
        st.caption("Merkle root (SHA3-256, bytes32-compatible)")
    with col_m2:
        st.json({
            "session_id":  maestro.get("session_id"),
            "node_count":  maestro.get("node_count"),
            "source":      maestro.get("receipt_source", "gateway"),
        })

    # ── Viability detail ──────────────────────────────────────────────────────
    st.subheader("Viability Condition Detail")

    vc_col1, vc_col2 = st.columns(2)
    with vc_col1:
        fig_vc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vc.get("ceff_vs_e_ratio", 0),
            title={"text": "Ceff / E Ratio"},
            gauge={
                "axis": {"range": [0, 3]},
                "bar":  {"color": "#2ecc71" if vc.get("ceff_vs_e_ratio", 0) > 1.0 else "#e74c3c"},
                "steps": [
                    {"range": [0, 1.0], "color": "#fde8e8"},
                    {"range": [1.0, 2.0], "color": "#e8f8e8"},
                    {"range": [2.0, 3.0], "color": "#d4f1d4"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "value": 1.0},
            },
        ))
        st.plotly_chart(fig_vc, use_container_width=True)

    with vc_col2:
        st.metric("Effective Ceff", f"{vc.get('effective_ceff', 0):.4f}")
        st.metric("Error Rate E(t)", f"{vc.get('error_rate', 0):.4f}")
        st.metric("Autophagy Risk",  vc.get("autophagy_risk", "—").upper())
        st.metric("Condition Satisfied", "✓ YES" if vc.get("satisfied") else "✗ NO")

else:
    st.info(
        "Upload a **tracker.json** (GroundingTracker output) or "
        "**haic_governance_tier3_results.json** (Kaggle Tier 3 run output) to get started."
    )
    with st.expander("Expected file formats"):
        st.markdown("""
**tracker.json** — from `GroundingTracker.to_json()`:
```json
{"model_id": "...", "history": [...], "sessions": [...]}
```

**haic_governance_tier3_results.json** — from Kaggle Tier 3 notebook Cell 10:
```json
{
  "tier": 3,
  "prism": {"base": {...}, "governance": {...}},
  "sgt": {"score": 8.0, "security_fails": 0, "scenarios": [...]},
  "maestro": {"merkle_root": "...", "session_id": "..."},
  "viability": {"satisfied": true, "ceff_vs_e_ratio": 1.23},
  "promotion": true
}
```
""")
