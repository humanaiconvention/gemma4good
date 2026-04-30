import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Ensure repo root is in path to import from viability
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

# Use a default path or let the user upload a JSON file
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
            # Uploaded file
            data = json.load(file_or_path)
            tracker.sessions = data.get("sessions", [])
            tracker.history = data.get("history", [])
            tracker.model_id = data.get("model_id", "unknown")
            tracker.created_at = data.get("created_at", "")
    except Exception as e:
        st.error(f"Error loading tracker: {e}")
    return tracker

uploaded_file = st.file_uploader("Upload tracker.json", type=["json"])
if uploaded_file is not None:
    tracker = load_tracker(uploaded_file)
else:
    tracker = load_tracker(default_tracker_path)

if tracker and tracker.history:
    st.header(f"Model: {tracker.model_id}")
    
    # Overview Metrics
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
    
    # Convert history to DataFrame
    df = pd.DataFrame(tracker.history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot Ceff vs E(t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['effective_ceff'], mode='lines+markers', name='Ceff(t)'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['error_rate'], mode='lines+markers', name='E(t)'))
    fig.update_layout(title="Correction Bandwidth vs Error Accumulation", xaxis_title="Time", yaxis_title="Rate (events/day)")
    st.plotly_chart(fig, use_container_width=True)

    # Plot Autophagy Risk Over Time
    st.subheader("Representation Autophagy Risk")
    risk_mapping = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    df['risk_level'] = df['autophagy_risk'].map(risk_mapping)
    
    fig2 = px.line(df, x='timestamp', y='risk_level', title="Autophagy Risk Over Time", markers=True)
    fig2.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=["None", "Low", "Medium", "High", "Critical"])
    st.plotly_chart(fig2, use_container_width=True)
    
    # Show Session Details
    st.subheader("Recent Sessions")
    sessions_df = pd.DataFrame(tracker.sessions)
    if not sessions_df.empty:
        # Select key columns
        display_cols = ['session_id', 'timestamp', 'corrections', 'consent_valid', 'viability_satisfied']
        available_cols = [c for c in display_cols if c in sessions_df.columns]
        st.dataframe(sessions_df[available_cols].tail(10))

else:
    st.info("No grounding history available to display. Upload a tracker.json file to view metrics.")
