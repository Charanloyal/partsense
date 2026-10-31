"""PartSense: Batch Component Scoring & Telemetry Fleet Audit.

Accepts batch CSV uploads of in-service component telemetry, executes physics
feature engineering, calculates failure probabilities, evaluates tuned threshold policies,
identifies failure modes, flags unsupervised anomalies, and generates exportable reports.
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils import (
    get_sample_scoring_csv,
    load_metrics,
    load_models,
    render_header,
    render_kpi_card,
)
from partsense.features.engineer import FeatureEngineer

st.set_page_config(
    page_title="Batch Scoring – PartSense",
    page_icon="📦",
    layout="wide",
)

render_header(
    title="📦 Batch Component Scoring & Fleet Audit",
    subtitle="High-throughput batch inference, anomaly screening, and risk categorization",
)

# Load models & metrics
metrics = load_metrics()
models = load_models()
champion_model = models["champion"]
ovr_models = models["ovr_models"]
iso_forest = models["isolation_forest"]

tuned_th = metrics["main_xgboost"]["tuned_threshold_policy"]["threshold"]

# Download Sample Template
col_up, col_info = st.columns([2, 1])

with col_info:
    st.markdown("#### 📥 Sample Telemetry Data")
    st.markdown(
        "Need a test file? Download a 25-component telemetry sample extracted from the test split:"
    )
    st.download_button(
        label="📄 Download Sample CSV (25 Components)",
        data=get_sample_scoring_csv(),
        file_name="partsense_telemetry_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_up:
    st.markdown("#### 📤 Upload Telemetry File")
    uploaded_file = st.file_uploader(
        "Upload Telemetry CSV (Accepts raw sensors: Type, Air Temp, Process Temp, RPM, Torque, Tool Wear)",
        type=["csv"],
    )

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(raw_df)} records from uploaded file.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("💡 Please upload a telemetry CSV or download and test with the sample above.")
    st.stop()

# Scoring Pipeline
with st.spinner("Engineering physics features and running multi-stage inference..."):
    # Preserve tracking IDs if present
    id_cols = [c for c in ["UDI", "Product ID", "component_id"] if c in raw_df.columns]
    trace_df = (
        raw_df[id_cols].copy() if id_cols else pd.DataFrame({"Row_ID": range(1, len(raw_df) + 1)})
    )

    # Engineer features
    engineer = FeatureEngineer()
    try:
        features_df = engineer.transform(raw_df)
    except Exception as e:
        st.error(f"Feature engineering failed. Ensure required sensor columns are present: {e}")
        st.stop()

    # Model inference
    probs = champion_model.predict_proba(features_df)[:, 1]
    tuned_preds = (probs >= tuned_th).astype(int)

    # OvR Failure modes
    ovr_probs = {}
    for mode_name, mode_model in ovr_models.items():
        ovr_probs[mode_name] = mode_model.predict_proba(features_df)[:, 1]

    mode_df = pd.DataFrame(ovr_probs)
    top_modes = mode_df.idxmax(axis=1)
    top_mode_probs = mode_df.max(axis=1)

    # Isolation Forest Anomaly
    iso_scores = iso_forest.decision_function(features_df)
    iso_preds = iso_forest.predict(features_df)  # -1 is anomaly

    # Combine into results
    scored_df = trace_df.copy()
    scored_df["Failure_Probability"] = (probs * 100).round(2)
    scored_df["Alert_Triggered"] = tuned_preds == 1
    scored_df["Top_Failure_Mode"] = [
        mode if p >= 0.15 else "None" for mode, p in zip(top_modes, top_mode_probs, strict=False)
    ]
    scored_df["Mode_Likelihood"] = (top_mode_probs * 100).round(1)
    scored_df["Anomaly_Score"] = iso_scores.round(4)
    scored_df["Anomaly_Flag"] = iso_preds == -1

    # Risk Tier assignment
    risk_tiers = []
    for p, anom in zip(probs, iso_preds == -1, strict=False):
        if p >= tuned_th:
            risk_tiers.append("CRITICAL")
        elif anom or p >= 0.20:
            risk_tiers.append("ELEVATED")
        else:
            risk_tiers.append("NOMINAL")
    scored_df["Risk_Tier"] = risk_tiers

# Fleet Summary KPI Cards
st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_rows = len(scored_df)
crit_count = int((scored_df["Risk_Tier"] == "CRITICAL").sum())
elev_count = int((scored_df["Risk_Tier"] == "ELEVATED").sum())
anom_count = int(scored_df["Anomaly_Flag"].sum())

with kpi1:
    render_kpi_card(
        label="Components Scored",
        value=f"{total_rows:,}",
        delta="Total Fleet Ingest",
        delta_color="normal",
    )
with kpi2:
    render_kpi_card(
        label="Critical Alerts",
        value=f"{crit_count:,}",
        delta=f"{(crit_count / total_rows) * 100:.1f}% Alert Rate (p*={tuned_th:.3f})",
        delta_color="negative" if crit_count > 0 else "positive",
    )
with kpi3:
    render_kpi_card(
        label="Elevated Risk",
        value=f"{elev_count:,}",
        delta=f"{(elev_count / total_rows) * 100:.1f}% Watchlist",
        delta_color="normal",
    )
with kpi4:
    render_kpi_card(
        label="Flagged Anomalies",
        value=f"{anom_count:,}",
        delta="Isolation Forest (Normal Regime Outliers)",
        delta_color="normal",
    )

# Distribution Visualizations
st.markdown("### 📊 Fleet Risk Distribution")
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    fig_hist = px.histogram(
        scored_df,
        x="Failure_Probability",
        color="Risk_Tier",
        color_discrete_map={"CRITICAL": "#EF4444", "ELEVATED": "#F59E0B", "NOMINAL": "#10B981"},
        nbins=25,
        title="Failure Probability Distribution (%)",
        labels={"Failure_Probability": "Predicted Failure Probability (%)"},
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=300,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    mode_counts = (
        scored_df[scored_df["Alert_Triggered"]]["Top_Failure_Mode"].value_counts().reset_index()
    )
    mode_counts.columns = ["Failure Mode", "Alert Count"]
    fig_mode = px.bar(
        mode_counts,
        x="Failure Mode",
        y="Alert Count",
        color="Failure Mode",
        title="Critical Alerts Breakdown by Root Cause Mode",
    )
    fig_mode.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=300,
    )
    st.plotly_chart(fig_mode, use_container_width=True)

# Scored Records Table
st.markdown("### 📋 Detailed Scored Components")

# Filter controls
col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    tier_filter = st.multiselect(
        "Filter by Risk Tier",
        options=["CRITICAL", "ELEVATED", "NOMINAL"],
        default=["CRITICAL", "ELEVATED", "NOMINAL"],
    )

filtered_df = scored_df[scored_df["Risk_Tier"].isin(tier_filter)]

st.dataframe(
    filtered_df.sort_values(by="Failure_Probability", ascending=False),
    use_container_width=True,
    height=320,
)

# Export Full Results
csv_buffer = io.StringIO()
scored_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="💾 Download Scored Results CSV",
    data=csv_buffer.getvalue().encode("utf-8"),
    file_name="partsense_scored_fleet_telemetry.csv",
    mime="text/csv",
    use_container_width=True,
)
