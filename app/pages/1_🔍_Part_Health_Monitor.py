"""PartSense: Part Health Monitor (Single Component Telemetry & Explainability).

Provides real-time interactive telemetry sliders, risk gauge, multi-mode failure
diagnosis, Isolation Forest anomaly scoring, SHAP waterfall, and plain-English recommendations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from app.utils import (
    generate_recommendation,
    load_metrics,
    load_models,
    load_shap_explainer,
    render_header,
)
from partsense.features.engineer import FeatureEngineer

st.set_page_config(
    page_title="Part Health Monitor – PartSense",
    page_icon="🔍",
    layout="wide",
)

render_header(
    title="🔍 Part Health Monitor",
    subtitle="Interactive telemetry inspection, multi-mode diagnosis, anomaly detection & SHAP explanation",
)

# Load cached models and metrics
metrics = load_metrics()
models = load_models()
champion_model = models["champion"]
baseline_model = models["baseline"]
ovr_models = models["ovr_models"]
iso_forest = models["isolation_forest"]
explainer = load_shap_explainer(champion_model)

tuned_threshold = metrics["main_xgboost"]["tuned_threshold_policy"]["threshold"]

# Presets for quick demonstration
st.sidebar.subheader("⚡ Quick Telemetry Presets")
preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    [
        "Custom Manual Inputs",
        "Nominal Healthy Component",
        "Heat Dissipation Failure (HDF Scenario)",
        "Overstrain Failure (OSF Scenario)",
        "Power Surge / High Torque (PWF Scenario)",
        "Severe Tool Wear (TWF Scenario)",
    ],
    index=0,
)

# Preset default values
presets_map = {
    "Nominal Healthy Component": {
        "type": "M",
        "air_temp": 300.0,
        "proc_temp": 310.0,
        "rpm": 1530,
        "torque": 38.5,
        "tool_wear": 45,
    },
    "Heat Dissipation Failure (HDF Scenario)": {
        "type": "L",
        "air_temp": 303.5,
        "proc_temp": 311.8,
        "rpm": 1340,
        "torque": 52.0,
        "tool_wear": 180,
    },
    "Overstrain Failure (OSF Scenario)": {
        "type": "L",
        "air_temp": 301.0,
        "proc_temp": 310.5,
        "rpm": 1320,
        "torque": 66.0,
        "tool_wear": 215,
    },
    "Power Surge / High Torque (PWF Scenario)": {
        "type": "H",
        "air_temp": 298.5,
        "proc_temp": 308.2,
        "rpm": 2650,
        "torque": 72.0,
        "tool_wear": 30,
    },
    "Severe Tool Wear (TWF Scenario)": {
        "type": "M",
        "air_temp": 300.5,
        "proc_temp": 310.0,
        "rpm": 1500,
        "torque": 45.0,
        "tool_wear": 240,
    },
}

current_preset = presets_map.get(preset, None)

col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("🎛️ Telemetry Sensors")

    type_val = st.selectbox(
        "Quality Variant (Type)",
        ["L", "M", "H"],
        index=["L", "M", "H"].index(current_preset["type"]) if current_preset else 0,
        help="L (Low: 50%), M (Medium: 30%), H (High: 20%) variant grade",
    )

    air_temp = st.slider(
        "Air Temperature [K]",
        min_value=290.0,
        max_value=310.0,
        value=current_preset["air_temp"] if current_preset else 300.5,
        step=0.1,
    )

    proc_temp = st.slider(
        "Process Temperature [K]",
        min_value=300.0,
        max_value=320.0,
        value=current_preset["proc_temp"] if current_preset else 310.0,
        step=0.1,
    )

    rpm = st.slider(
        "Rotational Speed [rpm]",
        min_value=1100,
        max_value=3000,
        value=current_preset["rpm"] if current_preset else 1500,
        step=10,
    )

    torque = st.slider(
        "Torque [Nm]",
        min_value=3.0,
        max_value=85.0,
        value=current_preset["torque"] if current_preset else 40.0,
        step=0.5,
    )

    tool_wear = st.slider(
        "Tool Wear [min]",
        min_value=0,
        max_value=260,
        value=current_preset["tool_wear"] if current_preset else 100,
        step=1,
        help="Operating wear minutes. Estimated operating hours = tool wear / 60.",
    )

    # Compute and display physics features
    engineer = FeatureEngineer()
    raw_input_df = pd.DataFrame(
        [
            {
                "Type": type_val,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )
    features_df = engineer.transform(raw_input_df)

    st.markdown("#### ⚡ Physics-Engineered Indicators")
    p_w = features_df["power_W"].iloc[0]
    dt_k = features_df["temp_diff_K"].iloc[0]
    strain_val = features_df["strain_minNm"].iloc[0]

    st.markdown(
        f"""
        <div style="background: #111827; border-radius: 8px; padding: 0.8rem; font-size: 0.85rem; border: 1px solid #1F2937;">
            <div>⚡ <b>Mechanical Power:</b> <code>{p_w:.1f} W</code></div>
            <div style="margin-top: 0.25rem;">🌡️ <b>Temperature Gradient (ΔT):</b> <code>{dt_k:.2f} K</code></div>
            <div style="margin-top: 0.25rem;">🔩 <b>Cumulative Strain:</b> <code>{strain_val:.1f} min·Nm</code></div>
            <div style="margin-top: 0.25rem;">⏱️ <b>Estimated Operating Hours:</b> <code>{tool_wear / 60.0:.2f} hrs</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_results:
    # 1. Main Binary Classifier Prediction
    prob_failure = float(champion_model.predict_proba(features_df)[0, 1])
    is_failing_tuned = prob_failure >= tuned_threshold

    # 2. Failure Mode Diagnoser (OvR)
    mode_probs = {}
    for mode_name, mode_model in ovr_models.items():
        mode_prob = float(mode_model.predict_proba(features_df)[0, 1])
        mode_probs[mode_name] = mode_prob

    # 3. Isolation Forest Anomaly Detection
    # decision_function gives negative for anomalies, positive for normal
    iso_score = float(iso_forest.decision_function(features_df)[0])
    is_anomaly = iso_forest.predict(features_df)[0] == -1

    # 4. Plain-English Recommendation
    rec = generate_recommendation(
        prob=prob_failure,
        threshold=tuned_threshold,
        mode_probs=mode_probs,
        anomaly_score=iso_score,
    )

    # Display Risk Gauge & Status Banner
    st.markdown(
        f"""
        <div style="background-color: {rec["badge_color"]}20; border: 1px solid {rec["badge_color"]}; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: {rec["badge_color"]}; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.05em;">
                {rec["status"]}
            </div>
            <div style="color: #F8FAFC; font-size: 1.2rem; font-weight: 600; margin-top: 0.25rem;">
                {rec["headline"]}
            </div>
            <div style="color: #CBD5E1; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.5;">
                👉 <b>Prescriptive Action:</b> {rec["action"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gauge_col, modes_col = st.columns([1, 1])

    with gauge_col:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_failure * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Failure Probability (%)", "font": {"size": 16, "color": "#F8FAFC"}},
                number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": "#38BDF8" if prob_failure < tuned_threshold else "#EF4444"},
                    "bgcolor": "#1E293B",
                    "borderwidth": 1,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, 30], "color": "#064E3B"},
                        {"range": [30, tuned_threshold * 100], "color": "#78350F"},
                        {"range": [tuned_threshold * 100, 100], "color": "#7F1D1D"},
                    ],
                    "threshold": {
                        "line": {"color": "#F59E0B", "width": 4},
                        "thickness": 0.8,
                        "value": tuned_threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin={"l": 20, "r": 20, "t": 30, "b": 10},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"Tuned Policy Threshold: **{tuned_threshold * 100:.1f}%** (orange line)")

    with modes_col:
        st.markdown("#### 🔬 Root-Cause Failure Modes")
        mode_data = [
            {"Mode": "HDF (Heat Dissipation)", "Probability": mode_probs.get("HDF", 0.0)},
            {"Mode": "PWF (Power Surge/Stall)", "Probability": mode_probs.get("PWF", 0.0)},
            {"Mode": "OSF (Overstrain Failure)", "Probability": mode_probs.get("OSF", 0.0)},
            {"Mode": "TWF (Tool Wear Failure)", "Probability": mode_probs.get("TWF", 0.0)},
        ]
        modes_df = pd.DataFrame(mode_data)
        fig_bar = go.Figure(
            go.Bar(
                x=modes_df["Probability"] * 100,
                y=modes_df["Mode"],
                orientation="h",
                marker={
                    "color": modes_df["Probability"] * 100,
                    "colorscale": "Reds",
                    "cmin": 0,
                    "cmax": 100,
                    "showscale": False,
                },
                text=[f"{p * 100:.1f}%" for p in modes_df["Probability"]],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"},
            height=240,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis={"range": [0, 105], "title": "Likelihood (%)"},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Anomaly Inspection Section
    st.markdown("---")
    st.markdown("#### 🌲 Unsupervised Anomaly Screening (Isolation Forest)")
    ano_col1, ano_col2 = st.columns([1, 2])
    with ano_col1:
        st.metric(
            label="Isolation Forest Score",
            value=f"{iso_score:.4f}",
            delta="Anomaly Flagged" if is_anomaly else "Normal Telemetry Envelope",
            delta_color="inverse" if is_anomaly else "normal",
        )
    with ano_col2:
        st.markdown(
            f"""
            <div style="background: #111827; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #1F2937; font-size: 0.9rem; color: #CBD5E1;">
                Trained strictly on <b>normal (non-failing) operational telemetry</b>.
                Lower / negative values indicate departure from standard cutting regimes even if failure has not yet manifested.
                <br/><b>Inspection Status:</b> {"⚠️ Component in anomalous sensor region" if is_anomaly else "✅ Component inside standard operational envelope"}.
            </div>
            """,
            unsafe_allow_html=True,
        )

# SHAP Local Waterfall Explanation
st.markdown("---")
st.subheader("📊 Local SHAP Waterfall Decomposition")
st.markdown(
    "Explains exactly how each sensor telemetry reading pushed the model log-odds "
    "above or below the base expected value $E[f(X)]$."
)

shap_values = explainer(features_df)

fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0B0F19")
fig.patch.set_facecolor("#0B0F19")
ax.set_facecolor("#0B0F19")
plt.rcParams["text.color"] = "#F8FAFC"
plt.rcParams["axes.labelcolor"] = "#F8FAFC"
plt.rcParams["xtick.color"] = "#94A3B8"
plt.rcParams["ytick.color"] = "#F8FAFC"

shap.plots.waterfall(shap_values[0], max_display=8, show=False)
plt.title(
    f"SHAP Waterfall: Prediction Explanation (P[Failure] = {prob_failure * 100:.1f}%)",
    color="#F8FAFC",
    fontsize=12,
    pad=15,
)
plt.tight_layout()

st.pyplot(fig, clear_figure=True)
