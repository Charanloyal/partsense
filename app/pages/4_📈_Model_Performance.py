"""PartSense: Model Performance, Validation & Explainability.

Presents honest, audited metrics from reports/metrics.json, interactive threshold
trade-off slider, dynamic confusion matrices, ROC/PR curves, and global SHAP importance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from app.utils import (
    load_metrics,
    load_models,
    load_test_split,
    render_header,
)

st.set_page_config(
    page_title="Model Performance – PartSense",
    page_icon="📈",
    layout="wide",
)

render_header(
    title="📈 Model Performance & Validation",
    subtitle="Audited evaluation on 2,000 holdout test components (3.4% failure prevalence)",
)

metrics = load_metrics()
models = load_models()
test_df = load_test_split()

champion_model = models["champion"]
baseline_model = models["baseline"]

feature_cols = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "power_W",
    "temp_diff_K",
    "strain_minNm",
    "Type_H",
    "Type_L",
    "Type_M",
]

X_test = test_df[feature_cols]
y_test = test_df["Machine failure"].values

# Model predictions
y_prob_xgb = champion_model.predict_proba(X_test)[:, 1]
y_prob_base = baseline_model.predict_proba(X_test)[:, 1]

# Section 1: Champion vs Baseline Comparison Table
st.subheader("🏆 Model Benchmarking: Champion XGBoost vs Baseline")

xgb_dict = metrics["main_xgboost"]
base_dict = metrics["baseline_logistic_regression"]
tuned_policy = xgb_dict["tuned_threshold_policy"]

comparison_data = [
    {
        "Model": "PartSense Champion XGBoost",
        "Test ROC-AUC": f"{xgb_dict['discrimination']['roc_auc']:.4f}",
        "Test PR-AUC": f"{xgb_dict['discrimination']['pr_auc']:.4f}",
        "Default F1 (p=0.5)": f"{xgb_dict['default_threshold_0_5']['f1']:.4f}",
        "Tuned Policy Recall": f"{tuned_policy['recall'] * 100:.1f}%",
        "Tuned Policy Precision": f"{tuned_policy['precision'] * 100:.1f}%",
        "Operating Threshold": f"p* = {tuned_policy['threshold']:.4f}",
    },
    {
        "Model": "Baseline Logistic Regression (Standardized)",
        "Test ROC-AUC": f"{base_dict['roc_auc']:.4f}",
        "Test PR-AUC": f"{base_dict['pr_auc']:.4f}",
        "Default F1 (p=0.5)": f"{base_dict['f1_at_05']:.4f}",
        "Tuned Policy Recall": f"{base_dict['recall_at_05'] * 100:.1f}% (at p=0.5)",
        "Tuned Policy Precision": f"{base_dict['precision_at_05'] * 100:.1f}% (at p=0.5)",
        "Operating Threshold": "p = 0.5000",
    },
]
st.table(pd.DataFrame(comparison_data))

st.markdown("---")

# Section 2: Interactive Threshold Explorer & Dynamic Confusion Matrix
st.subheader("🎚️ Dynamic Decision Threshold Explorer")
st.caption(
    "Adjust the classification threshold $p^*$ to explore the precision-recall trade-off on the 2,000 test cases."
)

col_slider, col_cm, col_kpis = st.columns([1, 1, 1])

with col_slider:
    tuned_th = float(tuned_policy["threshold"])
    curr_th = st.slider(
        "Decision Threshold (p*)",
        min_value=0.05,
        max_value=0.95,
        value=tuned_th,
        step=0.01,
        help="Components with P[Failure] >= p* trigger preventative alerts",
    )
    if st.button(f"Reset to Tuned Policy (p*={tuned_th:.3f})"):
        curr_th = tuned_th

    # Dynamic metrics at curr_th
    y_pred_dynamic = (y_prob_xgb >= curr_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dynamic).ravel()

    dyn_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    dyn_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dyn_f1 = 2 * dyn_prec * dyn_rec / (dyn_prec + dyn_rec) if (dyn_prec + dyn_rec) > 0 else 0.0

with col_cm:
    cm_matrix = [[tn, fp], [fn, tp]]
    fig_cm = px.imshow(
        cm_matrix,
        text_auto=True,
        labels={"x": "Predicted Label", "y": "True Label", "color": "Count"},
        x=["Pass (0)", "Alert (1)"],
        y=["Pass (0)", "Failure (1)"],
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix (at p* = {curr_th:.2f})",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=260,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_kpis:
    st.markdown("#### Performance at Selected Threshold")
    st.markdown(f"**Recall (Sensitivity):** `{dyn_rec * 100:.1f}%` ({tp} of {tp + fn} caught)")
    st.markdown(f"**Precision:** `{dyn_prec * 100:.1f}%` ({tp} true of {tp + fp} alarms)")
    st.markdown(f"**F1 Score:** `{dyn_f1:.4f}`")
    st.markdown(f"**Missed Failures (FN):** `{fn}`")
    st.markdown(f"**False Alarms (FP):** `{fp}`")

st.markdown("---")

# Section 3: ROC and Precision-Recall Curves
st.subheader("📈 Discrimination Curves")
col_roc, col_pr = st.columns([1, 1])

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)

prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
prec_base, rec_base, _ = precision_recall_curve(y_test, y_prob_base)

with col_roc:
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr_xgb,
            y=tpr_xgb,
            mode="lines",
            name=f"XGBoost (AUC = {xgb_dict['discrimination']['roc_auc']:.3f})",
            line={"color": "#38BDF8", "width": 3},
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=fpr_base,
            y=tpr_base,
            mode="lines",
            name=f"Baseline LogReg (AUC = {base_dict['roc_auc']:.3f})",
            line={"color": "#94A3B8", "dash": "dash"},
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line={"color": "#475569", "dash": "dot"},
        )
    )
    fig_roc.update_layout(
        title="Receiver Operating Characteristic (ROC)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        legend={"yanchor": "bottom", "y": 0.05, "xanchor": "right", "x": 0.95},
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_pr:
    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=rec_xgb,
            y=prec_xgb,
            mode="lines",
            name=f"XGBoost (PR-AUC = {xgb_dict['discrimination']['pr_auc']:.3f})",
            line={"color": "#10B981", "width": 3},
        )
    )
    fig_pr.add_trace(
        go.Scatter(
            x=rec_base,
            y=prec_base,
            mode="lines",
            name=f"Baseline LogReg (PR-AUC = {base_dict['pr_auc']:.3f})",
            line={"color": "#94A3B8", "dash": "dash"},
        )
    )
    fig_pr.add_hline(
        y=0.034,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Prevalence = 3.4%",
    )
    fig_pr.update_layout(
        title="Precision-Recall Curve (Severe Class Imbalance)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        legend={"yanchor": "bottom", "y": 0.05, "xanchor": "right", "x": 0.95},
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")

# Section 4: Global SHAP Feature Importance
st.subheader("🌐 Global SHAP Feature Importance")
col_shap_img, col_shap_tbl = st.columns([1, 1])

shap_fig_path = (
    Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "shap_summary.png"
)

with col_shap_img:
    if shap_fig_path.exists():
        st.image(str(shap_fig_path), caption="SHAP Summary Plot (Mean |SHAP| across Test Split)")
    else:
        st.info("SHAP plot image generated in reports/figures/shap_summary.png.")

with col_shap_tbl:
    st.markdown("#### Top Predictive Telemetry Drivers")
    shap_ranking = metrics.get("shap_importance_ranking", [])
    if shap_ranking:
        shap_df = pd.DataFrame(shap_ranking)
        fig_shap_bar = px.bar(
            shap_df.head(8),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Viridis",
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig_shap_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"},
            height=320,
            yaxis={"autorange": "reversed"},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig_shap_bar, use_container_width=True)

st.markdown("---")

# Section 5: Auxiliary Models Performance (OvR Modes & Isolation Forest)
st.subheader("🔬 Multi-Mode Root-Cause & Anomaly Detection Performance")
col_ovr, col_iso = st.columns([1, 1])

with col_ovr:
    st.markdown("#### One-vs-Rest Failure Mode Diagnosers")
    ovr_dict = metrics.get("failure_modes_ovr", {})
    ovr_rows = []
    for mode, data in ovr_dict.items():
        if mode == "RNF":
            ovr_rows.append(
                {
                    "Mode": "RNF (Random Noise)",
                    "Test Cases": data.get("test_cases", 4),
                    "ROC-AUC": "N/A",
                    "PR-AUC": "N/A",
                    "Status": "Mathematically Not Learnable",
                }
            )
        else:
            ovr_rows.append(
                {
                    "Mode": mode,
                    "Test Cases": data.get("test_cases", 0),
                    "ROC-AUC": f"{data.get('roc_auc', 0):.4f}",
                    "PR-AUC": f"{data.get('pr_auc', 0):.4f}",
                    "Status": f"F1 = {data.get('f1', 0):.4f}",
                }
            )
    st.table(pd.DataFrame(ovr_rows))

with col_iso:
    st.markdown("#### Unsupervised Isolation Forest (Normal-Only Training)")
    st.caption("How many test failures are captured in the top-k% most anomalous components")
    iso_dict = metrics.get("anomaly_detection_isolation_forest", {})
    iso_rows = []
    for tier, tdata in iso_dict.items():
        iso_rows.append(
            {
                "Top-k% Budget": tier.replace("top_", "").replace("pct", "%"),
                "Components": tdata["k_samples"],
                "True Failures Captured": tdata["detected_failures"],
                "Precision@k": f"{tdata['precision_at_k'] * 100:.1f}%",
                "Recall@k": f"{tdata['recall_at_k'] * 100:.1f}%",
            }
        )
    st.table(pd.DataFrame(iso_rows))
