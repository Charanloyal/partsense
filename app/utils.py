"""PartSense Streamlit UI Utilities and Cached Model/Data Loaders.

Provides resource-cached models, data loaders, DuckDB connections, and
reusable presentation components for all multipage dashboard views.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
import streamlit as st

from partsense.config import load_config
from partsense.kpis.runner import KPIRunner

ROOT_DIR = Path(__file__).resolve().parent.parent


@st.cache_resource(show_spinner=False)
def get_config():
    """Load and cache typed application configuration."""
    return load_config(ROOT_DIR / "configs" / "config.yaml")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    """Load the honest, audited metrics computed by the evaluation pipeline."""
    metrics_path = ROOT_DIR / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {metrics_path}. "
            "Please run 'python tasks.py evaluate' first."
        )
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, Any]:
    """Load serialized models via st.cache_resource for zero startup overhead."""
    models_dir = ROOT_DIR / "models"
    champion_path = models_dir / "xgboost_failure_classifier.joblib"
    baseline_path = models_dir / "baseline_logistic_regression.joblib"
    ovr_path = models_dir / "xgboost_failure_modes_ovr.joblib"
    iso_path = models_dir / "isolation_forest.joblib"

    if not champion_path.exists():
        raise FileNotFoundError(
            f"Models not found in {models_dir}. Please run 'python tasks.py train' first."
        )

    champion = joblib.load(champion_path)
    baseline = joblib.load(baseline_path)
    ovr_models = joblib.load(ovr_path)
    iso_forest = joblib.load(iso_path)

    return {
        "champion": champion,
        "baseline": baseline,
        "ovr_models": ovr_models,
        "isolation_forest": iso_forest,
    }


@st.cache_resource(show_spinner=False)
def load_shap_explainer(_model: Any) -> shap.TreeExplainer:
    """Create and cache a SHAP TreeExplainer for the champion XGBoost model."""
    return shap.TreeExplainer(_model)


@st.cache_resource(show_spinner=False)
def get_kpi_runner() -> KPIRunner:
    """Initialize and cache the DuckDB KPI execution engine."""
    cfg = get_config()
    duckdb_path = ROOT_DIR / cfg.data.duckdb_path
    sql_dir = ROOT_DIR / "sql"
    return KPIRunner(duckdb_path=duckdb_path, sql_dir=sql_dir)


@st.cache_data(show_spinner=False)
def load_test_split() -> pd.DataFrame:
    """Load the stratified held-out test split."""
    test_path = ROOT_DIR / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}.")
    return pd.read_parquet(test_path)


@st.cache_data(show_spinner=False)
def get_sample_scoring_csv() -> bytes:
    """Generate a clean raw CSV sample for user download in Batch Scoring."""
    test_df = load_test_split()
    sample_df = test_df.sample(n=min(25, len(test_df)), random_state=42)

    # Provide canonical raw telemetry columns
    # In test split, columns are already sanitized. Let's provide raw or clean:
    export_df = pd.DataFrame(
        {
            "Product ID": [f"M1486{i:02d}" for i in range(len(sample_df))],
            "Type": [
                "L" if r["Type_L"] == 1 else ("M" if r["Type_M"] == 1 else "H")
                for _, r in sample_df.iterrows()
            ],
            "Air temperature [K]": sample_df["air_temperature_k"].round(2).values,
            "Process temperature [K]": sample_df["process_temperature_k"].round(2).values,
            "Rotational speed [rpm]": sample_df["rotational_speed_rpm"].astype(int).values,
            "Torque [Nm]": sample_df["torque_nm"].round(2).values,
            "Tool wear [min]": sample_df["tool_wear_min"].astype(int).values,
        }
    )
    return export_df.to_csv(index=False).encode("utf-8")


def render_header(title: str, subtitle: str, badge: str = "SYNTHETIC BENCHMARK") -> None:
    """Render standardized high-aesthetic page header with synthetic disclaimer badge."""
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <p style="margin: 0.35rem 0 0 0; font-size: 1.05rem; color: #94A3B8;">
                    {subtitle}
                </p>
            </div>
            <div>
                <span style="background-color: #1E293B; color: #38BDF8; border: 1px solid #38BDF8; padding: 0.35rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: inline-block;">
                    ⚙️ {badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: str = "normal",
    help_text: str | None = None,
) -> None:
    """Render an automotive telemetry card."""
    delta_html = ""
    if delta:
        color = (
            "#10B981"
            if delta_color == "positive"
            else ("#EF4444" if delta_color == "negative" else "#94A3B8")
        )
        delta_html = f'<div style="color: {color}; font-size: 0.85rem; font-weight: 500; margin-top: 0.25rem;">{delta}</div>'

    help_attr = f'title="{help_text}"' if help_text else ""
    st.markdown(
        f"""
        <div {help_attr} style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); transition: transform 0.2s ease, border-color 0.2s ease;">
            <div style="font-size: 0.85rem; font-weight: 500; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">
                {label}
            </div>
            <div style="font-size: 2.1rem; font-weight: 700; color: #F8FAFC; line-height: 1.1;">
                {value}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_recommendation(
    prob: float,
    threshold: float,
    mode_probs: dict[str, float],
    anomaly_score: float,
) -> dict[str, str]:
    """Generate prescriptive engineering advice from telemetry and physics modes."""
    is_failing = prob >= threshold
    top_mode = max(mode_probs.items(), key=lambda x: x[1])[0] if mode_probs else "None"
    top_mode_prob = mode_probs.get(top_mode, 0.0)

    if not is_failing:
        if anomaly_score < -0.10:
            return {
                "status": "ELEVATED VIBRATION / UNUSUAL SENSOR PROFILE",
                "badge_color": "#F59E0B",
                "headline": "Telemetry Anomaly Detected (Normal Cutting Regime But Outlier Signature)",
                "action": (
                    "Component is operating within primary safety thresholds, but Isolation Forest "
                    f"detected an unusual sensor signature (score: {anomaly_score:.3f}). "
                    "Recommend scheduling acoustic vibration and spindle alignment check during next scheduled shift."
                ),
            }
        return {
            "status": "HEALTHY / NORMAL OPERATION",
            "badge_color": "#10B981",
            "headline": f"Component Operational (Failure Risk: {prob * 100:.1f}%)",
            "action": (
                "All thermal dissipation, spindle power, and strain metrics are within nominal limits. "
                "Continue standard maintenance interval."
            ),
        }

    # Action recommendations based on physical failure mechanism
    mode_guidance = {
        "HDF": (
            "Heat Dissipation Failure detected. Temperature differential or spindle cooling is insufficient. "
            "Action: Inspect coolant flow rate, purge radiator blockages, and lower continuous spindle speed."
        ),
        "PWF": (
            "Power Surge / Stall Failure detected. Operating torque and RPM combination exceeds motor capability envelope. "
            "Action: Inspect electrical drive inverters, check tool feed rate, and check for workpiece jamming."
        ),
        "OSF": (
            "Overstrain Failure detected. Product of cumulative tool wear and high cutting torque exceeds shear limit. "
            "Action: Immediately retract Spindle. Replace worn cutting insert and reduce feed-rate depth."
        ),
        "TWF": (
            "Tool Wear Failure detected. Cumulative cutting duration has exceeded material breakdown threshold. "
            "Action: Replace cutting insert and recalibrate zero-point offset before next production run."
        ),
    }

    advice = mode_guidance.get(
        top_mode,
        "General mechanical degradation detected. Halt machine for detailed visual and ultrasonic inspection.",
    )

    return {
        "status": f"CRITICAL RISK: {top_mode} PREDICTED",
        "badge_color": "#EF4444",
        "headline": f"Failure Imminent: {prob * 100:.1f}% Probability (Top Driver: {top_mode} at {top_mode_prob * 100:.1f}%)",
        "action": advice,
    }
