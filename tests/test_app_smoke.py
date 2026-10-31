"""Smoke tests for Streamlit application pages and presentation helpers."""

from __future__ import annotations

import compileall
from pathlib import Path

from app.utils import generate_recommendation, get_sample_scoring_csv, load_metrics

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_streamlit_files_compile_without_syntax_errors():
    """Verify all Streamlit app scripts compile with zero syntax errors."""
    app_dir = ROOT_DIR / "app"
    result = compileall.compile_dir(str(app_dir), quiet=True)
    assert result, "Streamlit app files failed python compilation."


def test_app_utils_load_metrics():
    """Verify load_metrics reads valid metrics dictionary."""
    metrics = load_metrics()
    assert "main_xgboost" in metrics
    assert "baseline_logistic_regression" in metrics
    assert "tuned_threshold_policy" in metrics["main_xgboost"]


def test_app_utils_sample_csv_generation():
    """Verify sample CSV download generator returns non-empty CSV bytes with required columns."""
    sample_bytes = get_sample_scoring_csv()
    assert len(sample_bytes) > 0
    content = sample_bytes.decode("utf-8")
    assert "Product ID" in content
    assert "Air temperature [K]" in content
    assert "Rotational speed [rpm]" in content


def test_generate_recommendation_states():
    """Verify prescriptive recommendations for normal, anomaly, and failure states."""
    # 1. Healthy normal
    rec_healthy = generate_recommendation(
        prob=0.05, threshold=0.39, mode_probs={"HDF": 0.01}, anomaly_score=0.20
    )
    assert "HEALTHY" in rec_healthy["status"]

    # 2. Anomaly in normal regime
    rec_anom = generate_recommendation(
        prob=0.10, threshold=0.39, mode_probs={"HDF": 0.02}, anomaly_score=-0.25
    )
    assert "ANOMALY" in rec_anom["status"] or "ELEVATED" in rec_anom["status"]

    # 3. Critical failure
    rec_crit = generate_recommendation(
        prob=0.85, threshold=0.39, mode_probs={"HDF": 0.90, "PWF": 0.05}, anomaly_score=-0.35
    )
    assert "CRITICAL" in rec_crit["status"]
    assert "HDF" in rec_crit["status"]
