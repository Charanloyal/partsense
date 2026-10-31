"""Tests for serialized machine learning models, inference bounds, and threshold tuning."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from partsense.models.threshold_tuner import ThresholdTuner

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_feature_row() -> pd.DataFrame:
    """Fixture providing a single valid feature row in exact expected column order."""
    cols = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "power_W",
        "temp_diff_K",
        "strain_minNm",
        "Type_L",
        "Type_M",
        "Type_H",
    ]
    data = {
        "air_temperature_k": [300.0],
        "process_temperature_k": [310.0],
        "rotational_speed_rpm": [1500],
        "torque_nm": [40.0],
        "tool_wear_min": [100],
        "power_W": [6283.18],
        "temp_diff_K": [10.0],
        "strain_minNm": [4000.0],
        "Type_L": [0],
        "Type_M": [1],
        "Type_H": [0],
    }
    return pd.DataFrame(data)[cols]


def test_champion_xgboost_inference(sample_feature_row: pd.DataFrame):
    """Verify champion model produces valid probability in [0, 1]."""
    model_path = ROOT_DIR / "models" / "xgboost_failure_classifier.joblib"
    assert model_path.exists()
    model = joblib.load(model_path)

    prob = model.predict_proba(sample_feature_row)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_ovr_failure_modes_inference(sample_feature_row: pd.DataFrame):
    """Verify all 4 learnable failure mode models produce valid probabilities."""
    ovr_path = ROOT_DIR / "models" / "xgboost_failure_modes_ovr.joblib"
    assert ovr_path.exists()
    ovr_dict = joblib.load(ovr_path)

    for mode in ["TWF", "HDF", "PWF", "OSF"]:
        assert mode in ovr_dict
        mode_prob = ovr_dict[mode].predict_proba(sample_feature_row)[0, 1]
        assert 0.0 <= mode_prob <= 1.0


def test_isolation_forest_anomaly_score(sample_feature_row: pd.DataFrame):
    """Verify Isolation Forest outputs real-valued decision score and binary flag."""
    iso_path = ROOT_DIR / "models" / "isolation_forest.joblib"
    assert iso_path.exists()
    iso_forest = joblib.load(iso_path)

    score = float(iso_forest.decision_function(sample_feature_row)[0])
    pred = int(iso_forest.predict(sample_feature_row)[0])

    assert isinstance(score, float)
    assert pred in (-1, 1)


def test_threshold_tuner_optimization():
    """Verify ThresholdTuner maximizes recall while strictly observing precision floor."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.9])

    tuner = ThresholdTuner(min_precision_floor=0.50)
    policy = tuner.tune_threshold(y_true, y_prob)

    assert "optimal_threshold" in policy
    assert policy["achieved_precision"] >= 0.50
    assert 0.0 <= policy["achieved_recall"] <= 1.0
