"""Tests for data schema validation and strict zero data leakage enforcement."""

from __future__ import annotations

import pandas as pd
import pytest

from partsense.data.validation import (
    DataLeakageError,
    DataValidationError,
    validate_raw_dataset,
    verify_zero_data_leakage,
)


def test_validate_raw_dataset_success():
    """Verify standard telemetry passes schema and bounds checks."""
    df = pd.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["M14860", "L47181"],
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 298.2],
            "Process temperature [K]": [308.6, 308.7],
            "Rotational speed [rpm]": [1551, 1408],
            "Torque [Nm]": [42.8, 46.3],
            "Tool wear [min]": [0, 3],
            "Machine failure": [0, 0],
            "TWF": [0, 0],
            "HDF": [0, 0],
            "PWF": [0, 0],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )
    audit = validate_raw_dataset(df)
    assert audit["status"] == "PASSED"
    assert audit["total_records"] == 2
    assert audit["total_failures"] == 0


def test_validate_raw_dataset_missing_column():
    """Verify validation raises DataValidationError when required column is missing."""
    df = pd.DataFrame({"UDI": [1], "Product ID": ["M14860"]})
    with pytest.raises(DataValidationError, match="missing mandatory columns"):
        validate_raw_dataset(df)


def test_verify_zero_data_leakage_clean():
    """Verify audit passes when only clean features are present."""
    clean_cols = [
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
    # Should execute cleanly without error
    verify_zero_data_leakage(clean_cols)


@pytest.mark.parametrize(
    "leaked_col",
    ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
)
def test_verify_zero_data_leakage_catches_forbidden(leaked_col: str):
    """Verify that any presence of forbidden target or identifier flags raises DataLeakageError."""
    feature_cols = ["air_temperature_k", "torque_nm", leaked_col]
    with pytest.raises(DataLeakageError, match="CRITICAL DATA LEAKAGE VIOLATION"):
        verify_zero_data_leakage(feature_cols)
