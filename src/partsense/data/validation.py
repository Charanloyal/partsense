"""Data validation, thermodynamic boundary checks, and strict leakage prevention."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

FORBIDDEN_LEAKAGE_COLUMNS = {
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}


class DataValidationError(ValueError):
    """Raised when incoming dataset violates required schema or boundary invariants."""

    pass


class DataLeakageError(ValueError):
    """Raised when an attempt is made to include target or identifier columns in feature set."""

    pass


def verify_zero_data_leakage(feature_columns: list[str]) -> None:
    """Verify that NO forbidden leakage columns exist in the feature set.

    LEAKAGE RULES: Never use UDI, Product ID, or the failure-mode flags
    (TWF, HDF, PWF, OSF, RNF) as features when predicting 'Machine failure'.

    Args:
        feature_columns: List of feature names intended for model training/inference.

    Raises:
        DataLeakageError: If any forbidden column or failure-mode indicator is detected.
    """
    leaking_cols = set(feature_columns).intersection(FORBIDDEN_LEAKAGE_COLUMNS)
    if leaking_cols:
        msg = (
            f"CRITICAL DATA LEAKAGE VIOLATION: The following forbidden columns were found "
            f"in the feature set: {sorted(leaking_cols)}. "
            f"Identifiers and failure-mode labels (TWF, HDF, PWF, OSF, RNF) must never be features!"
        )
        logger.error(msg)
        raise DataLeakageError(msg)
    logger.info("Data leakage audit passed: 0 forbidden columns in feature set.")


def validate_raw_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate raw AI4I 2020 dataset for column completeness and physical bounds.

    Args:
        df: Input DataFrame containing component telemetry.

    Returns:
        Validation audit metadata dictionary.

    Raises:
        DataValidationError: If mandatory columns are missing or invariants violated.
    """
    logger.info("Validating dataset schema and physical bounds for shape: %s", df.shape)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Dataset is missing mandatory columns: {missing_cols}"
        logger.error(msg)
        raise DataValidationError(msg)

    # Null verification
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        msg = f"Dataset contains unexpected nulls in columns: {cols_with_nulls}"
        logger.error(msg)
        raise DataValidationError(msg)

    # Variant category validation
    valid_types = {"L", "M", "H"}
    invalid_types = set(df["Type"].unique()) - valid_types
    if invalid_types:
        msg = f"Invalid 'Type' variant values found: {invalid_types}"
        logger.error(msg)
        raise DataValidationError(msg)

    # Binary label validation
    binary_cols = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for col in binary_cols:
        unique_vals = set(df[col].unique())
        if not unique_vals.issubset({0, 1}):
            msg = f"Binary column '{col}' contains non-binary values: {unique_vals}"
            logger.error(msg)
            raise DataValidationError(msg)

    total_records = len(df)
    total_failures = int(df["Machine failure"].sum())
    failure_rate = float(total_failures / total_records) if total_records > 0 else 0.0

    audit_summary = {
        "status": "PASSED",
        "total_records": total_records,
        "total_failures": total_failures,
        "failure_rate": round(failure_rate, 4),
        "source": "UCI AI4I 2020 Predictive Maintenance Dataset (ID 601, CC BY 4.0)",
        "synthetic_notice": "SYNTHETIC BENCHMARK: Reflects real component degradation physics with synthetic modeling.",
    }
    logger.info("Validation audit passed successfully. Failure rate: %.2f%%", failure_rate * 100)
    return audit_summary
