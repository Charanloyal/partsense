"""Physics-based feature engineering for automotive component failure prediction.

Derives thermodynamic, mechanical power, and cumulative strain metrics directly
mapped to physical breakdown mechanisms (HDF, PWF, OSF).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from partsense.data.validation import verify_zero_data_leakage

logger = logging.getLogger(__name__)

RAW_COLUMN_MAP = {
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}


def compute_power_w(torque: pd.Series | float, rpm: pd.Series | float) -> pd.Series | float:
    """Compute mechanical power in Watts: power_W = torque * rpm * 2 * pi / 60.

    Relates directly to Power Failure (PWF): Mechanical power absorbed by the rotating spindle.
    A cutting process stalls if power drops below 3,500 W (cutter binding or motor torque loss),
    while electrical/mechanical surge failure occurs if power exceeds 9,000 W (extreme overload).
    """
    omega = rpm * (2.0 * math.pi / 60.0)
    return torque * omega


def compute_temp_diff_k(
    process_temp: pd.Series | float, air_temp: pd.Series | float
) -> pd.Series | float:
    """Compute thermodynamic gradient in Kelvin: temp_diff_K = process_temp - air_temp.

    Relates directly to Heat Dissipation Failure (HDF): The temperature delta governs convective
    and conductive heat dissipation away from the component. When temp_diff_K drops below 8.6 K
    concurrent with rotational speed < 1,380 RPM, the thermal dissipation capacity is overwhelmed,
    causing rapid heat accumulation and catastrophic thermal seizure.
    """
    return process_temp - air_temp


def compute_strain_min_nm(
    tool_wear: pd.Series | float, torque: pd.Series | float
) -> pd.Series | float:
    """Compute cumulative mechanical strain in min*Nm: strain_minNm = tool_wear * torque.

    Relates directly to Overstrain Failure (OSF): The product of accumulated operating tool wear
    and cutting torque reflects the total mechanical shear and torsional fatigue on the component.
    Overstrain failure occurs when this cumulative strain exceeds product material limits
    (11,000 min*Nm for Type L, 12,000 for Type M, and 13,000 for Type H).
    """
    return tool_wear * torque


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn transformer computing physical invariants and one-hot quality variants."""

    def __init__(self) -> None:
        self.feature_columns_: list[str] = [
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        """Fit transformer (stateless physics calculations)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute physics features and one-hot encode variant Type.

        Args:
            X: Input DataFrame with raw sensor telemetry or normalized columns.

        Returns:
            Transformed DataFrame containing canonical features.
        """
        df = X.copy()

        # Map raw column names with units/brackets if present
        for raw_col, clean_col in RAW_COLUMN_MAP.items():
            if raw_col in df.columns:
                df[clean_col] = df[raw_col]

        # 1. Mechanical Cutting Power (Watts)
        df["power_W"] = compute_power_w(df["torque_nm"], df["rotational_speed_rpm"])

        # 2. Thermodynamic Temperature Difference (Kelvin)
        df["temp_diff_K"] = compute_temp_diff_k(
            df["process_temperature_k"], df["air_temperature_k"]
        )

        # 3. Cumulative Mechanical Strain (min*Nm)
        df["strain_minNm"] = compute_strain_min_nm(df["tool_wear_min"], df["torque_nm"])

        # 4. One-Hot Encoding for Quality Variant Type (Type_L, Type_M, Type_H)
        # Quality variants define material fatigue limits and warranty tiering
        if "Type" in df.columns:
            df["Type_L"] = (df["Type"] == "L").astype(int)
            df["Type_M"] = (df["Type"] == "M").astype(int)
            df["Type_H"] = (df["Type"] == "H").astype(int)
        else:
            for col in ["Type_L", "Type_M", "Type_H"]:
                if col not in df.columns:
                    df[col] = 0

        # Verify strict zero data leakage
        verify_zero_data_leakage(self.feature_columns_)

        return df

    def get_feature_columns(self) -> list[str]:
        """Return canonical model input feature column names (100% leakage-free)."""
        verify_zero_data_leakage(self.feature_columns_)
        return list(self.feature_columns_)
