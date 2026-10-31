"""Tests for domain physics feature engineering calculations and docstrings."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from partsense.features.engineer import (
    FeatureEngineer,
    compute_power_w,
    compute_strain_min_nm,
    compute_temp_diff_k,
)


def test_feature_engineer_physics_calculations():
    """Verify exact numerical physics calculations for power, temperature diff, and strain."""
    engineer = FeatureEngineer()

    # Known sample inputs
    rpm = 1500.0
    torque = 40.0
    air_temp = 300.0
    proc_temp = 310.0
    tool_wear = 120.0

    df_raw = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    df_feats = engineer.transform(df_raw)

    # 1. Spindle Power: P = torque * rpm * 2 * pi / 60
    expected_power = torque * (rpm * 2.0 * math.pi / 60.0)
    assert pytest.approx(df_feats["power_W"].iloc[0], rel=1e-4) == expected_power

    # 2. Temperature Gradient: delta_T = proc_temp - air_temp
    expected_delta_t = proc_temp - air_temp
    assert pytest.approx(df_feats["temp_diff_K"].iloc[0], rel=1e-4) == expected_delta_t

    # 3. Tool Strain: tool_wear * torque
    expected_strain = tool_wear * torque
    assert pytest.approx(df_feats["strain_minNm"].iloc[0], rel=1e-4) == expected_strain

    # 4. One-hot Type: 'M' should produce Type_M=1, Type_L=0, Type_H=0
    assert df_feats["Type_M"].iloc[0] == 1
    assert df_feats["Type_L"].iloc[0] == 0
    assert df_feats["Type_H"].iloc[0] == 0


def test_feature_engineer_docstrings_reference_failure_mechanisms():
    """Verify that feature engineering functions document physical links to HDF, PWF, OSF."""
    power_doc = compute_power_w.__doc__ or ""
    temp_doc = compute_temp_diff_k.__doc__ or ""
    strain_doc = compute_strain_min_nm.__doc__ or ""

    assert "PWF" in power_doc or "Power Failure" in power_doc
    assert "HDF" in temp_doc or "Heat Dissipation" in temp_doc
    assert "OSF" in strain_doc or "Overstrain Failure" in strain_doc
