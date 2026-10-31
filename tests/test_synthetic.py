"""Tests for synthetic aftersales layer generation and integrity."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from partsense.synthetic import generate_synthetic_aftersales_layer

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_synthetic_aftersales_database_tables():
    """Verify generated DuckDB contains all required aftersales tables and expected columns."""
    duckdb_path = ROOT_DIR / "data" / "synthetic" / "aftersales.duckdb"
    assert duckdb_path.exists(), "aftersales.duckdb must exist."

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

    expected_tables = {
        "parts_catalog",
        "components_in_service",
        "parts_sales",
        "warranty_claims",
    }
    assert expected_tables.issubset(set(tables))

    # Verify 6 parts families in parts_catalog
    catalog_df = conn.execute("SELECT * FROM parts_catalog").fetchdf()
    assert len(catalog_df) == 6
    expected_families = {
        "Wheel Bearing",
        "Water Pump",
        "Alternator",
        "Starter Motor",
        "Brake Caliper",
        "Fuel Pump",
    }
    assert set(catalog_df["part_family"].values) == expected_families

    # Verify operating hours assumption: operating_hours = tool_wear_min / 60
    comp_sample = conn.execute(
        "SELECT operating_hours, tool_wear_min FROM components_in_service LIMIT 5"
    ).fetchdf()
    for _, row in comp_sample.iterrows():
        expected_hrs = round(row["tool_wear_min"] / 60.0, 2)
        assert pytest.approx(row["operating_hours"], abs=0.02) == expected_hrs

    conn.close()


def test_synthetic_generator_deterministic_execution():
    """Verify that generate_synthetic_aftersales_layer returns SUCCESS status and valid counts."""
    config_path = ROOT_DIR / "configs" / "config.yaml"
    summary = generate_synthetic_aftersales_layer(str(config_path))

    assert summary["status"] == "SUCCESS"
    assert summary["parts_catalog_count"] == 6
    assert summary["components_count"] == 10000
    assert summary["warranty_claims_count"] == 339
    assert summary["total_warranty_cost"] > 0
