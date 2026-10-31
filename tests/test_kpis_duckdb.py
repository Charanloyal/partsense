"""Tests for DuckDB KPI queries and KPIRunner execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from partsense.kpis.runner import KPIRunner

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def kpi_runner() -> KPIRunner:
    """Fixture providing initialized KPIRunner with generated DuckDB database."""
    duckdb_path = ROOT_DIR / "data" / "synthetic" / "aftersales.duckdb"
    sql_dir = ROOT_DIR / "sql"
    assert duckdb_path.exists(), "aftersales.duckdb must exist to run KPI tests."
    return KPIRunner(duckdb_path=duckdb_path, sql_dir=sql_dir)


@pytest.mark.parametrize(
    "query_name",
    [
        "failure_rate_by_dimension",
        "mtbf_by_part_family",
        "warranty_claims_per_1000_sales",
        "warranty_cost_per_unit_sold",
        "monthly_claim_trends",
        "pareto_failure_modes",
        "top_parts_at_risk",
    ],
)
def test_all_sql_queries_execute_and_return_data(kpi_runner: KPIRunner, query_name: str):
    """Verify each of the 7 .sql query files executes cleanly and returns rows."""
    df = kpi_runner.run_query(query_name)
    assert not df.empty, f"Query '{query_name}' returned empty DataFrame."
    assert len(df) > 0


def test_mtbf_operating_hours_validity(kpi_runner: KPIRunner):
    """Verify MTBF query produces realistic positive operating hours."""
    df = kpi_runner.run_query("mtbf_by_part_family")
    assert "mtbf_hours" in df.columns
    assert (df["mtbf_hours"] > 0).all()


def test_pareto_failure_modes_cumulative_percentage(kpi_runner: KPIRunner):
    """Verify Pareto query produces monotonic cumulative percentage ending near 100%."""
    df = kpi_runner.run_query("pareto_failure_modes")
    assert "cumulative_pct" in df.columns
    # Check monotonic increase
    assert df["cumulative_pct"].is_monotonic_increasing
    assert pytest.approx(df["cumulative_pct"].iloc[-1], abs=0.5) == 100.0
