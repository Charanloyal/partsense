"""KPI query runner executing SQL files against DuckDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from partsense.config import load_config

logger = logging.getLogger(__name__)


class KPIRunner:
    """Executes SQL queries against the PartSense DuckDB analytics database."""

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        duckdb_path: str | Path | None = None,
        sql_dir: str | Path | None = None,
    ) -> None:
        current_dir = Path(__file__).resolve().parent
        self.repo_root = current_dir.parent.parent.parent

        if duckdb_path is not None:
            self.duckdb_path = Path(duckdb_path)
        else:
            self.config = load_config(config_path)
            self.duckdb_path = self.repo_root / self.config.data.duckdb_path

        if sql_dir is not None:
            self.sql_dir = Path(sql_dir)
        else:
            self.sql_dir = self.repo_root / "sql"

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Establish read-only connection to DuckDB."""
        if not self.duckdb_path.exists():
            raise FileNotFoundError(
                f"DuckDB database not found at {self.duckdb_path}. Run 'tasks.py synthetic' first."
            )
        return duckdb.connect(str(self.duckdb_path), read_only=True)

    def run_query(self, query_or_name: str) -> pd.DataFrame:
        """Execute query either as raw SQL string or as a query file name."""
        # Check if query_or_name is a known SQL file name in sql_dir
        sql_filename = query_or_name if query_or_name.endswith(".sql") else f"{query_or_name}.sql"
        candidate_path = self.sql_dir / sql_filename
        if candidate_path.exists():
            with open(candidate_path, encoding="utf-8") as f:
                sql_to_run = f.read()
        else:
            sql_to_run = query_or_name

        con = self.get_connection()
        try:
            return con.execute(sql_to_run).df()
        finally:
            con.close()

    def run_sql_file(self, sql_filename: str) -> pd.DataFrame:
        """Read .sql file from sql/ directory and execute."""
        return self.run_query(sql_filename)

    def get_all_kpis(self) -> dict[str, Any]:
        """Execute all standard KPI queries and return dictionary of results."""
        return {
            "failure_rate_by_dimension": self.run_query("failure_rate_by_dimension"),
            "mtbf_by_part_family": self.run_query("mtbf_by_part_family"),
            "warranty_claims_per_1000_sales": self.run_query("warranty_claims_per_1000_sales"),
            "warranty_cost_per_unit_sold": self.run_query("warranty_cost_per_unit_sold"),
            "monthly_claim_trends": self.run_query("monthly_claim_trends"),
            "pareto_failure_modes": self.run_query("pareto_failure_modes"),
            "top_parts_at_risk": self.run_query("top_parts_at_risk"),
        }


if __name__ == "__main__":
    runner = KPIRunner()
    print("=" * 70)
    print("  PARTSENSE DUCKDB AFTERSALES KPI SUMMARY (SYNTHETIC BENCHMARK)")
    print("=" * 70)
    mtbf = runner.run_query("mtbf_by_part_family")
    print("\n--- MTBF by Part Family (Operating Hours = Tool Wear Min / 60) ---")
    print(mtbf.to_string(index=False))

    iptv = runner.run_query("warranty_claims_per_1000_sales")
    print("\n--- Warranty Claims per 1,000 Units Sold (IPTV) ---")
    print(iptv.to_string(index=False))

    pareto = runner.run_query("pareto_failure_modes")
    print("\n--- Pareto 80/20 Failure Mode Spend ---")
    print(pareto.to_string(index=False))
