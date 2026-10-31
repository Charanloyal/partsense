"""Seeded synthetic warranty, parts catalog, sales, and service claims generator.

Generates an automotive-aftersales layer linking telemetric records from the AI4I
dataset to parts catalog families, operating hours, regional deployments,
monthly sales, and warranty claim payouts stored in DuckDB.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from partsense.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_aftersales_layer(
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Generate deterministic synthetic aftersales database and tables in DuckDB.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Summary dictionary of generated records.
    """
    config = load_config(config_path)
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent.parent

    # Fix random seed for strict determinism
    seed = config.project.random_seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    raw_path = repo_root / config.data.raw_path
    if not raw_path.exists():
        # Run data pipeline to ensure raw CSV is available
        from partsense.data.pipeline import run_data_pipeline

        logger.info("Raw data not found. Running data ingestion pipeline first...")
        run_data_pipeline(config_path)

    df_raw = pd.read_csv(raw_path)
    n_records = len(df_raw)

    synthetic_dir = repo_root / config.data.synthetic_dir
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    duckdb_file = repo_root / config.data.duckdb_path
    duckdb_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Parts Catalog Table
    catalog_configs = config.synthetic_aftersales.get("part_families", [])
    parts_catalog = pd.DataFrame(
        [
            {
                "part_family_id": i + 1,
                "part_family": p["name"],
                "unit_cost": float(p["unit_cost"]),
                "warranty_months": int(p["warranty_months"]),
                "warranty_miles": int(p["warranty_months"] * 1000),
            }
            for i, p in enumerate(catalog_configs)
        ]
    )

    part_names = parts_catalog["part_family"].tolist()
    part_cost_map = dict(
        zip(parts_catalog["part_family"], parts_catalog["unit_cost"], strict=False)
    )

    # 2. Components In Service (Linking 10,000 AI4I records)
    regions = config.synthetic_aftersales.get(
        "regions", ["North America", "Europe", "Asia-Pacific", "Latin America"]
    )
    start_date = datetime.strptime(
        config.synthetic_aftersales.get("start_date", "2024-01-01"), "%Y-%m-%d"
    )
    total_months = int(config.synthetic_aftersales.get("months", 24))

    # Deterministic assignment of each record to part family and region
    assigned_parts = rng.choice(part_names, size=n_records)
    assigned_regions = rng.choice(regions, size=n_records)

    # Timeline assignment: randomly placed within 24 months
    install_day_offsets = rng.integers(0, total_months * 30, size=n_records)
    install_dates = [start_date + timedelta(days=int(d)) for d in install_day_offsets]

    # ASSUMPTION: Operating hours = Tool wear [min] / 60.0
    operating_hours = np.round(df_raw["Tool wear [min]"] / 60.0, 2)

    components_df = pd.DataFrame(
        {
            "component_id": df_raw["UDI"],
            "product_id": df_raw["Product ID"],
            "quality_type": df_raw["Type"],
            "part_family": assigned_parts,
            "region": assigned_regions,
            "installation_date": [d.strftime("%Y-%m-%d") for d in install_dates],
            "operating_hours": operating_hours,
            "machine_failure": df_raw["Machine failure"],
            "TWF": df_raw["TWF"],
            "HDF": df_raw["HDF"],
            "PWF": df_raw["PWF"],
            "OSF": df_raw["OSF"],
            "RNF": df_raw["RNF"],
            "air_temperature_k": df_raw["Air temperature [K]"],
            "process_temperature_k": df_raw["Process temperature [K]"],
            "rotational_speed_rpm": df_raw["Rotational speed [rpm]"],
            "torque_nm": df_raw["Torque [Nm]"],
            "tool_wear_min": df_raw["Tool wear [min]"],
        }
    )

    # 3. Monthly Parts Sales Table (Monthly units sold per part per region)
    sales_records = []
    base_sales_volume = {
        "Wheel Bearing": 850,
        "Water Pump": 620,
        "Alternator": 480,
        "Starter Motor": 510,
        "Brake Caliper": 920,
        "Fuel Pump": 390,
    }
    region_multipliers = {
        "North America": 1.25,
        "Europe": 1.10,
        "Asia-Pacific": 1.35,
        "Latin America": 0.70,
    }

    for month_idx in range(total_months):
        month_date = start_date + timedelta(days=month_idx * 30)
        month_str = month_date.strftime("%Y-%m")

        for part in part_names:
            base = base_sales_volume.get(part, 500)
            unit_price = part_cost_map.get(part, 100.0)

            for reg in regions:
                reg_mult = region_multipliers.get(reg, 1.0)
                # Add seasonal / random variation
                seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * month_idx / 12)
                noise = rng.normal(1.0, 0.05)
                units_sold = int(round(base * reg_mult * seasonal * noise))

                sales_records.append(
                    {
                        "sale_month": month_str,
                        "part_family": part,
                        "region": reg,
                        "units_sold": units_sold,
                        "unit_price": unit_price,
                        "revenue": round(units_sold * unit_price, 2),
                    }
                )

    parts_sales = pd.DataFrame(sales_records)

    # 4. Service Records & Warranty Claims Table (Derived from actual failures)
    failure_indices = df_raw.index[df_raw["Machine failure"] == 1].tolist()
    claim_records = []

    # Failure mode specific lognormal cost parameters
    # PWF (electrical surge) and OSF (shattered casting) cost more to repair
    mode_cost_params = {
        "PWF": (6.9, 0.35),  # Median ~$1,000
        "OSF": (6.7, 0.40),  # Median ~$810
        "HDF": (6.4, 0.35),  # Median ~$600
        "TWF": (5.9, 0.30),  # Median ~$360
        "RNF": (6.2, 0.40),  # Median ~$490
        "NORMAL": (5.8, 0.30),
    }

    for idx in failure_indices:
        comp_row = components_df.iloc[idx]
        fail_date = datetime.strptime(comp_row["installation_date"], "%Y-%m-%d") + timedelta(
            days=int(rng.integers(10, 180))
        )

        # Primary failure mode
        if comp_row["TWF"] == 1:
            primary_mode = "TWF"
        elif comp_row["HDF"] == 1:
            primary_mode = "HDF"
        elif comp_row["PWF"] == 1:
            primary_mode = "PWF"
        elif comp_row["OSF"] == 1:
            primary_mode = "OSF"
        elif comp_row["RNF"] == 1:
            primary_mode = "RNF"
        else:
            primary_mode = "UNSPECIFIED"

        # Claim lag: 7 to 45 days after failure
        claim_lag_days = int(rng.integers(7, 46))
        claim_date = fail_date + timedelta(days=claim_lag_days)

        # Cost drawn from lognormal distribution
        mu, sigma = mode_cost_params.get(primary_mode, (6.3, 0.35))
        claim_amount = float(np.round(rng.lognormal(mu, sigma), 2))

        # Check warranty coverage based on part warranty period
        part_info = parts_catalog[parts_catalog["part_family"] == comp_row["part_family"]].iloc[0]
        days_in_service = (
            fail_date - datetime.strptime(comp_row["installation_date"], "%Y-%m-%d")
        ).days
        in_warranty = int(days_in_service <= (part_info["warranty_months"] * 30))

        claim_records.append(
            {
                "claim_id": f"CLM-{1000 + len(claim_records) + 1}",
                "component_id": int(comp_row["component_id"]),
                "product_id": comp_row["product_id"],
                "quality_type": comp_row["quality_type"],
                "part_family": comp_row["part_family"],
                "region": comp_row["region"],
                "failure_date": fail_date.strftime("%Y-%m-%d"),
                "claim_date": claim_date.strftime("%Y-%m-%d"),
                "claim_month": claim_date.strftime("%Y-%m"),
                "claim_lag_days": claim_lag_days,
                "primary_failure_mode": primary_mode,
                "operating_hours": float(comp_row["operating_hours"]),
                "claim_amount": claim_amount,
                "is_approved_warranty": in_warranty,
            }
        )

    warranty_claims = pd.DataFrame(claim_records)

    # 5. Export to Parquet files
    parts_catalog.to_parquet(synthetic_dir / "parts_catalog.parquet", index=False)
    components_df.to_parquet(synthetic_dir / "components_in_service.parquet", index=False)
    parts_sales.to_parquet(synthetic_dir / "parts_sales.parquet", index=False)
    warranty_claims.to_parquet(synthetic_dir / "warranty_claims.parquet", index=False)

    # 6. Initialize DuckDB Database and Tables
    logger.info("Initializing DuckDB database at %s...", duckdb_file)
    con = duckdb.connect(str(duckdb_file))

    con.execute("CREATE OR REPLACE TABLE parts_catalog AS SELECT * FROM parts_catalog")
    con.execute("CREATE OR REPLACE TABLE components_in_service AS SELECT * FROM components_df")
    con.execute("CREATE OR REPLACE TABLE parts_sales AS SELECT * FROM parts_sales")
    con.execute("CREATE OR REPLACE TABLE warranty_claims AS SELECT * FROM warranty_claims")

    # Create Views for Clean SQL Analytics
    con.execute("""
        CREATE OR REPLACE VIEW v_active_failures AS
        SELECT * FROM components_in_service WHERE machine_failure = 1
    """)

    con.close()
    logger.info("Successfully populated DuckDB tables and exported Parquet files.")

    return {
        "status": "SUCCESS",
        "parts_catalog_count": len(parts_catalog),
        "components_count": len(components_df),
        "total_failures": int(components_df["machine_failure"].sum()),
        "parts_sales_records": len(parts_sales),
        "total_units_sold": int(parts_sales["units_sold"].sum()),
        "warranty_claims_count": len(warranty_claims),
        "total_warranty_cost": round(float(warranty_claims["claim_amount"].sum()), 2),
        "duckdb_path": str(duckdb_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PartSense Synthetic Aftersales Generator")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    res = generate_synthetic_aftersales_layer(args.config)
    logger.info("Synthetic Generation Output: %s", res)
