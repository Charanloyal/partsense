"""Data ingestion, validation, physics feature extraction, and splitting pipeline."""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from partsense.config import load_config
from partsense.data.validation import validate_raw_dataset, verify_zero_data_leakage
from partsense.features.engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_raw_ai4i_data(raw_path: Path, uci_id: int, fallback_url: str) -> pd.DataFrame:
    """Fetch AI4I 2020 dataset using ucimlrepo package with fallback to UCI static URL.

    Args:
        raw_path: Local destination path for CSV.
        uci_id: UCI Repository dataset ID (601).
        fallback_url: Direct download URL if ucimlrepo fails.

    Returns:
        DataFrame of raw dataset.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info("Found cached raw dataset at %s (%d bytes)", raw_path, raw_path.stat().st_size)
        return pd.read_csv(raw_path)

    logger.info("Fetching AI4I 2020 dataset (UCI id=%d) via ucimlrepo...", uci_id)
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=uci_id)
        # Combine features and targets into single unified dataframe
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        # ucimlrepo might return columns without units or with varying case; check and standardize
        if "Machine failure" not in df.columns and "machine_failure" in df.columns:
            df = df.rename(columns={"machine_failure": "Machine failure"})

        # If UDI column wasn't included by ucimlrepo, add it or fetch from fallback
        if "UDI" not in df.columns or len(df.columns) < 14:
            raise ValueError(
                "ucimlrepo returned partial schema, triggering fallback to full static CSV"
            )

        df.to_csv(raw_path, index=False)
        logger.info("Successfully fetched and saved %d records via ucimlrepo", len(df))
        return df
    except Exception as exc:
        logger.warning(
            "ucimlrepo fetch encountered '%s'. Falling back to static UCI URL: %s",
            exc,
            fallback_url,
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(fallback_url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(raw_path, "wb") as out_f:
            out_f.write(resp.read())

        df = pd.read_csv(raw_path)
        logger.info("Successfully downloaded and saved %d records via static URL", len(df))
        return df


def run_data_pipeline(config_path: str = "configs/config.yaml") -> dict[str, int]:
    """Execute complete data preparation pipeline: fetch, validate, engineer, split.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Record counts dictionary for train and test splits.
    """
    config = load_config(config_path)
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent.parent.parent

    raw_path = repo_root / config.data.raw_path
    train_path = repo_root / config.data.processed_train_path
    test_path = repo_root / config.data.processed_test_path

    # Step 1: Fetch raw dataset
    df_raw = fetch_raw_ai4i_data(raw_path, config.data.uci_id, config.data.fallback_url)

    # Step 2: Validate schema & physical boundaries
    audit = validate_raw_dataset(df_raw)
    logger.info("Raw data audit results: %s", audit)

    # Step 3: Feature Engineering (Physics features)
    logger.info("Engineering domain physics features...")
    engineer = FeatureEngineer()
    df_engineered = engineer.transform(df_raw)

    # Step 4: Strict Zero-Leakage Audit
    feature_cols = engineer.get_feature_columns()
    verify_zero_data_leakage(feature_cols)

    # Step 5: Stratified Train/Test Split (80% train / 20% test)
    logger.info(
        "Splitting dataset with test_size=%.2f, seed=%d, stratified on '%s'",
        config.data.test_size,
        config.project.random_seed,
        config.data.target_column,
    )
    train_df, test_df = train_test_split(
        df_engineered,
        test_size=config.data.test_size,
        random_state=config.project.random_seed,
        stratify=df_engineered[config.data.target_column],
    )

    # Step 6: Save processed parquet artifacts
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    train_failures = int(train_df[config.data.target_column].sum())
    test_failures = int(test_df[config.data.target_column].sum())
    logger.info(
        "Saved train split to %s (Rows: %d, Failures: %d, Rate: %.2f%%)",
        train_path,
        len(train_df),
        train_failures,
        (train_failures / len(train_df)) * 100,
    )
    logger.info(
        "Saved test split to %s (Rows: %d, Failures: %d, Rate: %.2f%%)",
        test_path,
        len(test_df),
        test_failures,
        (test_failures / len(test_df)) * 100,
    )

    return {
        "total_records": len(df_engineered),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_failures": train_failures,
        "test_failures": test_failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PartSense Data Pipeline")
    parser.add_argument("--stage", default="data", help="Pipeline stage to execute")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_data_pipeline(args.config)
