"""Training pipeline for PartSense ML models: XGBoost, OvR Diagnoser, Isolation Forest, and Baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from partsense.config import load_config
from partsense.data.validation import verify_zero_data_leakage
from partsense.features.engineer import FeatureEngineer
from partsense.models.threshold_tuner import ThresholdTuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_pipeline(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Execute complete training pipeline and serialize trained model artifacts.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Summary dictionary of cross-validation results and serialized artifact paths.
    """
    config = load_config(config_path)
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent.parent.parent

    train_path = repo_root / config.data.processed_train_path
    if not train_path.exists():
        from partsense.data.pipeline import run_data_pipeline

        logger.info("Processed training data not found. Running data pipeline first...")
        run_data_pipeline(config_path)

    train_df = pd.read_parquet(train_path)

    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns()

    # Enforce strict zero data leakage
    verify_zero_data_leakage(feature_cols)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[config.data.target_column].to_numpy()

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    logger.info(
        "Training data: N=%d, Features=%d, Failures=%d (%.2f%%). scale_pos_weight=%.2f",
        len(X_train),
        len(feature_cols),
        pos_count,
        (pos_count / len(X_train)) * 100,
        scale_pos_weight,
    )

    # 1. 5-Fold Stratified Cross-Validation on Training Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.project.random_seed)

    # Baseline Pipeline: Scaler + LogisticRegression(class_weight='balanced')
    baseline_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=config.project.random_seed,
                ),
            ),
        ]
    )

    # Main Model: XGBoost with scale_pos_weight
    xgb_params = config.models.get("binary_classifier", {}).get("xgb_params", {})
    main_xgb = xgb.XGBClassifier(
        n_estimators=int(xgb_params.get("n_estimators", 150)),
        max_depth=int(xgb_params.get("max_depth", 4)),
        learning_rate=float(xgb_params.get("learning_rate", 0.05)),
        subsample=float(xgb_params.get("subsample", 0.8)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
        scale_pos_weight=scale_pos_weight,
        random_state=config.project.random_seed,
        eval_metric="logloss",
        n_jobs=-1,
    )

    logger.info("--- Running 5-Fold Stratified CV on Training Split ---")
    oof_xgb_probs = np.zeros(len(X_train))
    oof_base_probs = np.zeros(len(X_train))

    cv_xgb_rocs = []
    cv_xgb_prs = []
    cv_base_rocs = []
    cv_base_prs = []

    for _fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = y_train[trn_idx], y_train[val_idx]

        # Train Baseline
        baseline_clf.fit(X_trn, y_trn)
        base_val_prob = baseline_clf.predict_proba(X_val)[:, 1]
        oof_base_probs[val_idx] = base_val_prob
        cv_base_rocs.append(roc_auc_score(y_val, base_val_prob))
        cv_base_prs.append(average_precision_score(y_val, base_val_prob))

        # Train XGBoost
        main_xgb.fit(X_trn, y_trn)
        xgb_val_prob = main_xgb.predict_proba(X_val)[:, 1]
        oof_xgb_probs[val_idx] = xgb_val_prob
        cv_xgb_rocs.append(roc_auc_score(y_val, xgb_val_prob))
        cv_xgb_prs.append(average_precision_score(y_val, xgb_val_prob))

    logger.info(
        "[Baseline LogReg] CV ROC-AUC: %.4f (+/- %.4f) | CV PR-AUC: %.4f",
        float(np.mean(cv_base_rocs)),
        float(np.std(cv_base_rocs)),
        float(np.mean(cv_base_prs)),
    )
    logger.info(
        "[Champion XGBoost] CV ROC-AUC: %.4f (+/- %.4f) | CV PR-AUC: %.4f",
        float(np.mean(cv_xgb_rocs)),
        float(np.std(cv_xgb_rocs)),
        float(np.mean(cv_xgb_prs)),
    )

    # 2. Decision Threshold Tuning on Out-of-Fold (OOF) Predictions
    min_prec_floor = float(
        config.models.get("binary_classifier", {})
        .get("threshold_tuning", {})
        .get("min_precision_floor", 0.50)
    )
    tuner = ThresholdTuner(min_precision_floor=min_prec_floor)
    tuning_result = tuner.tune_threshold(y_train, oof_xgb_probs)
    tuned_threshold = tuning_result["optimal_threshold"]

    # 3. Fit Champion XGBoost on Entire 80% Training Split
    logger.info("Fitting final champion XGBoost on entire training dataset...")
    final_xgb = xgb.XGBClassifier(
        n_estimators=int(xgb_params.get("n_estimators", 150)),
        max_depth=int(xgb_params.get("max_depth", 4)),
        learning_rate=float(xgb_params.get("learning_rate", 0.05)),
        subsample=float(xgb_params.get("subsample", 0.8)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
        scale_pos_weight=scale_pos_weight,
        random_state=config.project.random_seed,
        eval_metric="logloss",
        n_jobs=-1,
    )
    final_xgb.fit(X_train, y_train)

    # Fit final baseline model on full training split
    final_baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=config.project.random_seed,
                ),
            ),
        ]
    )
    final_baseline.fit(X_train, y_train)

    # 4. Train One-vs-Rest Failure Mode Models for TWF, HDF, PWF, OSF
    learnable_modes = config.models.get("failure_mode_ovr", {}).get(
        "learnable_modes", ["TWF", "HDF", "PWF", "OSF"]
    )
    ovr_models: dict[str, xgb.XGBClassifier] = {}
    ovr_cv_metrics: dict[str, dict[str, float]] = {}

    logger.info("--- Training One-vs-Rest XGBoost Failure Mode Diagnosers ---")
    for mode in learnable_modes:
        y_mode = train_df[mode].to_numpy()
        pos_m = int(np.sum(y_mode == 1))
        neg_m = int(np.sum(y_mode == 0))
        spw_m = float(neg_m / pos_m) if pos_m > 0 else 1.0

        mode_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=spw_m,
            random_state=config.project.random_seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
        mode_xgb.fit(X_train, y_mode)
        ovr_models[mode] = mode_xgb

        # Quick in-sample or CV score for reporting
        pred_p = mode_xgb.predict_proba(X_train)[:, 1]
        ovr_cv_metrics[mode] = {
            "pos_cases": pos_m,
            "train_roc_auc": round(float(roc_auc_score(y_mode, pred_p)), 4),
            "train_pr_auc": round(float(average_precision_score(y_mode, pred_p)), 4),
        }
        logger.info(
            "Mode [%s]: %d cases. Train ROC-AUC=%.4f, PR-AUC=%.4f",
            mode,
            pos_m,
            ovr_cv_metrics[mode]["train_roc_auc"],
            ovr_cv_metrics[mode]["train_pr_auc"],
        )

    # 5. Anomaly Detection: Isolation Forest on Normal-Only Training Rows
    logger.info(
        "--- Fitting Isolation Forest on NORMAL-ONLY Training Records (Machine failure == 0) ---"
    )
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.034,
        random_state=config.project.random_seed,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_normal)
    logger.info(
        "Isolation Forest fitted successfully on %d normal training records.", len(X_train_normal)
    )

    # 6. Save Model Artifacts
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    xgb_path = repo_root / config.models["binary_classifier"]["artifact_path"]
    base_path = repo_root / config.models["baseline"]["artifact_path"]
    ovr_path = repo_root / config.models["failure_mode_ovr"]["artifact_path"]
    iso_path = repo_root / config.models["anomaly_detector"]["artifact_path"]
    feat_path = models_dir / "feature_names.joblib"
    th_path = models_dir / "threshold_info.joblib"

    joblib.dump(final_xgb, xgb_path)
    joblib.dump(final_baseline, base_path)
    joblib.dump(ovr_models, ovr_path)
    joblib.dump(iso_forest, iso_path)
    joblib.dump(feature_cols, feat_path)
    joblib.dump(tuning_result, th_path)

    logger.info("Serialized XGBoost champion to %s", xgb_path)
    logger.info("Serialized Baseline to %s", base_path)
    logger.info("Serialized OvR Diagnoser to %s", ovr_path)
    logger.info("Serialized Isolation Forest to %s", iso_path)

    return {
        "status": "SUCCESS",
        "cv_xgb_roc_auc": round(float(np.mean(cv_xgb_rocs)), 4),
        "cv_xgb_pr_auc": round(float(np.mean(cv_xgb_prs)), 4),
        "cv_base_roc_auc": round(float(np.mean(cv_base_rocs)), 4),
        "cv_base_pr_auc": round(float(np.mean(cv_base_prs)), 4),
        "tuned_threshold": tuned_threshold,
        "ovr_metrics": ovr_cv_metrics,
        "feature_count": len(feature_cols),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PartSense Model Training")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    train_pipeline(args.config)
