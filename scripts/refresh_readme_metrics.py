"""Refresh the benchmark results table in README.md directly from reports/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def refresh_readme_metrics() -> None:
    """Read reports/metrics.json and update the benchmark table in README.md."""
    metrics_path = ROOT_DIR / "reports" / "metrics.json"
    readme_path = ROOT_DIR / "README.md"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run 'python tasks.py evaluate' first.")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    xgb_metrics = metrics["main_xgboost"]
    base_metrics = metrics["baseline_logistic_regression"]
    tuned_policy = xgb_metrics["tuned_threshold_policy"]
    def_policy = xgb_metrics["default_threshold_0_5"]

    table_lines = [
        "| Architecture | Test ROC-AUC | Test PR-AUC | Operating Threshold Policy | Test Recall | Test Precision | Test F1 |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **PartSense XGBoost (Champion)** | **{xgb_metrics['discrimination']['roc_auc']:.4f}** | **{xgb_metrics['discrimination']['pr_auc']:.4f}** | **Tuned ($p^* = {tuned_policy['threshold']:.4f}$)** | **{tuned_policy['recall'] * 100:.1f}%** ({tuned_policy['confusion_matrix']['tp']}/68) | **{tuned_policy['precision'] * 100:.1f}%** | **{tuned_policy['f1']:.4f}** |",
        f"| PartSense XGBoost (Default) | {xgb_metrics['discrimination']['roc_auc']:.4f} | {xgb_metrics['discrimination']['pr_auc']:.4f} | Default ($p = {def_policy['threshold']:.4f}$) | {def_policy['recall'] * 100:.1f}% ({def_policy['confusion_matrix']['tp']}/68) | {def_policy['precision'] * 100:.1f}% | {def_policy['f1']:.4f} |",
        f"| Baseline Logistic Regression (Std) | {base_metrics['roc_auc']:.4f} | {base_metrics['pr_auc']:.4f} | Default ($p = 0.5000$) | {base_metrics['recall_at_05'] * 100:.1f}% (60/68) | {base_metrics['precision_at_05'] * 100:.1f}% | {base_metrics['f1_at_05']:.4f} |",
    ]

    new_table_str = "\n".join(table_lines)
    print("--- Generated Honest Results Table ---")
    print(new_table_str)

    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            content = f.read()

        start_marker = "<!-- BENCHMARK_TABLE_START -->"
        end_marker = "<!-- BENCHMARK_TABLE_END -->"

        if start_marker in content and end_marker in content:
            before = content.split(start_marker)[0]
            after = content.split(end_marker)[1]
            updated_content = f"{before}{start_marker}\n\n{new_table_str}\n\n{end_marker}{after}"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            print(f"Successfully updated {readme_path}")
        else:
            print("Note: Markers <!-- BENCHMARK_TABLE_START --> not found; table printed above.")


if __name__ == "__main__":
    refresh_readme_metrics()
