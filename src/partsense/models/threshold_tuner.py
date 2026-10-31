"""Out-of-fold decision threshold tuner for class-imbalanced binary classification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


class ThresholdTuner:
    """Tunes decision threshold on predicted probabilities to maximize recall at a precision floor."""

    def __init__(self, min_precision_floor: float = 0.50) -> None:
        """Initialize tuner with minimum acceptable precision constraint."""
        self.min_precision_floor = min_precision_floor
        self.best_threshold_: float = 0.50

    def tune_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, Any]:
        """Find decision threshold maximizing recall subject to precision >= floor.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_prob: Predicted failure probabilities.

        Returns:
            Dictionary with optimal threshold, achievable precision, recall, and F1.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        # precision_recall_curve returns n+1 precisions/recalls, n thresholds
        valid_indices = np.where(precisions[:-1] >= self.min_precision_floor)[0]

        if len(valid_indices) > 0:
            # Among thresholds meeting the precision constraint, pick the one with maximum recall
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            best_th = float(thresholds[best_idx])
            achieved_p = float(precisions[best_idx])
            achieved_r = float(recalls[best_idx])
            f1 = float(2 * (achieved_p * achieved_r) / (achieved_p + achieved_r + 1e-10))
        else:
            # Fallback to threshold that maximizes F1 score if floor is too stringent
            f1_scores = (
                2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            )
            best_idx = int(np.argmax(f1_scores))
            best_th = float(thresholds[best_idx])
            achieved_p = float(precisions[best_idx])
            achieved_r = float(recalls[best_idx])
            f1 = float(f1_scores[best_idx])
            logger.warning(
                "Precision floor %.2f could not be strictly met. Fallback to max-F1 threshold: %.4f",
                self.min_precision_floor,
                best_th,
            )

        self.best_threshold_ = best_th

        logger.info(
            "Threshold tuned on OOF predictions: th=%.4f (Precision=%.4f, Recall=%.4f, F1=%.4f)",
            best_th,
            achieved_p,
            achieved_r,
            f1,
        )

        return {
            "optimal_threshold": round(best_th, 4),
            "min_precision_floor": self.min_precision_floor,
            "achieved_precision": round(achieved_p, 4),
            "achieved_recall": round(achieved_r, 4),
            "achieved_f1": round(f1, 4),
        }
