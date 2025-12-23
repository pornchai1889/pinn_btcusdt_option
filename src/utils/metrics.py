# src/utils/metrics.py
from typing import Dict, Any
import numpy as np


class MetricsCalculator:
    """
    Static utility class to handle all metric calculations.
    """

    @staticmethod
    def calculate_smape(true: np.ndarray, pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error (SMAPE).
        Range: 0-100%
        """
        denominator = (np.abs(true) + np.abs(pred)) / 2.0
        diff = np.abs(true - pred)
        return np.mean(diff / (denominator + 1e-8)) * 100

    @staticmethod
    def compute_kink_metrics(
        true_at_kink: np.ndarray, pred_at_kink: np.ndarray
    ) -> Dict[str, float]:
        """
        [New] Computes specific metrics for the Kink point (S=K, t=0).
        Args:
            true_at_kink (np.array): Ground truth at Kink (should be all 0s).
            pred_at_kink (np.array): Predicted values at Kink.
        Returns:
            dict: Dictionary with Kink-specific metrics.
        """
        diff = pred_at_kink - true_at_kink
        abs_diff = np.abs(diff)

        # Calculate MAE specifically for Kink (Mean Absolute Error)
        kink_mae = np.mean(abs_diff)

        return {"kink_mae": kink_mae}

    @staticmethod
    def compute_all_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """
        Computes standard regression metrics: RMSE, MAE, SMAPE, Bias, Corr, Max Error.
        """
        diff = pred - true
        abs_diff = np.abs(diff)

        # 1. RMSE
        rmse = np.sqrt(np.mean(diff**2))

        # 2. MAE
        mae = np.mean(abs_diff)

        # 3. SMAPE
        smape = MetricsCalculator.calculate_smape(true, pred)

        # 4. Bias
        bias = np.mean(diff)

        # 5. Corr
        if np.std(true) == 0 or np.std(pred) == 0:
            r_score = 0.0
        else:
            corr_matrix = np.corrcoef(true, pred)
            r_score = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

        # 6. Max Error
        max_error = np.max(abs_diff)

        return {
            "rmse": rmse,
            "mae": mae,
            "smape": smape,
            "bias": bias,
            "r_score": r_score,
            "max_error": max_error,
        }
