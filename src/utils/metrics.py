import numpy as np

class MetricsCalculator:
    """
    Static utility class to handle all metric calculations.
    """
    
    @staticmethod
    def calculate_smape(true, pred):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE).
        Range: 0-100%
        """
        denominator = (np.abs(true) + np.abs(pred)) / 2.0
        diff = np.abs(true - pred)
        # Avoid division by zero with small epsilon
        return np.mean(diff / (denominator + 1e-8)) * 100

    @staticmethod
    def compute_all_metrics(true, pred):
        """
        Computes standard regression metrics: RMSE, MAE, SMAPE, Bias, R, Max Error.
        Args:
            true (np.array): Ground truth values (e.g., V/K ratio).
            pred (np.array): Predicted values.
        Returns:
            dict: Dictionary containing all metrics.
        """
        diff = pred - true
        abs_diff = np.abs(diff)
        
        # 1. RMSE
        rmse = np.sqrt(np.mean(diff**2))
        
        # 2. MAE
        mae = np.mean(abs_diff)
        
        # 3. SMAPE
        smape = MetricsCalculator.calculate_smape(true, pred)
        
        # 4. Bias (Mean Signed Difference)
        bias = np.mean(diff)
        
        # 5. R (Correlation Coefficient)
        # Handle potential constant arrays to avoid NaN
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
            "max_error": max_error
        }