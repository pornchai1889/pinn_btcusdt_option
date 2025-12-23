import sys
import os
import pytest
import numpy as np
from datetime import date, datetime
import pytz
from typing import Dict, Any

# --- Environment Setup ---
# Add the project root to sys.path to ensure imports work correctly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import MetricsCalculator
from src.utils.date_utils import DateUtils


# =============================================================================
# Test Suite 1: Metrics Calculator (Evaluation Standards)
# =============================================================================
class TestMetricsCalculator:

    def test_smape_logic(self) -> None:
        """
        Verifies Symmetric Mean Absolute Percentage Error (SMAPE) logic.
        Formula: 100 * mean(|diff| / (|true| + |pred|)/2)
        """
        # Case 1: Exact Match -> SMAPE should be 0
        true = np.array([100.0, 200.0])
        pred = np.array([100.0, 200.0])
        smape = MetricsCalculator.calculate_smape(true, pred)
        assert smape == 0.0, "SMAPE must be 0 for exact matches"

        # Case 2: Known Deviation
        # True=100, Pred=110 -> Diff=10, Avg=105 -> 10/105 * 100 approx 9.52%
        true_2 = np.array([100.0])
        pred_2 = np.array([110.0])
        smape_2 = MetricsCalculator.calculate_smape(true_2, pred_2)

        expected = (10.0 / 105.0) * 100.0
        np.testing.assert_allclose(
            smape_2, expected, atol=1e-5, err_msg="SMAPE calculation mismatch"
        )

    def test_smape_zero_handling(self) -> None:
        """
        Ensures SMAPE handles zero values gracefully (epsilon check).
        Prevents DivisionByZero errors when True=0 and Pred=0.
        """
        true = np.array([0.0, 0.0])
        pred = np.array([0.0, 0.0])

        # Should not crash and return 0 (due to epsilon logic)
        smape = MetricsCalculator.calculate_smape(true, pred)
        assert smape < 1e-5, "SMAPE should be effectively 0 when inputs are 0"

    def test_kink_metrics_computation(self) -> None:
        """
        Verifies specific metrics for the 'Kink' point (S=K).
        """
        # Ground truth at Kink is always 0 (Payoff of ATM option at maturity)
        true_at_kink = np.zeros(5)
        # Predictions have some error
        pred_at_kink = np.array([0.1, -0.1, 0.2, 0.0, 0.1])

        metrics = MetricsCalculator.compute_kink_metrics(true_at_kink, pred_at_kink)

        # Expected MAE: (|0.1| + |-0.1| + |0.2| + |0| + |0.1|) / 5 = 0.5 / 5 = 0.1
        expected_mae = 0.1

        assert "kink_mae" in metrics, "Result dictionary missing 'kink_mae' key"
        np.testing.assert_allclose(metrics["kink_mae"], expected_mae, atol=1e-6)

    def test_all_metrics_structure(self) -> None:
        """
        Smoke test to ensure 'compute_all_metrics' returns all required keys
        and valid values.
        """
        true = np.array([100.0, 105.0])
        pred = np.array([102.0, 104.0])

        metrics = MetricsCalculator.compute_all_metrics(true, pred)

        required_keys = ["rmse", "mae", "smape", "bias", "r_score", "max_error"]
        for key in required_keys:
            assert key in metrics, f"Missing metric key: {key}"
            assert not np.isnan(metrics[key]), f"Metric {key} returned NaN"


# =============================================================================
# Test Suite 2: Date Utilities (Financial Time Logic)
# =============================================================================
class TestDateUtils:

    def test_symbol_date_parsing(self) -> None:
        """
        Verifies parsing of Binance-style date codes in symbols.
        Format: YYMMDD
        """
        # Test Case: 26th Dec 2025
        symbol = "BTC-251226-90000-C"
        parsed_date = DateUtils.parse_symbol_date(symbol)

        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 26

    def test_invalid_symbol_parsing(self) -> None:
        """
        Ensures proper error handling for malformed symbols.
        """
        with pytest.raises(ValueError):
            DateUtils.parse_symbol_date("INVALID-FORMAT")

    def test_expiration_type_logic(self) -> None:
        """
        Verifies classification logic (Daily, Weekly, Monthly, Quarterly).
        Based on deterministic calendar rules.
        """
        # 1. Quarterly: Last Friday of Quarter Month (Mar, Jun, Sep, Dec)
        # Dec 26, 2025 is a Friday and is the Last Friday of Dec 2025.
        sym_quarterly = "BTC-251226-90000-C"
        assert DateUtils.get_expiration_type(sym_quarterly) == "Quarterly"

        # 2. Monthly: Last Friday of non-quarter month (e.g., Jan, Feb)
        # Jan 31, 2025 is a Friday and Last Friday of Jan.
        sym_monthly = "BTC-250131-90000-C"
        assert DateUtils.get_expiration_type(sym_monthly) == "Monthly"

        # 3. Weekly: A Friday that is NOT the last Friday
        # Dec 19, 2025 is a Friday (Last is 26th).
        sym_weekly = "BTC-251219-90000-C"
        assert DateUtils.get_expiration_type(sym_weekly) == "Weekly"

        # 4. Daily: Any non-Friday
        # Dec 25, 2025 is a Thursday.
        sym_daily = "BTC-251225-90000-C"
        assert DateUtils.get_expiration_type(sym_daily) == "Daily"

    def test_contract_period_calculation(self) -> None:
        """
        Verifies that start/end timestamps are calculated and are logical.
        Start < End.
        """
        symbol = "BTC-251226-90000-C"  # Quarterly
        exp_type = "Quarterly"

        start_ms, end_ms = DateUtils.calculate_contract_period(symbol, exp_type)

        assert start_ms is not None and end_ms is not None
        assert isinstance(start_ms, int)
        assert isinstance(end_ms, int)

        # Fundamental check: Start must be before End
        assert start_ms < end_ms, "Contract start time must be before end time"

        # Check specific expiry time logic (15:00 UTC+7 -> 08:00 UTC)
        # We can check if the timestamp matches 08:00 UTC using datetime
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=pytz.utc)
        assert (
            end_dt.hour == 8 and end_dt.minute == 0
        ), "Expiration should be at 08:00 UTC (15:00 ICT)"
