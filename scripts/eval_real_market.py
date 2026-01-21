# scripts/eval_real_market.py
import os
import sys
import yaml
import json
import glob
import re
import logging
import requests
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from scipy.stats import norm
from typing import Tuple, List, Dict, Any, Optional

# --- Environment Setup ---
# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Import Project Modules ---
from src.models.pinn_net import UniversalPINN
from src.physics.call_option import CallOption
from src.physics.put_option import PutOption
from src.data.normalizer import MarketNormalizer

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Evaluator")


class MarketDataHandler:
    """
    Handles data fetching, timeframe parsing, and volatility calculations.
    Encapsulates all logic related to market data preprocessing.
    """

    def __init__(self, risk_free_rate: float, lookback_days: int):
        self.r = risk_free_rate
        self.lookback_days = lookback_days

    def get_timeframe_params(self, filename: str) -> Tuple[str, int, float]:
        """
        Parses the timeframe from the filename and calculates window size.
        """
        try:
            # Extract timeframe string (e.g., '2h' from '..._2h.csv')
            timeframe_str = filename.replace(".csv", "").split("_")[-1]

            tf_map = {
                "15m": 0.25,
                "30m": 0.5,
                "1h": 1.0,
                "2h": 2.0,
                "4h": 4.0,
                "1d": 24.0,
            }

            if timeframe_str not in tf_map:
                logger.warning(
                    f"Unknown timeframe '{timeframe_str}'. Defaulting to 1h."
                )
                hours_per_candle = 1.0
                timeframe_str = "1h"
            else:
                hours_per_candle = tf_map[timeframe_str]

            # Calculate window size: (24h / hours_per_candle) * days
            candles_per_day = 24 / hours_per_candle
            window_size = int(candles_per_day * self.lookback_days)

            if window_size < 10:
                window_size = 10

            candles_per_year = (365 * 24) / hours_per_candle
            annual_factor = np.sqrt(candles_per_year)

            return timeframe_str, window_size, annual_factor

        except Exception as e:
            logger.error(f"Error parsing timeframe: {e}")
            return "1h", 24 * self.lookback_days, np.sqrt(365 * 24)

    def fetch_btc_lookback(
        self, start_time_ms: int, timeframe_str: str, limit: int
    ) -> List[float]:
        """
        Fetches historical BTC prices from Binance with pagination.
        Used for volatility calculation.
        """
        url = "https://api.binance.com/api/v3/klines"
        all_prices = []
        current_end_time = start_time_ms - 1
        remaining = limit

        while remaining > 0:
            batch_size = min(remaining, 1000)
            params = {
                "symbol": "BTCUSDT",
                "interval": timeframe_str,
                "endTime": current_end_time,
                "limit": batch_size,
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code != 200:
                    break

                data = response.json()
                if not data:
                    break

                # Extract close prices
                prices = [float(x[4]) for x in data]
                all_prices = prices + all_prices

                # Update pagination cursor
                oldest_time_in_batch = int(data[0][0])
                current_end_time = oldest_time_in_batch - 1
                remaining -= len(data)

                if len(data) < batch_size:
                    break  # End of data

            except Exception as e:
                logger.warning(f"Failed to fetch historical BTC: {e}")
                break

        return all_prices[-limit:]

    def calculate_dynamic_volatility(
        self,
        main_prices: np.ndarray,
        lookback_prices: List[float],
        window: int,
        annual_factor: float,
    ) -> np.ndarray:
        """
        Calculates rolling volatility using combined historical and current data.
        """
        combined_prices = pd.Series(lookback_prices + list(main_prices))
        log_ret = np.log(combined_prices / combined_prices.shift(1))
        vol_combined = log_ret.rolling(window=window).std() * annual_factor

        # Slice to match the main data length
        vol_result = vol_combined.iloc[len(lookback_prices) :].values

        # Backfill NaNs at the beginning
        if np.isnan(vol_result).any():
            vol_result = pd.Series(vol_result).bfill().ffill().values

        return vol_result


class RealMarketEvaluator:
    """
    Main class for evaluating the PINN model against real market data.
    """

    def __init__(self, config_path: str):
        self.eval_config = self._load_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup paths
        self.run_folder = os.path.join(
            project_root, self.eval_config["paths"]["run_folder"]
        )

        # Load Training Config (Mother Config)
        self.train_config = self._load_train_config()

        # Initialize Components
        self._init_model()
        self._init_physics_engines()
        self.normalizer = MarketNormalizer(self.train_config)
        self.data_helper = MarketDataHandler(
            self.eval_config["financial"]["risk_free_rate"],
            self.eval_config["financial"]["lookback_days"],
        )

        # Setup Output Directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out = self.eval_config["paths"]["output_dirname"]
        self.output_dir = os.path.join(self.run_folder, f"{base_out}_{ts}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Fix Matplotlib font
        try:
            mpl.rcParams["axes.unicode_minus"] = False
        except:
            pass

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_train_config(self) -> Dict[str, Any]:
        """Traverses up to find the original config.json/yaml."""
        curr = self.run_folder
        for _ in range(5):
            p_json = os.path.join(curr, "config.json")
            p_yaml = os.path.join(curr, "config.yaml")
            if os.path.exists(p_json):
                with open(p_json, "r") as f:
                    return json.load(f)
            if os.path.exists(p_yaml):
                with open(p_yaml, "r") as f:
                    return yaml.safe_load(f)
            curr = os.path.dirname(curr)
        raise FileNotFoundError(
            f"Original training config not found starting from {self.run_folder}"
        )

    def _init_model(self) -> None:
        """Initializes model and loads weights."""
        m_conf = self.train_config["model"]
        self.model = UniversalPINN(self.train_config).to(self.device)

        fname = self.eval_config["paths"]["model_filename"]
        model_path = os.path.join(self.run_folder, fname)
        if not os.path.exists(model_path):
            # Fallback check in checkpoints folder
            model_path = os.path.join(self.run_folder, "model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading weights from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _init_physics_engines(self) -> None:
        """Initializes physics engines for analytical comparison."""
        self.call_physics = CallOption(self.train_config)
        self.put_physics = PutOption(self.train_config)

        # Detect what the model was trained on
        exp_name = self.train_config["experiment"]["name"].lower()
        self.is_trained_on_put = "put" in exp_name
        logger.info(f"Model Logic: {'PUT' if self.is_trained_on_put else 'CALL'}")

    def run(self) -> None:
        """
        Main execution: Scans specifically the 'data_raw_dir' folder for CSVs.
        If no CSVs are found, it alerts and exits.
        """
        # Resolve the raw data directory from config
        data_dir_conf = self.eval_config["paths"]["data_raw_dir"]
        data_dir = os.path.join(project_root, data_dir_conf)

        if not os.path.exists(data_dir):
            logger.error(f"Directory not found: {data_dir}")
            return

        # Pattern match for CSV files in this specific directory
        csv_pattern = os.path.join(data_dir, "*.csv")
        files = glob.glob(csv_pattern)

        # Check if files exist
        if not files:
            logger.error(f"No CSV files found in: {data_dir}")
            return

        logger.info(
            f"Found {len(files)} CSV files in {data_dir}. Starting processing..."
        )

        # Process each file found
        for f in files:
            self.process_file(f)

    def process_file(self, csv_path: str) -> None:
        filename = os.path.basename(csv_path)
        logger.info(f"Processing: {filename}")

        # 1. Parse Timeframe & Volatility Parameters
        tf_str, window, ann_factor = self.data_helper.get_timeframe_params(filename)

        # 2. Load Data
        df = pd.read_csv(csv_path)
        required = [
            "btc_close_price",
            "time_to_maturity_t2m",
            "strike_price_K",
            "close",
            "open_time_unix_ms",
        ]
        if not all(col in df.columns for col in required):
            logger.warning(f"Skipping {filename}: Missing columns.")
            return

        # 3. Detect Mismatch (Call File vs Put Model)
        # Check filename for explicit type indicators
        is_file_call = "-C_" in filename or "-C-" in filename
        is_file_put = "-P_" in filename or "-P-" in filename

        # Determine if we can compare against market price (Model matches Data)
        can_compare_market = (is_file_call and not self.is_trained_on_put) or (
            is_file_put and self.is_trained_on_put
        )

        # 4. Prepare Inputs
        S_data = df["btc_close_price"].values
        t_data = df["time_to_maturity_t2m"].values
        K_data = df["strike_price_K"].values
        V_market = df["close"].values

        # Calculate Volatility
        first_ts = df["open_time_unix_ms"].iloc[0]
        lookback = self.data_helper.fetch_btc_lookback(first_ts, tf_str, limit=window)
        sigma_data = self.data_helper.calculate_dynamic_volatility(
            S_data, lookback, window, ann_factor
        )
        r_data = np.full_like(S_data, self.data_helper.r)

        # 5. Normalize & Predict
        X_input = self.normalizer.normalize_batch(
            t_data.reshape(-1, 1),
            S_data.reshape(-1, 1),
            sigma_data.reshape(-1, 1),
            r_data.reshape(-1, 1),
            K_data.reshape(-1, 1),
        )
        X_tensor = torch.from_numpy(X_input).float().to(self.device)

        with torch.no_grad():
            v_ratio_pred = self.model(X_tensor).cpu().numpy().flatten()

        V_pred_pinn = v_ratio_pred * K_data

        # 6. Analytical Benchmark
        physics = self.put_physics if self.is_trained_on_put else self.call_physics
        V_analytical = physics.analytical_solution(
            t_data, S_data, K_data, r_data, sigma_data
        )

        # 7. Metrics
        rmse_anal = np.sqrt(np.mean((V_analytical - V_pred_pinn) ** 2))
        corr_anal = (
            np.corrcoef(V_analytical, V_pred_pinn)[0, 1]
            if np.std(V_pred_pinn) > 0
            else 0
        )

        # Calculate market metrics only if comparable (logic kept, but display handled in plot)
        if can_compare_market:
            rmse_mkt = np.sqrt(np.mean((V_market - V_pred_pinn) ** 2))
            corr_mkt = (
                np.corrcoef(V_market, V_pred_pinn)[0, 1]
                if np.std(V_pred_pinn) > 0
                else 0
            )
        else:
            rmse_mkt, corr_mkt = 0.0, 0.0

        # 8. Visualization
        self._generate_plot(
            t_data,
            S_data,
            V_market,
            V_analytical,
            V_pred_pinn,
            sigma_data,
            filename,
            rmse_mkt,
            corr_mkt,
            rmse_anal,
            corr_anal,
            tf_str,
            window,
            can_compare_market,
        )

    def _generate_plot(
        self,
        t_data: np.ndarray,
        S_data: np.ndarray,
        V_market: np.ndarray,
        V_analytical: np.ndarray,
        V_pred: np.ndarray,
        sigma_data: np.ndarray,
        filename: str,
        rmse_mkt: float,
        corr_mkt: float,
        rmse_anal: float,
        corr_anal: float,
        tf_str: str,
        window_size: int,
        can_compare_market: bool,
    ) -> None:
        """
        Generates the evaluation plot.
        If mismatch (can_compare_market=False):
          - Market Price line is NOT plotted.
          - Market metrics are removed from the title.
        """
        file_basename = os.path.splitext(filename)[0]

        # Create Canvas: 2 Subplots (Price 3 parts, Sigma 1 part)
        fig, (ax1, ax_sigma) = plt.subplots(
            2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # --- Graph 1: Prices ---

        # Only plot Market Price if model type matches data type
        if can_compare_market:
            ax1.plot(
                t_data,
                V_market,
                label="Market Price",
                color="purple",
                alpha=0.5,
                linewidth=1.5,
            )

        ax1.plot(
            t_data,
            V_analytical,
            label="Analytical (BS)",
            color="dodgerblue",
            linestyle="-.",
            alpha=0.8,
        )
        ax1.plot(
            t_data,
            V_pred,
            label="PINN Prediction",
            color="darkorange",
            linestyle="--",
            linewidth=2,
        )

        ax1.set_ylabel("Option Price: V (USDT)", fontsize=12)

        # Title Formatting
        if can_compare_market:
            mkt_text = f"Market: RMSE={rmse_mkt:.2f}, Corr={corr_mkt:.4f}  |  "
        else:
            mkt_text = ""  # Clean title for mismatch case

        title_str = (
            f"Evaluation: {file_basename}\n"
            f"{mkt_text}Analytical: RMSE={rmse_anal:.2f}, Corr={corr_anal:.4f}"
        )

        ax1.set_title(title_str, fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Twin Axis for BTC Price
        ax2 = ax1.twinx()
        ax2.plot(
            t_data, S_data, label="BTC Price", color="green", alpha=0.3, linestyle=":"
        )
        ax2.set_ylabel("BTC Price: S (USDT)", color="green", fontsize=12)

        # Legend Combination
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

        # --- Graph 2: Volatility (Indicator) ---
        vol_label = (
            rf"Historical $\sigma$: {self.data_helper.lookback_days}-Day "
            f"({tf_str} Candles, N={window_size})"
        )

        ax_sigma.plot(
            t_data, sigma_data, label=vol_label, color="dimgrey", linewidth=1.5
        )
        ax_sigma.fill_between(t_data, sigma_data, 0, color="grey", alpha=0.1)

        ax_sigma.set_ylabel(r"Volatility ($\sigma$)", fontsize=12)
        ax_sigma.set_xlabel("Time to Maturity (Years)", fontsize=12)

        ax_sigma.legend(loc="lower right", frameon=True, fontsize=10)
        ax_sigma.grid(True, alpha=0.3)
        ax_sigma.set_ylim(bottom=0)

        # Invert X Axis (Time Decay View)
        ax1.invert_xaxis()

        plt.tight_layout()

        # Save Logic
        timestamp_suffix = datetime.now().strftime("%H%M%S")
        save_filename = f"result_{file_basename}_sigma{self.data_helper.lookback_days}day_{timestamp_suffix}.jpg"

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        logger.info(f"Graph saved to: {save_path}")
        plt.close(fig)


def main():
    # Setup Config Path
    config_path = os.path.join(project_root, "configs", "eval_real_market.yaml")

    if not os.path.exists(config_path):
        logger.error(f"Please create {config_path} first.")
        return

    # Run Evaluation
    evaluator = RealMarketEvaluator(config_path)
    evaluator.run()


if __name__ == "__main__":
    main()
