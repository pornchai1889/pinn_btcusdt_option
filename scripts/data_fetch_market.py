# scripts/data_fetch_market.py
import os
import sys
import csv
import yaml
import logging
from datetime import datetime
from typing import Any, Dict

# --- Environment Setup ---
# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import Custom Modules
from src.utils.date_utils import DateUtils
from src.data.binance_api import BinanceDataFetcher

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MarketFetcher")

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main() -> None:
    # 1. Load Configuration
    config_path = os.path.join(project_root, 'configs', 'data_fetch_market.yaml')
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Please create {config_path} first.")
        return

    fetch_conf = config['fetch']
    api_conf = config['api']
    
    symbol = fetch_conf['symbol']
    interval = fetch_conf['interval']
    
    # 2. Determine Contract Details
    # Parsing Symbol Format: BTC-YYMMDD-Strike-Type (e.g., BTC-251226-100000-C)
    try:
        parts = symbol.split('-')
        strike_price_K = parts[2]
        option_type_code = parts[-1] # Extract 'C' or 'P'
    except IndexError:
        logger.error(f"Invalid Symbol Format: {symbol}")
        return

    exp_type = DateUtils.get_expiration_type(symbol)
    start_ms, end_ms = DateUtils.calculate_contract_period(symbol, exp_type)
    
    if start_ms is None:
        logger.error("Could not calculate contract period.")
        return

    now_ms = int(datetime.now().timestamp() * 1000)
    fetch_end_ms = min(end_ms, now_ms)

    logger.info(f"--- Fetching Market Data: {symbol} ({interval}) ---")
    logger.info(f"Contract Type: {exp_type}")
    logger.info(f"Option Type: {'Call' if option_type_code == 'C' else 'Put'}")
    logger.info(f"Period: {datetime.fromtimestamp(start_ms/1000)} -> {datetime.fromtimestamp(fetch_end_ms/1000)}")

    # 3. Initialize Fetcher
    fetcher = BinanceDataFetcher(limit=api_conf.get('limit', 1000), timeout=api_conf.get('timeout', 10))

    # 4. Download Option Klines
    logger.info("Downloading Option Klines...")
    options_data = fetcher.fetch_klines(
        api_conf['option_url'], symbol, interval, start_ms, fetch_end_ms
    )
    
    if not options_data:
        logger.warning("[X] No Option data found.")
        return

    # Determine real time range from actual data
    if isinstance(options_data[0], dict):
        real_start = int(options_data[0]['openTime'])
        real_end = int(options_data[-1]['closeTime'])
    else:
        real_start = int(options_data[0][0])
        real_end = int(options_data[-1][6])
        
    logger.info(f"[OK] Total Option Candles: {len(options_data)}")

    # 5. Download BTC Spot Klines (Base Asset)
    logger.info("Downloading BTC Spot Klines...")
    btc_data = fetcher.fetch_klines(
        api_conf['spot_url'], fetch_conf['base_asset'], interval, real_start, real_end
    )
    
    # Map BTC Price by OpenTime for O(1) Lookup
    btc_price_map = {x[0]: x[4] for x in btc_data}
    logger.info(f"[OK] Total BTC Candles: {len(btc_data)}")

    # 6. Save Data to CSV (Organized by Option Type)
    logger.info("Saving to CSV...")
    
    # Determine sub-directory based on Option Type
    if option_type_code == 'C':
        sub_dir = 'call'
    elif option_type_code == 'P':
        sub_dir = 'put'
    else:
        sub_dir = 'others' # Fallback for unknown types
        logger.warning(f"Unknown option type '{option_type_code}'. Saving to 'others'.")

    # Construct Path: data/raw/{call|put}/filename.csv
    base_output_dir = fetch_conf['output_dir']
    save_dir = os.path.join(project_root, base_output_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f"{symbol}_{exp_type}_{interval}.csv"
    file_path = os.path.join(save_dir, file_name)

    headers = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'amount', 'trade_count', 
        'open_time_unix_ms', 'close_time_unix_ms', 
        'btc_close_price', 'strike_price_K',
        'time_to_maturity_t2m', 'current_time_t', 'contract_duration_T'
    ]

    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            contract_T_val = (end_ms - start_ms) / DateUtils.MILLISECONDS_IN_YEAR
            count = 0
            
            for k in options_data:
                # Handle Dict (Option API) vs List (Spot API) format differences
                if isinstance(k, dict):
                    open_ms, close_ms = int(k['openTime']), int(k['closeTime'])
                    o, h, l, c, v = k['open'], k['high'], k['low'], k['close'], k['volume']
                    amt, trd = k['amount'], k['tradeCount']
                else:
                    open_ms, close_ms = k[0], k[6]
                    o, h, l, c, v = k[1], k[2], k[3], k[4], k[5]
                    amt, trd = k[7], k[8]

                btc_price = btc_price_map.get(open_ms, 'N/A')
                
                # Calculate Time Metrics
                t2m = (end_ms - close_ms) / DateUtils.MILLISECONDS_IN_YEAR
                curr_t = (close_ms - start_ms) / DateUtils.MILLISECONDS_IN_YEAR
                
                writer.writerow([
                    datetime.fromtimestamp(open_ms/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    o, h, l, c, v,
                    datetime.fromtimestamp(close_ms/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    amt, trd, open_ms, close_ms,
                    btc_price, strike_price_K,
                    f"{t2m:.8f}", f"{curr_t:.8f}", f"{contract_T_val:.8f}"
                ])
                count += 1
        
        logger.info(f"[Done] Saved {count} rows to: {file_path}")
        
    except IOError as e:
        logger.error(f"Failed to write CSV: {e}")

if __name__ == "__main__":
    main()