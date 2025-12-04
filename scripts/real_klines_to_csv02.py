import requests
import json
import csv
import os
from datetime import datetime, date, time, timedelta
import calendar
import pytz 

# Constant for time conversion in years
MILLISECONDS_IN_YEAR = 365.25 * 24 * 60 * 60 * 1000

# --- Functions to analyze option contract ---
def get_expiration_type(symbol: str) -> str:
    """
    Analyzes the option symbol to determine its Expiration Type.
    """
    try:
        exp_date = parse_symbol_date(symbol)
        is_friday = (exp_date.weekday() == 4)
        last_day_of_month = calendar.monthrange(exp_date.year, exp_date.month)[1]
        last_date_of_month = date(exp_date.year, exp_date.month, last_day_of_month)
        offset = (last_date_of_month.weekday() - 4 + 7) % 7
        last_friday = last_date_of_month.day - offset
        is_last_friday_of_month = (exp_date.day == last_friday)
        is_quarterly_month = (exp_date.month in [3, 6, 9, 12])
        if is_last_friday_of_month and is_quarterly_month: return "Quarterly"
        elif is_last_friday_of_month: return "Monthly"
        elif is_friday: return "Weekly"
        else: return "Daily"
    except Exception:
        return "Unknown"

def parse_symbol_date(symbol: str) -> date:
    """
    Parses the date object from the option symbol string.
    """
    date_str = symbol.split('-')[1]
    year = 2000 + int(date_str[0:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    return date(year, month, day)

def calculate_contract_period(symbol: str, exp_type: str) -> tuple[int, int]:
    """
    Calculates the precise start and end of the contract (15:00 to 15:00 ICT)
    and converts them to UTC Unix Timestamps in milliseconds for the API.
    """
    ict_tz = pytz.timezone('Asia/Bangkok')
    utc_tz = pytz.utc
    
    exp_date = parse_symbol_date(symbol)
    start_date = None
    
    naive_end_dt = datetime.combine(exp_date, time(15, 0))
    aware_end_dt_ict = ict_tz.localize(naive_end_dt)
    
    if exp_type == "Daily":
        start_date = exp_date - timedelta(days=1)
    elif exp_type == "Weekly":
        start_date = exp_date - timedelta(days=7)
    elif exp_type == "Monthly":
        first_day_of_month = exp_date.replace(day=1)
        prev_month_last_day = first_day_of_month - timedelta(days=1)
        offset = (prev_month_last_day.weekday() - 4 + 7) % 7
        start_date = prev_month_last_day - timedelta(days=offset)
    elif exp_type == "Quarterly":
        if exp_date.month <= 3: start_month = 1
        elif exp_date.month <= 6: start_month = 4
        elif exp_date.month <= 9: start_month = 7
        else: start_month = 10
        first_day_of_quarter = exp_date.replace(month=start_month, day=1)
        prev_quarter_last_day = first_day_of_quarter - timedelta(days=1)
        offset = (prev_quarter_last_day.weekday() - 4 + 7) % 7
        start_date = prev_quarter_last_day - timedelta(days=offset)

    if start_date:
        naive_start_dt = datetime.combine(start_date, time(15, 0))
        aware_start_dt_ict = ict_tz.localize(naive_start_dt)
        
        start_time_ms = int(aware_start_dt_ict.astimezone(utc_tz).timestamp() * 1000)
        end_time_ms = int(aware_end_dt_ict.astimezone(utc_tz).timestamp() * 1000)
        return start_time_ms, end_time_ms
    return None, None

# --- Configuration ---
option_symbol = 'BTC-251017-110000-C'
time_interval = '1h'
# --------------------

# --- NEW: Extract Strike Price from the symbol ---
strike_price_K = option_symbol.split('-')[2]
# ------------------------------------------------

expiration_type = get_expiration_type(option_symbol)
start_time_ms, end_time_ms = calculate_contract_period(option_symbol, expiration_type)

print(f"--- Analyzing contract: {option_symbol} ({time_interval}) ---")
print(f"Option Type: {expiration_type}")
print(f"Strike Price (K): {strike_price_K}") # Optional: show the extracted strike price
if start_time_ms:
    ict_tz = pytz.timezone('Asia/Bangkok')
    print(f"Calculated Contract Period (Thai Time):")
    print(f"  Start: {datetime.fromtimestamp(start_time_ms/1000, tz=ict_tz).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End (Inclusive): {datetime.fromtimestamp(end_time_ms/1000, tz=ict_tz).strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    if not start_time_ms:
        raise ValueError("Could not determine contract period.")

    print("--- Step 1: Fetching data for Option within the precise period ---")
    
    option_klines_url = "https://eapi.binance.com/eapi/v1/klines"
    option_params = {'symbol': option_symbol, 'interval': time_interval, 'startTime': start_time_ms, 'endTime': end_time_ms, 'limit': 1000}
    response_options = requests.get(option_klines_url, params=option_params)
    response_options.raise_for_status()
    options_data = response_options.json()

    if isinstance(options_data, list):
        options_data.sort(key=lambda kline: int(kline['openTime']))

    if isinstance(options_data, list) and options_data:
        print(f"Found and sorted {len(options_data)} k-lines for the option.")

        print(f"--- Step 2: Fetching historical BTC price data for the same period ---\n")

        btc_klines_url = "https://api.binance.com/api/v3/klines"
        btc_params = {'symbol': 'BTCUSDT', 'interval': time_interval, 'startTime': start_time_ms, 'endTime': end_time_ms, 'limit': 1000}
        response_btc = requests.get(btc_klines_url, params=btc_params)
        response_btc.raise_for_status()
        btc_data = response_btc.json()
        
        btc_price_map = {kline[0]: kline[4] for kline in btc_data}
        print(f"Successfully mapped {len(btc_price_map)} BTC k-lines.\n")

        print("--- Step 3: Merging data and saving to CSV file ---")
        
        folder_path = 'btcusdt_options_call_V2/klines'
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"{option_symbol}_{expiration_type}_{time_interval}.csv"
        file_path = os.path.join(folder_path, file_name)

        # --- NEW: Added 'strike_price_K' to headers ---
        headers = [ 
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'amount', 'trade_count', 
            'open_time_unix_ms', 'close_time_unix_ms', 
            'btc_close_price', 'strike_price_K',
            'time_to_maturity_t2m', 'current_time_t', 'contract_duration_T'
        ]

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            contract_duration_T_ms = end_time_ms - start_time_ms
            contract_duration_T = contract_duration_T_ms / MILLISECONDS_IN_YEAR

            for kline in options_data:
                open_time_ms = int(kline['openTime'])
                close_time_ms = int(kline['closeTime'])
                btc_price_str = btc_price_map.get(open_time_ms, 'N/A')
                if btc_price_str != 'N/A': formatted_btc_price = f"{float(btc_price_str):.2f}"
                else: formatted_btc_price = 'N/A'
                
                time_to_maturity_t2m = (end_time_ms - open_time_ms) / MILLISECONDS_IN_YEAR
                current_time_t = (open_time_ms - start_time_ms) / MILLISECONDS_IN_YEAR
                
                # --- NEW: Added strike_price_K to the row ---
                writer.writerow([
                    datetime.fromtimestamp(open_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    kline['open'], kline['high'], kline['low'], kline['close'], kline['volume'],
                    datetime.fromtimestamp(close_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    kline['amount'], kline['tradeCount'], 
                    kline['openTime'], kline['closeTime'], 
                    formatted_btc_price,
                    strike_price_K,
                    f"{time_to_maturity_t2m:.8f}",
                    f"{current_time_t:.8f}",
                    f"{contract_duration_T:.8f}"
                ])
        
        print(f"\n Data saved successfully! File located at:\n{file_path}")

    else:
        print("K-line data not found for the option within the specified period.")

except Exception as e:
    print(f"An error occurred: {e}")