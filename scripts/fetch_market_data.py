import requests
import json
import csv
import os
from datetime import datetime, date, time, timedelta
import calendar
import pytz 

# ==========================================
# CONFIGURATION (ตั้งค่าสัญญาที่ต้องการดึงที่นี่)
# ==========================================
TARGET_OPTION_SYMBOL = 'BTC-251128-90000-C'  # ใส่ชื่อสัญญาที่ต้องการ
TIME_INTERVAL = '1h'                        # Timeframe
# ==========================================

# Constant for time conversion in years
# ใช้ 365 วันถ้วน เพื่อให้ตรงกับตอนเทรนโมเดล
MILLISECONDS_IN_YEAR = 365 * 24 * 60 * 60 * 1000

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
    and converts them to UTC Unix Timestamps in milliseconds.
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

def main():
    option_symbol = TARGET_OPTION_SYMBOL
    time_interval = TIME_INTERVAL

    # Extract Strike Price
    try:
        strike_price_K = option_symbol.split('-')[2]
    except IndexError:
        print("Error: Invalid symbol format.")
        return

    expiration_type = get_expiration_type(option_symbol)
    start_time_ms, end_time_ms = calculate_contract_period(option_symbol, expiration_type)

    print(f"--- Analyzing contract: {option_symbol} ({time_interval}) ---")
    print(f"Option Type: {expiration_type}")
    print(f"Strike Price (K): {strike_price_K}")
    
    if start_time_ms:
        ict_tz = pytz.timezone('Asia/Bangkok')
        print(f"Contract Period (Thai Time):")
        print(f"  Start: {datetime.fromtimestamp(start_time_ms/1000, tz=ict_tz)}")
        print(f"  End:   {datetime.fromtimestamp(end_time_ms/1000, tz=ict_tz)}\n")

    try:
        if not start_time_ms:
            raise ValueError("Could not determine contract period.")

        print("--- Step 1: Fetching Option Data ---")
        option_klines_url = "https://eapi.binance.com/eapi/v1/klines"
        option_params = {'symbol': option_symbol, 'interval': time_interval, 'startTime': start_time_ms, 'endTime': end_time_ms, 'limit': 1000}
        response_options = requests.get(option_klines_url, params=option_params)
        response_options.raise_for_status()
        options_data = response_options.json()

        if isinstance(options_data, list):
            options_data.sort(key=lambda kline: int(kline['openTime']))

        if isinstance(options_data, list) and options_data:
            print(f"Found {len(options_data)} k-lines for the option.")

            print(f"--- Step 2: Fetching BTC Price Data ---\n")
            btc_klines_url = "https://api.binance.com/api/v3/klines"
            btc_params = {'symbol': 'BTCUSDT', 'interval': time_interval, 'startTime': start_time_ms, 'endTime': end_time_ms, 'limit': 1000}
            response_btc = requests.get(btc_klines_url, params=btc_params)
            response_btc.raise_for_status()
            btc_data = response_btc.json()
            
            btc_price_map = {kline[0]: kline[4] for kline in btc_data}
            print(f"Successfully mapped {len(btc_price_map)} BTC k-lines.\n")

            print("--- Step 3: Saving to CSV ---")
            
            # --- PATH CONFIGURATION (ปรับให้เข้ากับโครงสร้างใหม่) ---
            # หาตำแหน่งของไฟล์ script นี้
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            # ถอยออกไป 1 ชั้นเพื่อเจอ root project -> แล้วเข้า data -> raw
            project_root = os.path.dirname(current_script_dir)
            folder_path = os.path.join(project_root, 'data', 'raw')
            
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{option_symbol}_{expiration_type}_{time_interval}.csv"
            file_path = os.path.join(folder_path, file_name)
            # ------------------------------------------------------

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
                    
                    # ใช้ Close time ในการคำนวณเวลาที่เหลือ
                    time_to_maturity_t2m = (end_time_ms - close_time_ms) / MILLISECONDS_IN_YEAR
                    current_time_t = (close_time_ms - start_time_ms) / MILLISECONDS_IN_YEAR
                    
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
            
            print(f"\n Data saved successfully to:\n{file_path}")

        else:
            print("No data found for this option symbol.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()