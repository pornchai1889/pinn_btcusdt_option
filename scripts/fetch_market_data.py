import requests
import csv
import os
from datetime import datetime, date, time, timedelta
import calendar
import pytz 
import time as t_module

# ==========================================
# CONFIGURATION (ตั้งค่าที่นี่)
# ==========================================
TARGET_OPTION_SYMBOL = 'BTC-251114-110000-C'
TIME_INTERVAL = '1h'  
# ==========================================

MILLISECONDS_IN_YEAR = 365 * 24 * 60 * 60 * 1000

def get_expiration_type(symbol: str) -> str:
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
    except Exception: return "Unknown"

def parse_symbol_date(symbol: str) -> date:
    date_str = symbol.split('-')[1]
    year = 2000 + int(date_str[0:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    return date(year, month, day)

def calculate_contract_period(symbol: str, exp_type: str) -> tuple[int, int]:
    ict_tz = pytz.timezone('Asia/Bangkok')
    utc_tz = pytz.utc
    exp_date = parse_symbol_date(symbol)
    start_date = None
    naive_end_dt = datetime.combine(exp_date, time(15, 0))
    aware_end_dt_ict = ict_tz.localize(naive_end_dt)
    
    if exp_type == "Daily": start_date = exp_date - timedelta(days=1)
    elif exp_type == "Weekly": start_date = exp_date - timedelta(days=7)
    elif exp_type == "Monthly":
        first = exp_date.replace(day=1)
        prev_last = first - timedelta(days=1)
        offset = (prev_last.weekday() - 4 + 7) % 7
        start_date = prev_last - timedelta(days=offset)
    elif exp_type == "Quarterly":
        m = exp_date.month
        start_m = 1 if m<=3 else 4 if m<=6 else 7 if m<=9 else 10
        first = exp_date.replace(month=start_m, day=1)
        prev_last = first - timedelta(days=1)
        offset = (prev_last.weekday() - 4 + 7) % 7
        start_date = prev_last - timedelta(days=offset)

    if start_date:
        naive_start = datetime.combine(start_date, time(15, 0))
        aware_start = ict_tz.localize(naive_start)
        return int(aware_start.astimezone(utc_tz).timestamp() * 1000), int(aware_end_dt_ict.astimezone(utc_tz).timestamp() * 1000)
    return None, None

def fetch_klines_forward_robust(url, symbol, interval, start_ms, end_ms):
    """
    ดึงข้อมูลแบบ Forward (อดีต -> ปัจจุบัน)
    - บังคับ Sort ข้อมูล
    - ใช้ Dictionary กันซ้ำ
    - บังคับขยับเวลาถ้าติด Loop
    """
    unique_data_map = {} 
    current_start = start_ms
    
    print(f"   Target: {datetime.fromtimestamp(start_ms/1000)} -> {datetime.fromtimestamp(end_ms/1000)}")
    
    while True:
        # เช็คเงื่อนไขจบ
        if current_start >= end_ms:
            print("   [Info] Reached target end time.")
            break

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000 
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) == 0:
                print("   [Info] No more data from API.")
                break

            # 1. [สำคัญมาก] บังคับเรียงข้อมูล (เก่า -> ใหม่) ทันที
            if isinstance(data[0], dict):
                data.sort(key=lambda x: int(x['openTime']))
            else:
                data.sort(key=lambda x: int(x[0]))
            
            # 2. เก็บข้อมูลลง Dict (Auto Deduplication)
            new_count = 0
            for k in data:
                if isinstance(k, dict): t = int(k['openTime'])
                else: t = int(k[0])
                
                if t <= end_ms:
                    if t not in unique_data_map:
                        unique_data_map[t] = k
                        new_count += 1
            
            # 3. เตรียมเวลาเริ่มรอบถัดไป
            # เอาเวลาปิดของแท่ง "สุดท้าย" ใน Batch นี้
            last_k = data[-1]
            if isinstance(last_k, dict): 
                last_close = int(last_k['closeTime'])
            else: 
                last_close = int(last_k[6])
            
            print(f"   ...fetched batch ending {datetime.fromtimestamp(last_close/1000)}. New items: {new_count} (Total: {len(unique_data_map)})")

            next_start = last_close + 1
            
            # 4. [กันตาย] ถ้าเวลาใหม่ ไม่ขยับไปข้างหน้า (Loop ตัน)
            if next_start <= current_start:
                print("   [Fix] Stuck at same time. Forcing jump forward...")
                # กระโดดข้ามไป 1 แท่ง (ตาม interval)
                if 'm' in interval: delta = int(interval.replace('m','')) * 60 * 1000
                elif 'h' in interval: delta = int(interval.replace('h','')) * 60 * 60 * 1000
                else: delta = 3600000
                current_start += delta
            else:
                current_start = next_start
            
            # ถ้าได้มาน้อยกว่า 1000 แสดงว่าข้อมูลหมดแล้ว
            if len(data) < 1000:
                print("   [Info] End of available history.")
                break
                
            t_module.sleep(0.1)
            
        except Exception as e:
            print(f"   [Error] Fetching failed: {e}")
            break
            
    # 5. ส่งคืนข้อมูลแบบเรียงลำดับ
    sorted_data = sorted(unique_data_map.values(), key=lambda x: int(x['openTime']) if isinstance(x, dict) else int(x[0]))
    return sorted_data

def main():
    try: strike_price_K = TARGET_OPTION_SYMBOL.split('-')[2]
    except: return

    exp_type = get_expiration_type(TARGET_OPTION_SYMBOL)
    start_ms, end_ms = calculate_contract_period(TARGET_OPTION_SYMBOL, exp_type)
    
    now_ms = int(datetime.now().timestamp() * 1000)
    fetch_end_ms = min(end_ms, now_ms)

    print(f"--- Fetching Market Data: {TARGET_OPTION_SYMBOL} ({TIME_INTERVAL}) ---")
    
    # 1. Fetch Option Data
    print("\n1. Downloading Option Klines...")
    option_url = "https://eapi.binance.com/eapi/v1/klines"
    options_data = fetch_klines_forward_robust(option_url, TARGET_OPTION_SYMBOL, TIME_INTERVAL, start_ms, fetch_end_ms)
    
    if not options_data:
        print("[X] No data found.")
        return

    if isinstance(options_data[0], dict):
        real_start = int(options_data[0]['openTime'])
        real_end = int(options_data[-1]['closeTime'])
    else:
        real_start = int(options_data[0][0])
        real_end = int(options_data[-1][6])
        
    print(f"[OK] Total Option Candles: {len(options_data)}")
    print(f"   Range: {datetime.fromtimestamp(real_start/1000)} -> {datetime.fromtimestamp(real_end/1000)}")

    # 2. Fetch BTC Data
    print("\n2. Downloading BTC Spot Klines...")
    btc_url = "https://api.binance.com/api/v3/klines"
    btc_data = fetch_klines_forward_robust(btc_url, 'BTCUSDT', TIME_INTERVAL, real_start, real_end)
    
    btc_price_map = {x[0]: x[4] for x in btc_data}
    print(f"[OK] Total BTC Candles: {len(btc_data)}")

    # 3. Save
    print("\n3. Saving CSV...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f"{TARGET_OPTION_SYMBOL}_{exp_type}_{TIME_INTERVAL}.csv"
    file_path = os.path.join(save_dir, file_name)

    headers = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'amount', 'trade_count', 
               'open_time_unix_ms', 'close_time_unix_ms', 
               'btc_close_price', 'strike_price_K',
               'time_to_maturity_t2m', 'current_time_t', 'contract_duration_T']

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        contract_T_val = (end_ms - start_ms) / MILLISECONDS_IN_YEAR

        count = 0
        for k in options_data:
            if isinstance(k, dict):
                open_ms, close_ms = int(k['openTime']), int(k['closeTime'])
                o, h, l, c, v = k['open'], k['high'], k['low'], k['close'], k['volume']
                amt, trd = k['amount'], k['tradeCount']
            else:
                open_ms, close_ms = k[0], k[6]
                o, h, l, c, v = k[1], k[2], k[3], k[4], k[5]
                amt, trd = k[7], k[8]

            btc_price = btc_price_map.get(open_ms, 'N/A')
            t2m = (end_ms - close_ms) / MILLISECONDS_IN_YEAR
            curr_t = (close_ms - start_ms) / MILLISECONDS_IN_YEAR
            
            writer.writerow([
                datetime.fromtimestamp(open_ms/1000).strftime('%Y-%m-%d %H:%M:%S'),
                o, h, l, c, v,
                datetime.fromtimestamp(close_ms/1000).strftime('%Y-%m-%d %H:%M:%S'),
                amt, trd, open_ms, close_ms,
                btc_price, strike_price_K,
                f"{t2m:.8f}", f"{curr_t:.8f}", f"{contract_T_val:.8f}"
            ])
            count += 1

    print(f"[Done] Saved {count} rows to: {file_path}")

if __name__ == "__main__":
    main()