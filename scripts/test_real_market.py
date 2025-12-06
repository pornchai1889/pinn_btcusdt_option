import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import requests
from scipy.stats import norm

# ==========================================
# CONFIGURATION
# ==========================================
# 1. โฟลเดอร์ผลลัพธ์การเทรน
RUN_FOLDER = "runs/train_2025-12-06_17-17-43_Universal5Inputs"  

# 2. ชื่อโมเดลที่ต้องการโหลดมาใช้งาน
MODELL = "checkpoint_epoch_230000.pth"

# 3. ไฟล์ข้อมูลตลาดจริง (CSV)
CSV_FILE = "data/raw/BTC-251114-110000-C_Weekly_1h.csv" 

# 4. ค่าสมมติ (สำหรับ r เพราะไม่มีข้อมูลจริงใน csv นี้)
RISK_FREE_RATE = 0.05



# [เพิ่ม] กำหนดจำนวนวันย้อนหลังที่ต้องการคำนวณ Sigma
# แนะนำ: 7 วัน (สำหรับ Weekly/Universal) หรือ 30 วัน (ถ้าเน้น Monthly)
LOOKBACK_DAYS = 7  
# ==========================================


# --- Model Definition ---
class UniversalPINN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        activation = nn.Tanh()
        layers = [nn.Linear(n_input, n_hidden), activation]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation)
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --- Helper Functions ---
def normalize_val(val, v_min, v_max):
    return (val - v_min) / (v_max - v_min)

def analytical_solution_np(S, K, t, r, sigma):
    t = np.maximum(t, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

# --- Timeframe & API Helpers ---
def get_timeframe_params(filename, days=1):
    """
    คำนวณ Window Size ตามจำนวนวันที่ต้องการ (days)
    """
    try:
        timeframe_str = filename.replace('.csv', '').split('_')[-1]
        tf_map = {
            '15m': 0.25, '30m': 0.5,
            '1h': 1.0,   '2h': 2.0, '4h': 4.0,
            '1d': 24.0
        }
        
        if timeframe_str not in tf_map:
            print(f" Warning: Unknown timeframe '{timeframe_str}'. Defaulting to 1h.")
            hours_per_candle = 1.0
        else:
            hours_per_candle = tf_map[timeframe_str]

        # สูตร: (24 ชั่วโมง / ชั่วโมงต่อแท่ง) * จำนวนวัน
        candles_per_day = 24 / hours_per_candle
        window_size = int(candles_per_day * days)
        
        if window_size < 10: window_size = 10 

        candles_per_year = (365 * 24) / hours_per_candle
        annual_factor = np.sqrt(candles_per_year)

        return timeframe_str, window_size, annual_factor
        
    except Exception as e:
        print(f"Error parsing timeframe: {e}")
        return '1h', 24 * days, np.sqrt(365*24)

def fetch_btc_lookback(start_time_ms, timeframe_str, limit=24):
    """
    ดึงราคา BTC ย้อนหลังแบบ Pagination (รองรับจำนวน > 1000 แท่ง)
    """
    url = "https://api.binance.com/api/v3/klines"
    all_prices = []
    
    # เริ่มนับถอยหลังจากเวลาที่กำหนด
    current_end_time = start_time_ms - 1
    remaining = limit
    
    print(f"Fetching {limit} historical candles...")
    
    while remaining > 0:
        # Binance รับ limit สูงสุด 1000
        batch_size = min(remaining, 1000)
        
        params = {
            'symbol': 'BTCUSDT',
            'interval': timeframe_str,
            'endTime': current_end_time,
            'limit': batch_size
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            # ดึงราคาปิด
            prices = [float(x[4]) for x in data]
            
            # เอาของใหม่ (ซึ่งเป็นอดีตที่เก่ากว่า) ไปแปะไว้ข้างหน้า
            # เพราะ Binance ส่งมาเรียงจาก เก่า -> ใหม่ ในแต่ละ Batch
            # Batch แรกที่ดึงคือ "ล่าสุด" -> Batch ต่อไปคือ "เก่ากว่า"
            all_prices = prices + all_prices 
            
            # เตรียมเวลาสำหรับรอบถัดไป (ขยับถอยหลังไปอีก)
            # data[0][0] คือ OpenTime ของแท่งที่เก่าที่สุดใน Batch นี้
            oldest_time_in_batch = int(data[0][0])
            current_end_time = oldest_time_in_batch - 1
            
            remaining -= len(data)
            
            # ถ้าดึงมาได้น้อยกว่าที่ขอ แปลว่าหมดข้อมูลแล้ว
            if len(data) < batch_size:
                break
                
        except Exception as e:
            print(f"Failed to fetch historical BTC: {e}")
            break
            
    print(f"Fetched total {len(all_prices)} historical BTC candles.")
    
    # ถ้าได้มาเกิน (เผื่อไว้) ให้ตัดเอาเฉพาะจำนวนที่ต้องการ (เอาตัวท้ายๆ ซึ่งคือล่าสุด)
    return all_prices[-limit:]

def calculate_dynamic_volatility(main_prices, lookback_prices, window, annual_factor):
    """รวมข้อมูลเก่า+ใหม่ แล้วคำนวณ Rolling Sigma"""
    combined_prices = pd.Series(lookback_prices + list(main_prices))
    log_ret = np.log(combined_prices / combined_prices.shift(1))
    vol_combined = log_ret.rolling(window=window).std() * annual_factor
    
    # ตัดส่วน Lookback ออก คืนค่าเฉพาะช่วงเวลาหลัก
    vol_result = vol_combined.iloc[len(lookback_prices):].values
    
    if np.isnan(vol_result).any():
        vol_result = pd.Series(vol_result).bfill().values
        
    return vol_result

def main():
    try: mpl.rcParams['axes.unicode_minus'] = False
    except: pass

    print(f"--- Testing Real Market Data (Dynamic Mode) ---")
    
    # 1. Parse Timeframe
    filename = os.path.basename(CSV_FILE)
    tf_str, window_size, ann_factor = get_timeframe_params(filename, days=LOOKBACK_DAYS)
    print(f"Detected Timeframe: {tf_str}")
    print(f"Sigma Lookback: {LOOKBACK_DAYS} days ({window_size} candles)")

    # 2. Load Config & Model
    config_path = os.path.join(RUN_FOLDER, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        TRAIN_CONFIG = json.load(f)
    c_m = TRAIN_CONFIG["market"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalPINN(
        TRAIN_CONFIG["model"]["n_input"], TRAIN_CONFIG["model"]["n_output"],
        TRAIN_CONFIG["model"]["n_hidden"], TRAIN_CONFIG["model"]["n_layers"]
    ).to(device)
    
    model_path = os.path.join(RUN_FOLDER, MODELL)
    if not os.path.exists(model_path): model_path = os.path.join(RUN_FOLDER, "final_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Load Main Data
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV not found at {CSV_FILE}")
        return
    df = pd.read_csv(CSV_FILE)
    S_data = df['btc_close_price'].values
    
    # 4. Calculate Sigma (Fetch Lookback + Calc)
    first_timestamp = df['open_time_unix_ms'].iloc[0]
    # ส่ง window_size ที่คำนวณได้เข้าไปเป็น limit (ไม่ใช้ 24 ตายตัวแล้ว)
    lookback_prices = fetch_btc_lookback(first_timestamp, tf_str, limit=window_size)
    
    sigma_data = calculate_dynamic_volatility(S_data, lookback_prices, window_size, ann_factor)
    
    # 5. Prepare Inputs
    if 'time_to_maturity_t2m' in df.columns:
        t_data = df['time_to_maturity_t2m'].values
    else: raise ValueError("Missing 'time_to_maturity_t2m' in CSV")
    
    r_data = np.full_like(S_data, RISK_FREE_RATE)
    
    if 'strike_price_K' in df.columns:
        K_data = df['strike_price_K'].values
    else:
        # Fallback: Try parsing from filename
        try:
            k_val = float(filename.split('-')[2])
            K_data = np.full_like(S_data, k_val)
        except: raise ValueError("Cannot find Strike Price")

    # 6. Normalize & Predict
    t_norm = normalize_val(t_data, 0, c_m["T_MAX"])
    S_norm = normalize_val(S_data, c_m["S_range"][0], c_m["S_range"][1])
    sig_norm = normalize_val(sigma_data, c_m["sigma_range"][0], c_m["sigma_range"][1])
    r_norm = normalize_val(r_data, c_m["r_range"][0], c_m["r_range"][1])
    K_norm = normalize_val(K_data, c_m["K_range"][0], c_m["K_range"][1])
    
    X_input = np.column_stack((t_norm, S_norm, sig_norm, r_norm, K_norm))
    X_tensor = torch.from_numpy(X_input).float().to(device)

    print("Running prediction...")
    with torch.no_grad():
        v_ratio_pred = model(X_tensor).cpu().numpy().flatten()
    
    V_pred_pinn = v_ratio_pred * K_data
    V_analytical = analytical_solution_np(S_data, K_data, t_data, r_data, sigma_data)
    V_market = df['close'].values

    # 7. Calculate Metrics (Both Market & Analytical)
    # Vs Market
    rmse_mkt = np.sqrt(np.mean((V_market - V_pred_pinn)**2))
    corr_mkt = np.corrcoef(V_market, V_pred_pinn)[0, 1]
    
    # Vs Analytical
    rmse_anal = np.sqrt(np.mean((V_analytical - V_pred_pinn)**2))
    corr_anal = np.corrcoef(V_analytical, V_pred_pinn)[0, 1]
    
    print("\n=== Evaluation Results ===")
    print(f"Vs Market   -> RMSE: {rmse_mkt:.4f} | R: {corr_mkt:.4f}")
    print(f"Vs Theory   -> RMSE: {rmse_anal:.4f} | R: {corr_anal:.4f}")
    print("==========================\n")

    # 8. Visualization (2 Subplots: Price & Volatility)
    file_basename = os.path.splitext(filename)[0]
    
    # สร้าง Canvas แบ่ง 2 ส่วน (บน 3 ส่วน, ล่าง 1 ส่วน)
    fig, (ax1, ax_sigma) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Graph 1: Prices ---
    ax1.plot(t_data, V_market, label='Market Price', color='purple', alpha=0.5, linewidth=1.5)
    ax1.plot(t_data, V_analytical, label='Analytical (BS)', color='dodgerblue', linestyle='-.', alpha=0.8)
    ax1.plot(t_data, V_pred_pinn, label='PINN Prediction', color='darkorange', linestyle='--', linewidth=2)
    
    ax1.set_ylabel('Option Price: V (USDT)', fontsize=12)
    ax1.set_title(f'Universal Model Evaluation: {file_basename}\n'
                  f'Vs Market: RMSE={rmse_mkt:.2f}, R={corr_mkt:.4f}  |  '
                  f'Vs Theory: RMSE={rmse_anal:.2f}, R={corr_anal:.4f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # แกนขวาสำหรับราคา BTC
    ax2 = ax1.twinx()
    ax2.plot(t_data, S_data, label='BTC Price', color='green', alpha=0.15, linestyle=':')
    ax2.set_ylabel('BTC Price: S (USDT)', color='green', fontsize=12)
    
    # รวม Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

    # --- Graph 2: Volatility (Indicator) ---
# --- Graph 2: Volatility (Indicator) ---
    
    # สร้าง Label แบบมืออาชีพ (บอกครบ 3 ค่า: วัน, Timeframe, จำนวนแท่ง)
    # LOOKBACK_DAYS มาจาก Global Config ด้านบน
    # tf_str และ window_size มาจากฟังก์ชัน get_timeframe_params ที่เราเรียกไปแล้ว
    vol_label = (f"Historical Volatility: {LOOKBACK_DAYS}-Day "
                 f"({tf_str} Candles, N={window_size})")

    ax_sigma.plot(t_data, sigma_data, label=vol_label, color='dimgrey', linewidth=1.5)
    ax_sigma.fill_between(t_data, sigma_data, 0, color='grey', alpha=0.1) 
    
    ax_sigma.set_ylabel('Volatility (sigma)', fontsize=12)
    ax_sigma.set_xlabel('Time to Maturity (Years)', fontsize=12)
    
    # ปรับ Legend ให้สวยงาม
    ax_sigma.legend(loc='upper right', frameon=True, fontsize=10)
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.set_ylim(bottom=0)

    # กลับแกนเวลา (จากมากไป 0)
    ax1.invert_xaxis() 
    
    plt.tight_layout()
    
    # Save Plot
    save_path = os.path.join(RUN_FOLDER, f"result_{file_basename}_sigma{LOOKBACK_DAYS}day_r{RISK_FREE_RATE}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()