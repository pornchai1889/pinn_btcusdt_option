import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import requests # เพิ่ม requests เพื่อดึง API
from scipy.stats import norm

# ==========================================
# CONFIGURATION
# ==========================================
RUN_FOLDER = "runs/train_2025-12-04_12-33-35_Universal5Inputs"  # <--- ชื่อโฟลเดอร์ที่เทรนแล้ว
CSV_FILE = "data/raw/BTC-251226-95000-C_Quarterly_1h.csv" # <--- ชื่อไฟล์ CSV
RISK_FREE_RATE = 0.05
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

# --- NEW: Timeframe & API Helpers ---
def get_timeframe_params(filename):
    """
    แกะ Timeframe จากชื่อไฟล์ และคืนค่าตัวคูณสำหรับ Annualization
    ตัวอย่างไฟล์: BTC-251226-95000-C_Quarterly_1h.csv
    """
    try:
        # แยกส่วนชื่อไฟล์ด้วย '_' และเอาส่วนสุดท้าย (ตัด .csv ออก)
        timeframe_str = filename.replace('.csv', '').split('_')[-1]
        
        # Map timeframe เป็นจำนวนชั่วโมงต่อแท่ง
        tf_map = {
            '15m': 0.25, '30m': 0.5,
            '1h': 1.0,   '2h': 2.0, '4h': 4.0,
            '1d': 24.0
        }
        
        if timeframe_str not in tf_map:
            print(f"⚠️ Warning: Unknown timeframe '{timeframe_str}'. Defaulting to 1h.")
            hours_per_candle = 1.0
        else:
            hours_per_candle = tf_map[timeframe_str]

        # คำนวณ Window ที่เหมาะสม (เช่น ดูย้อนหลัง 24 ชั่วโมง)
        # ถ้า 1h -> window 24, ถ้า 15m -> window 96
        window_size = int(24 / hours_per_candle)
        if window_size < 10: window_size = 10 # กันเหนียวไม่ให้น้อยไป

        # คำนวณตัวคูณ Annualization (sqrt(จำนวนแท่งใน 1 ปี))
        # 1 ปี = 365 * 24 ชั่วโมง
        candles_per_year = (365 * 24) / hours_per_candle
        annual_factor = np.sqrt(candles_per_year)

        return timeframe_str, window_size, annual_factor
        
    except Exception as e:
        print(f"Error parsing timeframe: {e}")
        return '1h', 24, np.sqrt(365*24)

def fetch_btc_lookback(start_time_ms, timeframe_str, limit=24):
    """
    ดึงราคา BTC ย้อนหลัง (Lookback) จาก Binance API เพื่อแก้ปัญหา Cold Start
    """
    url = "https://api.binance.com/api/v3/klines"
    
    # เราต้องการข้อมูล "ก่อน" start_time_ms
    # Binance API 'endTime' คือจุดสิ้นสุด (Inclusive)
    # เราจึงตั้ง endTime = start_time_ms - 1 เพื่อเอาข้อมูลก่อนหน้านั้น
    params = {
        'symbol': 'BTCUSDT',
        'interval': timeframe_str,
        'endTime': start_time_ms - 1,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # data[4] คือ Close Price
        prices = [float(x[4]) for x in data]
        print(f"Fetched {len(prices)} historical BTC candles for volatility calc.")
        return prices
    except Exception as e:
        print(f"Failed to fetch historical BTC: {e}")
        return []

def calculate_dynamic_volatility(main_prices, lookback_prices, window, annual_factor):
    """
    คำนวณ Volatility โดยเอาข้อมูลย้อนหลัง (Lookback) มาต่อหัวข้อมูลหลัก
    ทำให้ข้อมูลหลัก (main_prices) มีค่า Sigma ตั้งแต่แท่งแรก
    """
    # รวมข้อมูล: [Lookback + Main]
    combined_prices = pd.Series(lookback_prices + list(main_prices))
    
    # คำนวณ Log Returns
    log_ret = np.log(combined_prices / combined_prices.shift(1))
    
    # คำนวณ Rolling Std * Annual Factor
    vol_combined = log_ret.rolling(window=window).std() * annual_factor
    
    # ตัดส่วน Lookback ออก ให้เหลือเท่ากับความยาวข้อมูลหลัก
    # เราเติมมา 'len(lookback)' ตัว ดังนั้นข้อมูลจริงเริ่มที่ index = len(lookback)
    vol_result = vol_combined.iloc[len(lookback_prices):].values
    
    # ถ้ายังมี NaN (กรณีดึง API ไม่ได้) ให้ใช้ Backfill ช่วย
    if np.isnan(vol_result).any():
        vol_result = pd.Series(vol_result).bfill().values
        
    return vol_result

def main():
    try:
        mpl.rcParams['axes.unicode_minus'] = False
    except: pass

    print(f"--- Testing Real Market Data (Dynamic Mode) ---")
    
    # 1. Parse Timeframe & Parameters from Filename
    filename = os.path.basename(CSV_FILE)
    tf_str, window_size, ann_factor = get_timeframe_params(filename)
    print(f"Detected Timeframe: {tf_str}")
    print(f"Volatility Settings: Window={window_size}, AnnualFactor={ann_factor:.2f}")

    # 2. Load Config & Model
    config_path = os.path.join(RUN_FOLDER, "config.json")
    with open(config_path, 'r') as f:
        TRAIN_CONFIG = json.load(f)
    c_m = TRAIN_CONFIG["market"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalPINN(
        TRAIN_CONFIG["model"]["n_input"], TRAIN_CONFIG["model"]["n_output"],
        TRAIN_CONFIG["model"]["n_hidden"], TRAIN_CONFIG["model"]["n_layers"]
    ).to(device)
    
    model_path = os.path.join(RUN_FOLDER, "model.pth")
    if not os.path.exists(model_path): model_path = os.path.join(RUN_FOLDER, "final_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Load Main Data (CSV)
    df = pd.read_csv(CSV_FILE)
    S_data = df['btc_close_price'].values
    
    # 4. Fetch Historical Data for Volatility
    first_timestamp = df['open_time_unix_ms'].iloc[0]
    lookback_prices = fetch_btc_lookback(first_timestamp, tf_str, limit=window_size)
    
    # 5. Calculate Sigma (The Right Way)
    sigma_data = calculate_dynamic_volatility(S_data, lookback_prices, window_size, ann_factor)
    
    # 6. Prepare Other Inputs
    # t (Tau)
    if 'time_to_maturity_t2m' in df.columns:
        t_data = df['time_to_maturity_t2m'].values
    else: raise ValueError("Missing 'time_to_maturity_t2m' in CSV")
    
    # r (Risk-free)
    r_data = np.full_like(S_data, RISK_FREE_RATE)
    
    # K (Strike)
    if 'strike_price_K' in df.columns:
        K_data = df['strike_price_K'].values
    else:
        try:
            k_val = float(filename.split('-')[2])
            K_data = np.full_like(S_data, k_val)
        except: raise ValueError("Cannot find Strike Price")

    # 7. Normalize & Predict
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

    # 8. Metrics
    rmse_market = np.sqrt(np.mean((V_market - V_pred_pinn)**2))
    corr_market = np.corrcoef(V_market, V_pred_pinn)[0, 1]
    rmse_anal = np.sqrt(np.mean((V_analytical - V_pred_pinn)**2))
    
    print("\n=== Evaluation Results ===")
    print(f"vs. Market Price   -> RMSE: {rmse_market:.4f} | R: {corr_market:.4f}")
    print(f"vs. Analytical Sol -> RMSE: {rmse_anal:.4f}")
    print("==========================\n")

    # 9. Plot
    plt.figure(figsize=(14, 8))
    ax1 = plt.gca()
    ax1.plot(t_data, V_market, label='Market Price', color='purple', alpha=0.6)
    ax1.plot(t_data, V_analytical, label='Analytical (BS)', color='dodgerblue', linestyle='-.')
    ax1.plot(t_data, V_pred_pinn, label='PINN Prediction', color='darkorange', linestyle='--')
    ax1.set_xlabel('Time to Maturity (Years)'); ax1.set_ylabel('Option Price ($)')
    ax1.set_title(f'Universal Model Test: {filename}\nRMSE(Mkt): {rmse_market:.2f}, R: {corr_market:.4f}')
    ax1.invert_xaxis(); ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(t_data, S_data, label='BTC Price', color='green', alpha=0.15, linestyle=':')
    ax2.set_ylabel('BTC Price ($)', color='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    save_path = os.path.join(RUN_FOLDER, f"test_result_{filename.replace('.csv','.png')}")
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()