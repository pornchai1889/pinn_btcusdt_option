import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import requests
import glob
from datetime import datetime
from scipy.stats import norm

# ==========================================
# CONFIGURATION
# ==========================================
# 1. โฟลเดอร์ผลลัพธ์การเทรน (ระบุถึงโฟลเดอร์ Checkpoint ที่ต้องการเทส)
RUN_FOLDER = "runs/train_2025-12-09_18-18-46_PowerLawTime_Adaptive/checkpoints/epoch_30000"  

# 2. ชื่อโมเดลที่ต้องการโหลดมาใช้งาน
MODELL = "model.pth"

# 3. โหมดการรันและการตั้งค่าไฟล์
# "BATCH"  = รันทุกไฟล์ในโฟลเดอร์ data/raw
# "SINGLE" = รันเฉพาะไฟล์ที่ระบุใน SINGLE_TARGET_FILE
RUN_MODE = "BATCH" 

DATA_RAW_DIR = "data/raw"
SINGLE_TARGET_FILE = "BTC-251206-89000-C_Daily_30m.csv" # ใช้เมื่อ RUN_MODE = "SINGLE"

# 4. ค่า r มาตรฐาน
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
            
        # Last layer
        layers.append(nn.Linear(n_hidden, n_output))
            
        # --- [ADDED] Force Positive Output ---
        # Softplus is a smooth approximation of ReLU: log(1 + exp(x))
        # It ensures output is always > 0 and differentiable everywhere
        layers.append(nn.Softplus())
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
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
            
    # print(f"Fetched total {len(all_prices)} historical BTC candles.")
    
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

def process_file(csv_file_path, model, device, train_config, output_dir, append_time_to_filename=False):
    """
    ฟังก์ชันประมวลผลแยก (Refactored จาก main เดิมเพื่อรองรับ Loop)
    """
    filename = os.path.basename(csv_file_path)
    print(f"\nProcessing: {filename}")
    
    # 1. Parse Timeframe
    tf_str, window_size, ann_factor = get_timeframe_params(filename, days=LOOKBACK_DAYS)
    print(f"Detected Timeframe: {tf_str}")
    print(f"Sigma Lookback: {LOOKBACK_DAYS} days ({window_size} candles)")

    c_m = train_config["market"]
    
    # 3. Load Main Data
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV not found at {csv_file_path}")
        return
    df = pd.read_csv(csv_file_path)
    S_data = df['btc_close_price'].values
    
    # 4. Calculate Sigma (Fetch Lookback + Calc)
    first_timestamp = df['open_time_unix_ms'].iloc[0]
    # ส่ง window_size ที่คำนวณได้เข้าไปเป็น limit (ไม่ใช้ 24 ตายตัวแล้ว)
    lookback_prices = fetch_btc_lookback(first_timestamp, tf_str, limit=window_size)
    
    sigma_data = calculate_dynamic_volatility(S_data, lookback_prices, window_size, ann_factor)
    
    # 5. Prepare Inputs
    if 'time_to_maturity_t2m' in df.columns:
        t_data = df['time_to_maturity_t2m'].values
    else: 
        print(f"Missing 'time_to_maturity_t2m' in {filename}")
        return
    
    r_data = np.full_like(S_data, RISK_FREE_RATE)
    
    if 'strike_price_K' in df.columns:
        K_data = df['strike_price_K'].values
    else:
        # Fallback: Try parsing from filename
        try:
            k_val = float(filename.split('-')[2])
            K_data = np.full_like(S_data, k_val)
        except: 
            print("Cannot find Strike Price")
            return

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
    # [รักษาโค้ดส่วนแสดงผลเดิมไว้ทั้งหมด]
    file_basename = os.path.splitext(filename)[0]
    
    # สร้าง Canvas แบ่ง 2 ส่วน (บน 3 ส่วน, ล่าง 1 ส่วน)
    fig, (ax1, ax_sigma) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Graph 1: Prices ---
    ax1.plot(t_data, V_market, label='Market Price', color='purple', alpha=0.5, linewidth=1.5)
    ax1.plot(t_data, V_analytical, label='Analytical (BS)', color='dodgerblue', linestyle='-.', alpha=0.8)
    ax1.plot(t_data, V_pred_pinn, label='PINN Prediction', color='darkorange', linestyle='--', linewidth=2)
    
    ax1.set_ylabel('Option Price: V (USDT)', fontsize=12)
    ax1.set_title(f'PINN Model Evaluation: {file_basename}\n'
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
    if append_time_to_filename:
        # กรณี Single File แปะเวลาต่อท้ายไฟล์
        timestamp_suffix = datetime.now().strftime("%H%M%S")
        save_filename = f"result_{file_basename}_sigma{LOOKBACK_DAYS}day_{timestamp_suffix}.png"
    else:
        # กรณี Batch ชื่อโฟลเดอร์ระบุเวลาไว้แล้ว
        save_filename = f"result_{file_basename}_sigma{LOOKBACK_DAYS}day_r{RISK_FREE_RATE}.png"
        
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to: {save_path}")
    
    # plt.show() # ปิดไว้หากรันจำนวนมาก
    plt.close(fig) # ปิด Figure เพื่อคืน Memory

def main():
    try: mpl.rcParams['axes.unicode_minus'] = False
    except: pass

    print(f"--- Testing Real Market Data (Mode: {RUN_MODE}) ---")
    
    # ==========================================
    # 2. [UPDATED] Load Config from Run Root
    # ==========================================
    # Logic: Start from RUN_FOLDER (checkpoints/epoch_X) and search UPWARDS for config.json
    # This makes it general for any sub-folder depth.
    
    config_path = None
    current_search_dir = RUN_FOLDER
    
    # Traverse up (Limit to 5 levels to be safe)
    for _ in range(5): 
        candidate_path = os.path.join(current_search_dir, "config.json")
        if os.path.exists(candidate_path):
            config_path = candidate_path
            break
        
        # Move up one level
        parent_dir = os.path.dirname(current_search_dir)
        if parent_dir == current_search_dir: # Reached System Root
            break
        current_search_dir = parent_dir
    
    if config_path is None:
        print(f"Error: 'config.json' not found in hierarchy starting from: {RUN_FOLDER}")
        return

    print(f"Loaded Config from: {config_path}")

    with open(config_path, 'r') as f:
        TRAIN_CONFIG = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalPINN(
        TRAIN_CONFIG["model"]["n_input"], TRAIN_CONFIG["model"]["n_output"],
        TRAIN_CONFIG["model"]["n_hidden"], TRAIN_CONFIG["model"]["n_layers"]
    ).to(device)
    
    # Model is typically inside the specific checkpoint folder (RUN_FOLDER)
    model_path = os.path.join(RUN_FOLDER, MODELL)
    if not os.path.exists(model_path): 
        # Fallback in case user put model elsewhere or generic name
        model_path = os.path.join(RUN_FOLDER, "model.pth")
        
    if os.path.exists(model_path):
        print(f"Loading Model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Error: Model file '{MODELL}' not found in {RUN_FOLDER}")
        return

    # --- Setup Output Directory Logic ---
    model_name_clean = os.path.splitext(MODELL)[0]
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if RUN_MODE == "BATCH":
        # ตั้งชื่อโฟลเดอร์: eval_batch_{Date}_{Model} (เพื่อให้เรียงตามเวลาได้ง่าย)
        folder_name = f"eval_batch_{now_str}__{model_name_clean}"
        # Save results inside the Checkpoint folder (keep original logic)
        output_dir = os.path.join(RUN_FOLDER, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Batch Output Directory: {output_dir}")
        
        # ค้นหาทุกไฟล์ CSV
        search_path = os.path.join(DATA_RAW_DIR, "*.csv")
        csv_files = glob.glob(search_path)
        print(f"Found {len(csv_files)} files in {DATA_RAW_DIR}")
        
        for file_path in csv_files:
            process_file(file_path, model, device, TRAIN_CONFIG, output_dir, append_time_to_filename=False)
            
    elif RUN_MODE == "SINGLE":
        # ใช้โฟลเดอร์กลางสำหรับ Single Snapshots
        output_dir = os.path.join(RUN_FOLDER, "eval_single_snapshots")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Single Output Directory: {output_dir}")
        
        file_path = os.path.join(DATA_RAW_DIR, SINGLE_TARGET_FILE)
        
        # ส่ง append_time_to_filename=True เพื่อให้ไฟล์ไม่ทับกัน
        process_file(file_path, model, device, TRAIN_CONFIG, output_dir, append_time_to_filename=True)
        
    else:
        print(f"Unknown RUN_MODE: {RUN_MODE}")

if __name__ == "__main__":
    main()