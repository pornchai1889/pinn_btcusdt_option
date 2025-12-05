import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
import json
from datetime import datetime
from scipy.stats import norm
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION FOR FINE-TUNING
# ==============================================================================
BASE_RUN_DIR = "runs/train_2025-12-04_12-33-35_Universal5Inputs" 

FT_CONFIG = {
    "epochs": 200000,
    "lr": 1e-5,                 # LR ต่ำๆ
    "n_sample_data": 10000,     
    "n_sample_pde_multiplier": 5,
    "physics_loss_weight": 1.0,
    "val_interval": 1000,
    
    # --- [ส่วนที่ปรับได้อิสระ] ---
    "sampling": {
        # 1. Focus Moneyness: เน้น S รอบๆ K (เหมือนเดิม)
        "focus_ratio": 0.9,           
        "moneyness_range": [0.8, 1.2], # บีบให้แคบลงเพื่อความแม่นยำ
        
        # 2. [ใหม่] Focus K: เน้นสุ่ม K เฉพาะช่วงที่เราสนใจ (เช่น ราคาตลาดปัจจุบัน)
        # ถ้าไม่กำหนด (None) จะสุ่ม 10k-200k เหมือนเดิม
        # โมเดลจะเน้นเทรนแค่ช่วงนี้ แต่ใช้สเกลเดิม
        "target_k_range": [80000.0, 120000.0], 

        "trading_zone": [0.8, 1.2]    
    }
}
# ==============================================================================

# --- Utility Functions ---
def analytical_solution(S, K, t, r, sigma):
    t = np.maximum(t, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def calculate_smape(true, pred):
    denominator = (np.abs(true) + np.abs(pred)) / 2.0
    diff = np.abs(true - pred)
    return np.mean(diff / (denominator + 1e-8)) * 100

def main():
    # --- 1. Load Original Config ---
    config_path = os.path.join(BASE_RUN_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        CONFIG = json.load(f)
    
    # Update Config (เฉพาะส่วนที่อนุญาต)
    CONFIG["training"].update({k:v for k,v in FT_CONFIG.items() if k in CONFIG["training"]})
    CONFIG["sampling"] = FT_CONFIG["sampling"]

    # --- 2. Setup Directory ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ft_folder_name = f"ft_{current_time}_TargetedK"
    ft_result_dir = os.path.join(BASE_RUN_DIR, "fine_tune", ft_folder_name)
    os.makedirs(ft_result_dir, exist_ok=True)

    log_filename = os.path.join(ft_result_dir, "finetune_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    writer = SummaryWriter(log_dir=ft_result_dir)
    
    logging.info(f"--- Started Fine-Tuning: {ft_folder_name} ---")
    
    with open(os.path.join(ft_result_dir, "config_ft.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)

    # Extract Params (ใช้ Scale เดิม ห้ามเปลี่ยน!)
    DEVICE = torch.device(CONFIG["device"])
    c_m = CONFIG["market"]
    c_s = CONFIG["sampling"]
    S_min, S_max = c_m["S_range"]
    K_min, K_max = c_m["K_range"] # นี่คือสเกล 0-1 ของโมเดล (ห้ามแก้)
    t_min, t_max = c_m["t_range"]
    sig_min, sig_max = c_m["sigma_range"]
    r_min, r_max = c_m["r_range"]

    # --- 3. Normalization ---
    def normalize_val(val, v_min, v_max):
        return (val - v_min) / (v_max - v_min)
    def denormalize_val(val_norm, v_min, v_max):
        return val_norm * (v_max - v_min) + v_min

    # --- Data Gen (ปรับปรุงใหม่: รองรับ Target K) ---
    def get_diff_data(n):
        # 1. สุ่ม K (ตาม Target ใหม่ ถ้ามี)
        if c_s.get("target_k_range"):
            tk_min, tk_max = c_s["target_k_range"]
            # สุ่มในช่วงแคบๆ ที่เราสนใจ (เช่น 90k-100k)
            K_points = np.random.uniform(tk_min, tk_max, (n, 1))
        else:
            # ถ้าไม่กำหนด ก็สุ่มทั่วจักรวาลเหมือนเดิม
            K_points = np.random.uniform(K_min, K_max, (n, 1))
        
        # 2. สุ่ม S (Mixture)
        n_focus = int(n * c_s["focus_ratio"])
        n_wide = n - n_focus
        m_min, m_max = c_s["moneyness_range"]
        
        # Focus Group
        moneyness = np.random.uniform(m_min, m_max, (n_focus, 1)) 
        S_focus = K_points[:n_focus] * moneyness
        
        # Wide Group (ยังต้องมี เพื่อกันลืมความรู้เดิม)
        S_wide = np.random.uniform(S_min, S_max, (n_wide, 1))
        
        S_points = np.clip(np.concatenate([S_focus, S_wide], axis=0), S_min, S_max)

        # 3. Normalize (สำคัญ! ต้องใช้ K_min, K_max ตัวเดิมจาก Config แม่)
        K_norm = normalize_val(K_points, K_min, K_max) 
        S_norm = normalize_val(S_points, S_min, S_max)
        
        # Others
        t_points = np.random.uniform(t_min, t_max, (n, 1))
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        t_norm = normalize_val(t_points, t_min, t_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)

        return np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)

    # (ฟังก์ชัน get_ivp_data และ get_bvp_data ก็ต้องแก้ logic สุ่ม K เหมือนกัน)
    def get_ivp_data(n):
        t_points = np.zeros((n, 1))
        
        if c_s.get("target_k_range"):
            tk_min, tk_max = c_s["target_k_range"]
            K_points = np.random.uniform(tk_min, tk_max, (n, 1))
        else:
            K_points = np.random.uniform(K_min, K_max, (n, 1))
            
        n_focus = int(n * c_s["focus_ratio"])
        n_wide = n - n_focus
        m_min, m_max = c_s["moneyness_range"]
        moneyness = np.random.uniform(m_min, m_max, (n_focus, 1))
        S_focus = K_points[:n_focus] * moneyness
        S_wide = np.random.uniform(S_min, S_max, (n_wide, 1))
        S_points = np.clip(np.concatenate([S_focus, S_wide], axis=0), S_min, S_max)

        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        # Normalize (ใช้ Scale เดิม)
        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min, S_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        X_norm = np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)
        y_val = np.fmax(S_points - K_points, 0)
        return X_norm, y_val / K_points

    def get_bvp_data(n):
        # BVP ก็ควรเน้น K ช่วงนี้ด้วย
        if c_s.get("target_k_range"):
            tk_min, tk_max = c_s["target_k_range"]
            K_points = np.random.uniform(tk_min, tk_max, (n, 1))
        else:
            K_points = np.random.uniform(K_min, K_max, (n, 1))
            
        t_points = np.random.uniform(t_min, t_max, (n, 1))
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))
        
        t_norm = normalize_val(t_points, t_min, t_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        S1_norm = normalize_val(S_min * np.ones((n, 1)), S_min, S_max)
        X1_norm = np.concatenate([t_norm, S1_norm, sig_norm, r_norm, K_norm], axis=1)
        y1_val = np.zeros((n, 1))

        S2_points = S_max * np.ones((n, 1))
        S2_norm = normalize_val(S2_points, S_min, S_max)
        X2_norm = np.concatenate([t_norm, S2_norm, sig_norm, r_norm, K_norm], axis=1)
        y2_val = (S2_points - K_points * np.exp(-r_points * t_points)).reshape(-1, 1)

        return X1_norm, y1_val / K_points, X2_norm, y2_val / K_points

    # --- 4. Load Model & Train Loop (เหมือนเดิม) ---
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

    model = UniversalPINN(
        CONFIG["model"]["n_input"], CONFIG["model"]["n_output"], 
        CONFIG["model"]["n_hidden"], CONFIG["model"]["n_layers"]
    ).to(DEVICE)

    model_path = os.path.join(BASE_RUN_DIR, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    logging.info("Loaded Pre-trained Weights.")

    optimizer = torch.optim.Adam(model.parameters(), lr=FT_CONFIG["lr"])
    loss_fn = nn.MSELoss()

    # --- Validation Set (สร้างตาม Target K) ---
    logging.info("Generating Focused Validation Set...")
    n_val = 50000
    X_val_norm = get_diff_data(n_val) # จะสุ่ม K ในช่วง Target เอง
    
    t_val = denormalize_val(X_val_norm[:, 0], t_min, t_max)
    S_val = denormalize_val(X_val_norm[:, 1], S_min, S_max)
    sig_val = denormalize_val(X_val_norm[:, 2], sig_min, sig_max)
    r_val = denormalize_val(X_val_norm[:, 3], r_min, r_max)
    K_val = denormalize_val(X_val_norm[:, 4], K_min, K_max)
    V_val_true = analytical_solution(S_val, K_val, t_val, r_val, sig_val)
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(DEVICE)
    
    moneyness_val = S_val.flatten() / K_val.flatten()
    mask_trading_zone = (moneyness_val >= c_s["trading_zone"][0]) & (moneyness_val <= c_s["trading_zone"][1])

    # --- Loop ---
    logging.info("\n--- Starting Fine-Tuning ---")
    EPOCHS = FT_CONFIG["epochs"]
    N_SAMPLE_DATA = FT_CONFIG["n_sample_data"]
    N_SAMPLE_PDE = N_SAMPLE_DATA * FT_CONFIG["n_sample_pde_multiplier"]
    PHYSICS_WEIGHT = FT_CONFIG["physics_loss_weight"]

    for i in tqdm(range(EPOCHS), desc="Fine-Tuning", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        
        # A. Data Loss
        ivp_x, ivp_y = get_ivp_data(N_SAMPLE_DATA)
        ivp_pred = model(torch.from_numpy(ivp_x).float().to(DEVICE))
        loss_ivp = loss_fn(ivp_pred, torch.from_numpy(ivp_y).float().to(DEVICE))

        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = get_bvp_data(N_SAMPLE_DATA)
        pred_bvp1 = model(torch.from_numpy(bvp_x1).float().to(DEVICE))
        pred_bvp2 = model(torch.from_numpy(bvp_x2).float().to(DEVICE))
        
        loss_bvp1 = loss_fn(pred_bvp1, torch.from_numpy(bvp_y1).float().to(DEVICE))
        loss_bvp2 = loss_fn(pred_bvp2, torch.from_numpy(bvp_y2).float().to(DEVICE))
        loss_bvp_total = loss_bvp1 + loss_bvp2
        
        data_loss = loss_ivp + loss_bvp_total

        # B. PDE Loss
        X_pde_norm = get_diff_data(N_SAMPLE_PDE)
        X_pde_tensor = torch.from_numpy(X_pde_norm).float().to(DEVICE).requires_grad_()
        v_pred_norm = model(X_pde_tensor)

        S_pde = denormalize_val(X_pde_tensor[:, 1:2], S_min, S_max)
        sigma_pde = denormalize_val(X_pde_tensor[:, 2:3], sig_min, sig_max)
        r_pde = denormalize_val(X_pde_tensor[:, 3:4], r_min, r_max)
        K_pde = denormalize_val(X_pde_tensor[:, 4:5], K_min, K_max)
        V_real = v_pred_norm * K_pde

        grads = torch.autograd.grad(v_pred_norm, X_pde_tensor, grad_outputs=torch.ones_like(v_pred_norm), create_graph=True)[0]
        dv_dt_n, dv_ds_n = grads[:, 0:1], grads[:, 1:2]
        grads2 = torch.autograd.grad(dv_ds_n, X_pde_tensor, grad_outputs=torch.ones_like(dv_ds_n), create_graph=True)[0]
        d2v_ds2_n = grads2[:, 1:2]

        dV_dt = (K_pde / (t_max - t_min)) * dv_dt_n
        dV_dS = (K_pde / (S_max - S_min)) * dv_ds_n
        d2V_dS2 = (K_pde / (S_max - S_min)**2) * d2v_ds2_n

        pde_res = dV_dt - (0.5 * sigma_pde**2 * S_pde**2 * d2V_dS2 + r_pde * S_pde * dV_dS - r_pde * V_real)
        pde_loss = PHYSICS_WEIGHT * loss_fn(pde_res / K_pde, torch.zeros_like(pde_res))

        total_loss = data_loss + pde_loss
        total_loss.backward()
        optimizer.step()

        # C. Detailed Logging (TensorBoard)
        if i % 100 == 0:
            writer.add_scalar('Loss/Total', total_loss.item(), i)
            writer.add_scalar('Loss/PDE', pde_loss.item(), i)
            writer.add_scalar('Loss/Data_Total', data_loss.item(), i)
            
            # Granular Loss (เก็บครบเหมือนต้นฉบับ)
            writer.add_scalar('Loss_Detail/IVP', loss_ivp.item(), i)
            writer.add_scalar('Loss_Detail/BVP_Total', loss_bvp_total.item(), i)
            writer.add_scalar('Loss_Detail/BVP1_Min', loss_bvp1.item(), i)
            writer.add_scalar('Loss_Detail/BVP2_Max', loss_bvp2.item(), i)

        # D. Full Validation Metrics (เหมือนต้นฉบับ)
        if (i + 1) % FT_CONFIG["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                v_val_pred_ratio = model(X_val_tensor).cpu().numpy().flatten()
                V_val_pred = v_val_pred_ratio * K_val.flatten()
                V_true = V_val_true.flatten()
                
                diff = V_val_pred - V_true 
                
                # 1. Global Metrics
                rmse_g = np.sqrt(np.mean(diff**2))
                mae_g = np.mean(np.abs(diff))
                max_err_g = np.max(np.abs(diff))
                bias_g = np.mean(diff)
                r_g = np.corrcoef(V_true, V_val_pred)[0, 1]
                smape_g = calculate_smape(V_true, V_val_pred)

                # 2. Trading Zone Metrics
                rmse_tz, mae_tz, max_err_tz, bias_tz, r_tz, smape_tz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if np.sum(mask_trading_zone) > 0:
                    v_true_tz = V_true[mask_trading_zone]
                    v_pred_tz = V_val_pred[mask_trading_zone]
                    diff_tz = v_pred_tz - v_true_tz
                    
                    rmse_tz = np.sqrt(np.mean(diff_tz**2))
                    mae_tz = np.mean(np.abs(diff_tz))
                    max_err_tz = np.max(np.abs(diff_tz))
                    bias_tz = np.mean(diff_tz)
                    r_tz = np.corrcoef(v_true_tz, v_pred_tz)[0, 1]
                    smape_tz = calculate_smape(v_true_tz, v_pred_tz)

                # Log Metrics to TensorBoard
                writer.add_scalar('Metrics_Global/RMSE', rmse_g, i)
                writer.add_scalar('Metrics_Global/MAE', mae_g, i)
                writer.add_scalar('Metrics_Global/SMAPE', smape_g, i)
                writer.add_scalar('Metrics_Global/Bias', bias_g, i)
                writer.add_scalar('Metrics_Global/R', r_g, i)
                writer.add_scalar('Metrics_Global/Max_Error', max_err_g, i)
                
                writer.add_scalar('Metrics_TZ/RMSE', rmse_tz, i)
                writer.add_scalar('Metrics_TZ/MAE', mae_tz, i)
                writer.add_scalar('Metrics_TZ/SMAPE', smape_tz, i)
                writer.add_scalar('Metrics_TZ/Bias', bias_tz, i)
                writer.add_scalar('Metrics_TZ/R', r_tz, i)
                writer.add_scalar('Metrics_TZ/Max_Error', max_err_tz, i)

                # Log to Text File (Detailed)
                log_msg = (
                    f"Epoch {i+1:5d} | "
                    f"Loss: {total_loss.item():.5f} (PDE:{pde_loss.item():.5f} Data:{data_loss.item():.5f}) | "
                    f"Detail: [IVP:{loss_ivp.item():.5f} BVP:{loss_bvp_total.item():.5f} (L:{loss_bvp1.item():.5f} U:{loss_bvp2.item():.5f})] | "
                    f"Global: [RMSE:{rmse_g:.2f} MAE:{mae_g:.2f} SMAPE:{smape_g:.2f}% Bias:{bias_g:.2f} R:{r_g:.4f} MaxErr:{max_err_g:.2f}] | "
                    f"TZ: [RMSE:{rmse_tz:.2f} MAE:{mae_tz:.2f} SMAPE:{smape_tz:.2f}% Bias:{bias_tz:.2f} R:{r_tz:.4f} MaxErr:{max_err_tz:.2f}]"
                )
                logging.info(log_msg)
            model.train()

    logging.info("--- Fine-Tuning Finished ---")
    writer.close()

    model_save_path = os.path.join(ft_result_dir, "finetuned_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Fine-Tuned Model saved to: {model_save_path}")

if __name__ == "__main__":
    if torch.cuda.is_available(): torch.zeros(1).cuda()
    main()