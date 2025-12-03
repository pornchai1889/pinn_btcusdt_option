import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
import json
from datetime import datetime
from scipy.stats import norm

# ==========================================
# INPUT ORDER TO MODEL (ห้ามเปลี่ยน):
# [0] t (Time to Maturity)
# [1] S (Spot Price)
# [2] sigma (Volatility)
# [3] r (Risk-free Rate)
# [4] K (Strike Price)
# ==========================================

# --- 1. Utility: Analytical Solution (for Validation) ---
def analytical_solution(S, K, t, r, sigma):
    """
    Calculate Black-Scholes European Call Option Price
    Used for validating the model accuracy (RMSE)
    """
    # ป้องกัน t=0 (Time to maturity = 0) เพื่อไม่ให้ error หารด้วยศูนย์
    t = np.maximum(t, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def main():
    # --- 2. Setup Directory & Logging ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"train_{current_time}_Universal5Inputs"
    
    base_output_dir = "runs"
    result_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(result_dir, exist_ok=True)

    log_filename = os.path.join(result_dir, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    
    writer = SummaryWriter(log_dir=result_dir)
    logging.info(f"--- Started Experiment: {run_name} ---")

    # --- 3. Configuration ---
    CONFIG = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "market": {
            "T_MAX": 1.0,             # 1 ปี (ครอบคลุม Weekly - Yearly)
            "S_range": [0.0, 1000000.0],
            "K_range": [10000.0, 200000.0],
            "t_range": [0.0, 1.0],    # Input t คือ Time to Maturity
            "sigma_range": [0.1, 2.0],
            "r_range": [0.0, 0.15]
        },
        "model": {
            "n_input": 5, "n_output": 1, "n_hidden": 128, "n_layers": 8
        },
        "training": {
            "epochs": 80000,
            "lr": 1e-4,
            "n_sample_data": 8000,        # จุดข้อมูล IVP/BVP
            "n_sample_pde_multiplier": 4, # จุดข้อมูล PDE (4 เท่าของ Data)
            "physics_loss_weight": 1.0,
            "val_interval": 1000,         # ตรวจสอบ RMSE ทุกๆ 1000 epoch
            "n_val_sample": 2000          # จำนวนจุดทดสอบ Validation
        }
    }
    
    DEVICE = torch.device(CONFIG["device"])
    logging.info(f"Using device: {DEVICE}")

    # Save config for reproducibility
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)

    # Extract Config for easy usage
    c_m = CONFIG["market"]
    S_min, S_max = c_m["S_range"]
    K_min, K_max = c_m["K_range"]
    t_min, t_max = c_m["t_range"]
    sig_min, sig_max = c_m["sigma_range"]
    r_min, r_max = c_m["r_range"]

    # --- 4. Normalization Utilities ---
    def normalize_val(val, v_min, v_max):
        return (val - v_min) / (v_max - v_min)

    def denormalize_val(val_norm, v_min, v_max):
        return val_norm * (v_max - v_min) + v_min

    # --- 5. Data Generation Functions ---
    def get_diff_data(n):
        # 1. K & S (Mixture Sampling)
        K_points = np.random.uniform(K_min, K_max, (n, 1))
        
        n_focus = int(n * 0.8)
        n_wide = n - n_focus
        
        moneyness = np.random.uniform(0.5, 1.5, (n_focus, 1)) 
        S_focus = K_points[:n_focus] * moneyness
        S_wide = np.random.uniform(S_min, S_max, (n_wide, 1))
        
        S_points = np.concatenate([S_focus, S_wide], axis=0)
        S_points = np.clip(S_points, S_min, S_max)

        # 2. Others
        t_points = np.random.uniform(t_min, t_max, (n, 1))
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        # 3. Normalize
        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min, S_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        return np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)

    def get_ivp_data(n):
        # IVP: Time to maturity = 0
        t_points = np.zeros((n, 1)) 
        
        K_points = np.random.uniform(K_min, K_max, (n, 1))
        moneyness = np.random.uniform(0.5, 1.5, (n, 1))
        S_points = np.clip(K_points * moneyness, S_min, S_max)
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min, S_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        X_norm = np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)
        y_val = np.fmax(S_points - K_points, 0)
        
        # Normalize Output by K (Important!)
        return X_norm, y_val / K_points

    def get_bvp_data(n):
        t_points = np.random.uniform(t_min, t_max, (n, 1))
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))
        K_points = np.random.uniform(K_min, K_max, (n, 1))
        
        t_norm = normalize_val(t_points, t_min, t_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        # Lower Bound (S -> 0) => V -> 0
        S1_norm = normalize_val(S_min * np.ones((n, 1)), S_min, S_max)
        X1_norm = np.concatenate([t_norm, S1_norm, sig_norm, r_norm, K_norm], axis=1)
        y1_val = np.zeros((n, 1))

        # Upper Bound (S -> Inf) => V -> S - K*exp(-r*t)
        S2_points = S_max * np.ones((n, 1))
        S2_norm = normalize_val(S2_points, S_min, S_max)
        X2_norm = np.concatenate([t_norm, S2_norm, sig_norm, r_norm, K_norm], axis=1)
        
        # Boundary Value
        y2_val = (S2_points - K_points * np.exp(-r_points * t_points)).reshape(-1, 1)

        return X1_norm, y1_val / K_points, X2_norm, y2_val / K_points

    # --- 6. Model Definition ---
    class UniversalPINN(nn.Module):
        def __init__(self, n_input, n_output, n_hidden, n_layers):
            super().__init__()
            activation = nn.Tanh()
            layers = [nn.Linear(n_input, n_hidden), activation]
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(activation)
            layers.append(nn.Linear(n_hidden, n_output))
            
            for m in self.net.modules() if hasattr(self, 'net') else self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            self.net = nn.Sequential(*layers)

        def forward(self, x): return self.net(x)

    # --- 7. Setup Training ---
    N_INPUT = CONFIG["model"]["n_input"]
    EPOCHS = CONFIG["training"]["epochs"]
    LR = CONFIG["training"]["lr"]
    N_SAMPLE_DATA = CONFIG["training"]["n_sample_data"]
    N_SAMPLE_PDE = N_SAMPLE_DATA * CONFIG["training"]["n_sample_pde_multiplier"]
    
    model = UniversalPINN(N_INPUT, 1, CONFIG["model"]["n_hidden"], CONFIG["model"]["n_layers"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- 7.5 Create Validation Set (Fixed set for consistent monitoring) ---
    logging.info("Generating Validation Set...")
    # สุ่มข้อมูลมาทำชุดข้อสอบ
    n_val = CONFIG["training"]["n_val_sample"]
    X_val_norm = get_diff_data(n_val) # Numpy Array
    
    # Denormalize to compute True Values (Analytical)
    t_val = denormalize_val(X_val_norm[:, 0], t_min, t_max)
    S_val = denormalize_val(X_val_norm[:, 1], S_min, S_max)
    sig_val = denormalize_val(X_val_norm[:, 2], sig_min, sig_max)
    r_val = denormalize_val(X_val_norm[:, 3], r_min, r_max)
    K_val = denormalize_val(X_val_norm[:, 4], K_min, K_max)
    
    # คำนวณเฉลย
    V_val_true = analytical_solution(S_val, K_val, t_val, r_val, sig_val)
    
    # เตรียมข้อมูลสำหรับส่งเข้าโมเดล (Tensor)
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(DEVICE)

    # --- 8. Training Loop ---
    logging.info("\n--- Starting Training ---")
    
    for i in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # --- A. Data Loss (Detailed) ---
        # IVP
        ivp_x, ivp_y = get_ivp_data(N_SAMPLE_DATA)
        ivp_pred = model(torch.from_numpy(ivp_x).float().to(DEVICE))
        loss_ivp = loss_fn(ivp_pred, torch.from_numpy(ivp_y).float().to(DEVICE))

        # BVP
        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = get_bvp_data(N_SAMPLE_DATA)
        pred_bvp1 = model(torch.from_numpy(bvp_x1).float().to(DEVICE))
        pred_bvp2 = model(torch.from_numpy(bvp_x2).float().to(DEVICE))
        
        loss_bvp1 = loss_fn(pred_bvp1, torch.from_numpy(bvp_y1).float().to(DEVICE))
        loss_bvp2 = loss_fn(pred_bvp2, torch.from_numpy(bvp_y2).float().to(DEVICE))
        loss_bvp_total = loss_bvp1 + loss_bvp2
        
        loss_data_total = loss_ivp + loss_bvp_total

        # --- B. Physics Loss (PDE) ---
        X_pde_norm = get_diff_data(N_SAMPLE_PDE)
        X_pde_tensor = torch.from_numpy(X_pde_norm).float().to(DEVICE).requires_grad_()
        v_pred_norm = model(X_pde_tensor)

        # Denormalize Inputs
        S_pde = denormalize_val(X_pde_tensor[:, 1:2], S_min, S_max)
        sigma_pde = denormalize_val(X_pde_tensor[:, 2:3], sig_min, sig_max)
        r_pde = denormalize_val(X_pde_tensor[:, 3:4], r_min, r_max)
        K_pde = denormalize_val(X_pde_tensor[:, 4:5], K_min, K_max)
        
        V_real = v_pred_norm * K_pde # Output Scale

        # Gradients
        grads = torch.autograd.grad(v_pred_norm, X_pde_tensor, grad_outputs=torch.ones_like(v_pred_norm), create_graph=True)[0]
        dv_dt_n, dv_ds_n = grads[:, 0:1], grads[:, 1:2]
        
        grads2 = torch.autograd.grad(dv_ds_n, X_pde_tensor, grad_outputs=torch.ones_like(dv_ds_n), create_graph=True)[0]
        d2v_ds2_n = grads2[:, 1:2]

        # PDE Terms
        dV_dt = (K_pde / (t_max - t_min)) * dv_dt_n
        dV_dS = (K_pde / (S_max - S_min)) * dv_ds_n
        d2V_dS2 = (K_pde / (S_max - S_min)**2) * d2v_ds2_n

        # Residual (Time to Maturity Form: dV/dtau = ...)
        pde_res = dV_dt - (0.5 * sigma_pde**2 * S_pde**2 * d2V_dS2 + r_pde * S_pde * dV_dS - r_pde * V_real)
        
        # Loss
        pde_loss = CONFIG["training"]["physics_loss_weight"] * loss_fn(pde_res / K_pde, torch.zeros_like(pde_res))

        # Total Backprop
        total_loss = loss_data_total + pde_loss
        total_loss.backward()
        optimizer.step()

        # --- C. Logging (TensorBoard) --- (ดูด้วยคำสั่ง tensorboard --logdir=runs)
        # บันทึกทุก 10 Epoch (ปรับได้)
        if i % 10 == 0:
            writer.add_scalar('Loss/Total', total_loss.item(), i)
            writer.add_scalar('Loss/PDE', pde_loss.item(), i)
            writer.add_scalar('Loss/Data_Total', loss_data_total.item(), i)
            
            # Granular Loss
            writer.add_scalar('Loss_Detail/IVP', loss_ivp.item(), i)
            writer.add_scalar('Loss_Detail/BVP_Total', loss_bvp_total.item(), i)
            writer.add_scalar('Loss_Detail/BVP1_Min', loss_bvp1.item(), i)
            writer.add_scalar('Loss_Detail/BVP2_Max', loss_bvp2.item(), i)

        # --- D. Validation Metrics (RMSE & R) ---
        # ตรวจสอบทุกๆ val_interval Epochs
        if (i + 1) % CONFIG["training"]["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                v_val_pred_ratio = model(X_val_tensor).cpu().numpy().flatten()
                
                # แปลงค่ากลับเป็นราคาจริงเพื่อเทียบกับ V_val_true
                # K_val เป็น numpy array อยู่แล้ว (จากขั้นตอน Denormalize)
                V_val_pred = v_val_pred_ratio * K_val.flatten()
                
                # RMSE
                rmse = np.sqrt(np.mean((V_val_true.flatten() - V_val_pred)**2))
                
                # R (Correlation)
                r_score = np.corrcoef(V_val_true.flatten(), V_val_pred)[0, 1]
                
                # Log Metrics
                writer.add_scalar('Metrics/RMSE', rmse, i)
                writer.add_scalar('Metrics/R_Score', r_score, i)
                
                logging.info(f"Epoch {i+1}/{EPOCHS} | Loss: {total_loss.item():.6f} | RMSE: {rmse:.4f} | R: {r_score:.4f}")
            
            model.train() # กลับเข้าโหมดเทรน

    logging.info("--- Training Finished ---")
    writer.close()

    # --- 8. Save Model ---
    model_save_path = os.path.join(result_dir, "final_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()