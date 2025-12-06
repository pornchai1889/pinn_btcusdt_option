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

# ==========================================
# INPUT ORDER TO MODEL (ห้ามเปลี่ยน):
# [0] t (Time to Maturity)
# [1] S (Spot Price)
# [2] sigma (Volatility)
# [3] r (Risk-free Rate)
# [4] K (Strike Price)
# ==========================================

# --- 1. Utility Functions ---
def analytical_solution(S, K, t, r, sigma):
    """
    Calculate Black-Scholes European Call Option Price.
    t is time to maturity (tau).
    """
    t = np.maximum(t, 1e-10) # Avoid div by zero
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def calculate_smape(true, pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    Range: 0-100% (Stable for small values)
    """
    denominator = (np.abs(true) + np.abs(pred)) / 2.0
    diff = np.abs(true - pred)
    # Avoid division by zero
    return np.mean(diff / (denominator + 1e-8)) * 100

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
    logging.info(f"--- Artifacts saved to: {result_dir} ---")

    # --- 3. Configuration ---
    CONFIG = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "market": {
            "T_MAX": 1.0,            
            "S_range": [0.0, 1000000.0],
            "K_range": [10000.0, 500000.0],
            "K_step": 1000.0,          # Step การสุ่ม K
            "t_range": [0.0, 1.0],    # Input t is Time to Maturity (หน่วนปี) 0 ถึง T_MAX
            "sigma_range": [0.1, 2.0],
            "r_range": [0.0, 0.15]
        },
        "sampling": {
            "focus_ratio": 0.8,           # 80% focus around Strike
            "moneyness_range": [0.5, 1.5],# Training range
            "trading_zone": [0.8, 1.2]    # Evaluation range
        },
        "model": {
            "n_input": 5, "n_output": 1, "n_hidden": 256, "n_layers": 4
        },
        "training": {
            "epochs": 300000,
            "lr": 1e-4,
            "n_sample_data": 10000,
            "n_sample_pde_multiplier": 4,
            "physics_loss_weight": 1.0,
            "val_interval": 1000,
            "n_val_sample": 100000
        }
    }
    
    DEVICE = torch.device(CONFIG["device"])
    logging.info(f"Using device: {DEVICE}")

    # Save Config
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)

    # Extract Config
    c_m = CONFIG["market"]
    c_s = CONFIG["sampling"]
    S_min, S_max = c_m["S_range"]
    K_min, K_max = c_m["K_range"]
    K_step_val = c_m.get("K_step", 1) # 1 เป็นค่าพื้นฐานกรณีไม่ได้ระบุ step
    t_min, t_max = c_m["t_range"]
    sig_min, sig_max = c_m["sigma_range"]
    r_min, r_max = c_m["r_range"]

    # --- 4. Normalization Utilities ---
    def normalize_val(val, v_min, v_max):
        return (val - v_min) / (v_max - v_min)

    def denormalize_val(val_norm, v_min, v_max):
        return val_norm * (v_max - v_min) + v_min

    # --- Helper for Discrete K Sampling ---
    def get_discrete_K(n, k_min, k_max, step):
        if step is None or step <= 0:
            return np.random.uniform(k_min, k_max, (n, 1))
            
        aligned_min = np.ceil(k_min / step) * step
        aligned_max = np.floor(k_max / step) * step
        
        if aligned_max < aligned_min:
            return np.random.uniform(k_min, k_max, (n, 1))
            
        n_steps = int((aligned_max - aligned_min) / step)
        random_steps = np.random.randint(0, n_steps + 1, (n, 1))
        return aligned_min + random_steps * step

    # --- 5. Data Generation Functions ---
    def get_diff_data(n):
        # 1. K & S (Mixture Sampling)
        # --- [MODIFIED] Use Discrete K ---
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        # ---------------------------------
        
        n_focus = int(n * c_s["focus_ratio"])
        n_wide = n - n_focus
        
        m_min, m_max = c_s["moneyness_range"]
        moneyness = np.random.uniform(m_min, m_max, (n_focus, 1)) 
        S_focus = K_points[:n_focus] * moneyness
        S_wide = np.random.uniform(S_min, S_max, (n_wide, 1))
        S_points = np.clip(np.concatenate([S_focus, S_wide], axis=0), S_min, S_max)

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
        # IVP: t=0
        t_points = np.zeros((n, 1)) 
        
        # --- [MODIFIED] Use Discrete K ---
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        # ---------------------------------
        
        # Mixture Sampling for IVP as well
        n_focus = int(n * c_s["focus_ratio"])
        n_wide = n - n_focus
        m_min, m_max = c_s["moneyness_range"]
        
        moneyness = np.random.uniform(m_min, m_max, (n_focus, 1))
        S_focus = K_points[:n_focus] * moneyness
        S_wide = np.random.uniform(S_min, S_max, (n_wide, 1))
        S_points = np.clip(np.concatenate([S_focus, S_wide], axis=0), S_min, S_max)

        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min, S_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        X_norm = np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)
        
        # Payoff: max(S-K, 0)
        y_val = np.fmax(S_points - K_points, 0)
        return X_norm, y_val / K_points

    def get_bvp_data(n):
        t_points = np.random.uniform(t_min, t_max, (n, 1))
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))
        
        # --- [MODIFIED] Use Discrete K ---
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        # ---------------------------------
        
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
            for m in self.modules():
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
    PHYSICS_WEIGHT = CONFIG["training"]["physics_loss_weight"]
    
    model = UniversalPINN(N_INPUT, 1, CONFIG["model"]["n_hidden"], CONFIG["model"]["n_layers"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- 7.5 Create Validation Set ---
    logging.info("Generating Validation Set (Mixture Sampling)...")
    n_val = CONFIG["training"]["n_val_sample"]
    X_val_norm = get_diff_data(n_val)
    
    # Denormalize for Validation
    t_val = denormalize_val(X_val_norm[:, 0], t_min, t_max)
    S_val = denormalize_val(X_val_norm[:, 1], S_min, S_max)
    sig_val = denormalize_val(X_val_norm[:, 2], sig_min, sig_max)
    r_val = denormalize_val(X_val_norm[:, 3], r_min, r_max)
    K_val = denormalize_val(X_val_norm[:, 4], K_min, K_max)
    
    V_val_true = analytical_solution(S_val, K_val, t_val, r_val, sig_val)
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(DEVICE)
    
    # Masks
    moneyness_val = S_val.flatten() / K_val.flatten()
    tz_min, tz_max = c_s["trading_zone"]
    mask_trading_zone = (moneyness_val >= tz_min) & (moneyness_val <= tz_max)

    # --- 8. Training Loop ---
    logging.info("\n--- Starting Training ---")
    
    for i in tqdm(range(EPOCHS), desc="Training PINN", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        
        # Data Loss
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

        # Physics Loss
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

        # --- C. Logging (TensorBoard) ---
        if i % 10 == 0:
            writer.add_scalar('Loss/Total', total_loss.item(), i)
            writer.add_scalar('Loss/PDE', pde_loss.item(), i)
            writer.add_scalar('Loss/Data_Total', data_loss.item(), i)
            writer.add_scalar('Loss_Detail/IVP', loss_ivp.item(), i)
            writer.add_scalar('Loss_Detail/BVP_Total', loss_bvp_total.item(), i)
            writer.add_scalar('Loss_Detail/BVP1_Min', loss_bvp1.item(), i)
            writer.add_scalar('Loss_Detail/BVP2_Max', loss_bvp2.item(), i)

        # --- D. Validation Metrics (Interval Check) ---
        if (i + 1) % CONFIG["training"]["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                v_val_pred_ratio = model(X_val_tensor).cpu().numpy().flatten()
                V_val_pred = v_val_pred_ratio * K_val.flatten()
                V_true = V_val_true.flatten()
                
                # 1. Global Metrics
                diff = V_val_pred - V_true 
                rmse_g = np.sqrt(np.mean(diff**2))
                mae_g = np.mean(np.abs(diff))
                max_err_g = np.max(np.abs(diff))
                bias_g = np.mean(diff)
                r_g = np.corrcoef(V_true, V_val_pred)[0, 1]
                # SMAPE Global
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

                # 3. Log to TensorBoard
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

                # 4. Log to Text File
                log_msg = (
                    f"Epoch {i+1:5d} | "
                    f"Loss: {total_loss.item():.8f} (PDE:{pde_loss.item():.8f} Data:{data_loss.item():.8f}) | "
                    f"Detail: [IVP:{loss_ivp.item():.8f} BVP:{loss_bvp_total.item():.8f} (L:{loss_bvp1.item():.8f} U:{loss_bvp2.item():.8f})] | "
                    f"Global: [RMSE:{rmse_g:.2f} MAE:{mae_g:.2f} SMAPE:{smape_g:.2f}% Bias:{bias_g:.2f} R:{r_g:.4f} MaxErr:{max_err_g:.2f}] | "
                    f"TZ: [RMSE:{rmse_tz:.2f} MAE:{mae_tz:.2f} SMAPE:{smape_tz:.2f}% Bias:{bias_tz:.2f} R:{r_tz:.4f} MaxErr:{max_err_tz:.2f}]"
                )
                logging.info(log_msg)
                
        if (i + 1) % 10000 == 0:
            filename = f"checkpoint_epoch_{i+1}.pth"
            torch.save(model.state_dict(), os.path.join(result_dir, filename))
            print(f"Saved checkpoint: {filename}")
                    
            model.train()

    logging.info("--- Training Finished ---")
    writer.close()

    # --- 8. Save Model ---
    model_save_path = os.path.join(result_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    # --- (GPU Warmup) ---
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        print("GPU Warmed up and ready!")
        
    main()
    
    #กดรันบน terminal แยกไว้ก่อนเพื่อเตรียมดูผลการเทรน: tensorboard --logdir=runs