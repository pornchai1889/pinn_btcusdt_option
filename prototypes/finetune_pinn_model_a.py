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
# ระบุ Path ของ Run เดิมที่จะดึงมาจูน (ไม่ว่าจะเป็น Root หรือลูกที่เคยจูนมาแล้วก็ได้)
BASE_RUN_DIR = "runs/train_2025-12-07_08-17-11_DynamicBoundaries/fine_tune/ft_2025-12-08_09-01-39" 
MODEL_NAME = "checkpoint_epoch_190000.pth" # หรือไฟล์ล่าสุดหรือที่ loss ต่ำๆ

FT_CONFIG = {
    "epochs": 600000,
    "lr": 1e-5,                 # Learning Rate ต่ำๆ หรือเท่ากับต้นโมเดลต้นแบบ เพื่อประคอง Weight เดิม
    "n_sample_data": 10000,
    "n_sample_pde_multiplier": 4,
    "physics_loss_weight": 1.0,
    "val_interval": 1000,
    "n_val_samples": 100000,
    
    # --- [Sampling Configuration] ---
    "sampling": {
        # ไม่มี focus_ratio แล้ว
        # กำหนดกรอบ S ให้วิ่งรอบๆ K ในช่วง Moneyness นี้เท่านั้น (Dynamic Domain)
        "moneyness_range": [0.5, 1.5], # ตัวอย่าง: จูนให้เก่งเฉพาะช่วงแคบๆ รอบ ATM
        
        # Step การสุ่ม Strike Price
        "K_step": 1000.0, 

        # [Target Ranges]: ช่วงที่ต้องการเน้นเป็นพิเศษ (Fine-tune Scope)
        # ถ้าค่าไหนเป็น None จะไปดึง Global Range เดิมมาใช้
        "target_ranges": {
            "K": [10000.0, 500000.0],  # ตัวอย่าง: เน้นช่วงราคา BTC ปัจจุบัน
            "r": [0.0, 0.15],         # เน้นดอกเบี้ยช่วงนี้        
            "sigma": [0.1, 2.0],       # เน้น Volatility สูง
            "t": [0.0, 0.25]           # เน้นสัญญาใกล้หมดอายุ
        }
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
    # --- 1. Load Original Config (Mother Universe) ---
    config_path = os.path.join(BASE_RUN_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        MOTHER_CONFIG = json.load(f)
    
    # Merge Configs
    MOTHER_CONFIG["training"].update({k:v for k,v in FT_CONFIG.items() if k in MOTHER_CONFIG["training"]})
    MOTHER_CONFIG["sampling"] = FT_CONFIG["sampling"] # Overwrite sampling logic

    # --- 2. Setup Directory (Universal Path Logic) ---
    # Logic: ตรวจสอบว่า BASE_RUN_DIR เป็นลูก (fine_tune) หรือ Root
    # ถ้ามีคำว่า "fine_tune" อยู่ใน path ให้ตัดออกเพื่อหา Root ที่แท้จริง
    # เพื่อให้ ft_result_dir ไปโผล่ที่ runs/root_name/fine_tune/ เสมอ ไม่ซ้อนลึก
    
    # Standardize path separator to forward slash for checking
    normalized_base_path = BASE_RUN_DIR.replace("\\", "/")
    
    if "/fine_tune" in normalized_base_path:
        # กรณีจูนต่อจากตัวที่จูนมาแล้ว (Nested) -> ตัดกลับไปหา Root
        root_run_dir = normalized_base_path.split("/fine_tune")[0]
        # แปลงกลับเป็น OS separator ปัจจุบัน (Windows/Linux)
        root_run_dir = os.path.normpath(root_run_dir)
    else:
        # กรณีจูนจากโมเดลต้นฉบับครั้งแรก (Root)
        root_run_dir = BASE_RUN_DIR

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ft_folder_name = f"ft_{current_time}"
    
    # Save Path จะอยู่ที่: Root_Run/fine_tune/ft_Dateเสมอ
    ft_result_dir = os.path.join(root_run_dir, "fine_tune", ft_folder_name)
    os.makedirs(ft_result_dir, exist_ok=True)

    log_filename = os.path.join(ft_result_dir, "finetune_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    writer = SummaryWriter(log_dir=ft_result_dir)
    
    logging.info(f"--- Started Fine-Tuning ---")
    logging.info(f"Source Model: {os.path.join(BASE_RUN_DIR, MODEL_NAME)}")
    logging.info(f"Output Directory: {ft_result_dir}")
    
    with open(os.path.join(ft_result_dir, "config.json"), 'w') as f:
        json.dump(MOTHER_CONFIG, f, indent=4)

    # Extract Global Params (Scale เดิม ห้ามเปลี่ยน! เพื่อรักษาความรู้เดิม)
    DEVICE = torch.device(MOTHER_CONFIG["device"])
    c_m = MOTHER_CONFIG["market"]
    c_s = MOTHER_CONFIG["sampling"] # sampling ใหม่สำหรับ FT
    
    # Global Ranges (สำหรับ Normalization)
    S_min_glob, S_max_glob = c_m["S_range"]
    K_min_glob, K_max_glob = c_m["K_range"]
    t_min_glob, t_max_glob = c_m["t_range"]
    sig_min_glob, sig_max_glob = c_m["sigma_range"]
    r_min_glob, r_max_glob = c_m["r_range"]

    # --- 3. Normalization Helpers ---
    def normalize_val(val, v_min, v_max):
        return (val - v_min) / (v_max - v_min)
    def denormalize_val(val_norm, v_min, v_max):
        return val_norm * (v_max - v_min) + v_min

    # Helper: เลือกช่วงสุ่ม (Target vs Global)
    def get_sample_range(global_min, global_max, target_key):
        targets = c_s.get("target_ranges", {})
        if targets and targets.get(target_key):
            return targets[target_key][0], targets[target_key][1]
        return global_min, global_max

    # Helper: ฟังก์ชันสำหรับสุ่ม K แบบ Discrete Step
    def get_discrete_K(n, k_min_target, k_max_target, step):
        if step is None or step <= 0:
             return np.random.uniform(k_min_target, k_max_target, (n, 1))

        aligned_min = np.ceil(k_min_target / step) * step
        aligned_max = np.floor(k_max_target / step) * step
        
        if aligned_max < aligned_min:
            return np.random.uniform(k_min_target, k_max_target, (n, 1))
        
        n_steps = int((aligned_max - aligned_min) / step)
        random_steps = np.random.randint(0, n_steps + 1, (n, 1))
        return aligned_min + random_steps * step

    # --- 4. Data Gen (Dynamic Domain for Fine-tuning) ---
    def get_diff_data(n):
        # 1. Get Fine-tuning Targets
        curr_K_min, curr_K_max = get_sample_range(K_min_glob, K_max_glob, "K")
        curr_t_min, curr_t_max = get_sample_range(t_min_glob, t_max_glob, "t")
        curr_sig_min, curr_sig_max = get_sample_range(sig_min_glob, sig_max_glob, "sigma")
        curr_r_min, curr_r_max = get_sample_range(r_min_glob, r_max_glob, "r")

        # 2. Sample Parameters
        k_step_val = c_s.get("K_step", 1000.0) 
        K_points = get_discrete_K(n, curr_K_min, curr_K_max, step=k_step_val)
        
        t_points = np.random.uniform(curr_t_min, curr_t_max, (n, 1))
        sigma_points = np.random.uniform(curr_sig_min, curr_sig_max, (n, 1))
        r_points = np.random.uniform(curr_r_min, curr_r_max, (n, 1))

        # 3. Sample S based on Dynamic Moneyness (Fine-tuning Scope)
        m_min, m_max = c_s["moneyness_range"]
        moneyness = np.random.uniform(m_min, m_max, (n, 1)) 
        S_points = K_points * moneyness
        
        # Clip to Global boundaries (Just in case)
        S_points = np.clip(S_points, S_min_glob, S_max_glob)

        # 4. Normalize (Using GLOBAL Anchors)
        K_norm = normalize_val(K_points, K_min_glob, K_max_glob) 
        S_norm = normalize_val(S_points, S_min_glob, S_max_glob)
        t_norm = normalize_val(t_points, t_min_glob, t_max_glob)
        sig_norm = normalize_val(sigma_points, sig_min_glob, sig_max_glob)
        r_norm = normalize_val(r_points, r_min_glob, r_max_glob)

        return np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)

    def get_ivp_data(n):
        # IVP: t=0
        curr_K_min, curr_K_max = get_sample_range(K_min_glob, K_max_glob, "K")
        curr_sig_min, curr_sig_max = get_sample_range(sig_min_glob, sig_max_glob, "sigma")
        curr_r_min, curr_r_max = get_sample_range(r_min_glob, r_max_glob, "r")
        
        t_points = np.zeros((n, 1))
        
        k_step_val = c_s.get("K_step", 1000.0)
        K_points = get_discrete_K(n, curr_K_min, curr_K_max, step=k_step_val)
        
        sigma_points = np.random.uniform(curr_sig_min, curr_sig_max, (n, 1))
        r_points = np.random.uniform(curr_r_min, curr_r_max, (n, 1))

        m_min, m_max = c_s["moneyness_range"]
        moneyness = np.random.uniform(m_min, m_max, (n, 1))
        S_points = K_points * moneyness
        S_points = np.clip(S_points, S_min_glob, S_max_glob)

        # Normalize
        t_norm = normalize_val(t_points, t_min_glob, t_max_glob)
        S_norm = normalize_val(S_points, S_min_glob, S_max_glob)
        sig_norm = normalize_val(sigma_points, sig_min_glob, sig_max_glob)
        r_norm = normalize_val(r_points, r_min_glob, r_max_glob)
        K_norm = normalize_val(K_points, K_min_glob, K_max_glob)

        X_norm = np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)
        y_val = np.fmax(S_points - K_points, 0)
        return X_norm, y_val / K_points

    def get_bvp_data(n):
        # Params
        curr_K_min, curr_K_max = get_sample_range(K_min_glob, K_max_glob, "K")
        curr_t_min, curr_t_max = get_sample_range(t_min_glob, t_max_glob, "t")
        curr_sig_min, curr_sig_max = get_sample_range(sig_min_glob, sig_max_glob, "sigma")
        curr_r_min, curr_r_max = get_sample_range(r_min_glob, r_max_glob, "r")

        k_step_val = c_s.get("K_step", 1000.0)
        K_points = get_discrete_K(n, curr_K_min, curr_K_max, step=k_step_val)

        t_points = np.random.uniform(curr_t_min, curr_t_max, (n, 1))
        sigma_points = np.random.uniform(curr_sig_min, curr_sig_max, (n, 1))
        r_points = np.random.uniform(curr_r_min, curr_r_max, (n, 1))
        
        # Common Norm
        t_norm = normalize_val(t_points, t_min_glob, t_max_glob)
        sig_norm = normalize_val(sigma_points, sig_min_glob, sig_max_glob)
        r_norm = normalize_val(r_points, r_min_glob, r_max_glob)
        K_norm = normalize_val(K_points, K_min_glob, K_max_glob)
        
        # Moneyness Limits
        m_min, m_max = c_s["moneyness_range"]

        # --- Lower Boundary: S = K * m_min ---
        S1_points = K_points * m_min
        S1_points = np.clip(S1_points, S_min_glob, S_max_glob)
        S1_norm = normalize_val(S1_points, S_min_glob, S_max_glob)
        
        X1_norm = np.concatenate([t_norm, S1_norm, sig_norm, r_norm, K_norm], axis=1)
        y1_val = np.zeros((n, 1)) # Force 0 at lower bound

        # --- Upper Boundary: S = K * m_max ---
        S2_points = K_points * m_max
        S2_points = np.clip(S2_points, S_min_glob, S_max_glob)
        S2_norm = normalize_val(S2_points, S_min_glob, S_max_glob)
        
        X2_norm = np.concatenate([t_norm, S2_norm, sig_norm, r_norm, K_norm], axis=1)
        # Deep ITM Approximation
        y2_val = (S2_points - K_points * np.exp(-r_points * t_points))
        y2_val = np.maximum(y2_val, 0) # Ensure non-negative

        return X1_norm, y1_val / K_points, X2_norm, y2_val / K_points

    # --- 5. Load Model ---
    class UniversalPINN(nn.Module):
        def __init__(self, n_input, n_output, n_hidden, n_layers):
            super().__init__()
            activation = nn.Tanh()
            layers = [nn.Linear(n_input, n_hidden), activation]
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(activation)
            layers.append(nn.Linear(n_hidden, n_output))
            # --- [ADDED] Force Positive Output ---
            layers.append(nn.Softplus()) 
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    model = UniversalPINN(
        MOTHER_CONFIG["model"]["n_input"], MOTHER_CONFIG["model"]["n_output"], 
        MOTHER_CONFIG["model"]["n_hidden"], MOTHER_CONFIG["model"]["n_layers"]
    ).to(DEVICE)

    model_path = os.path.join(BASE_RUN_DIR, MODEL_NAME)
    # Load state dict (Softplus ไม่มี parameter ดังนั้นโหลดได้ปกติแม้ของเดิมไม่มี)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    logging.info("Loaded Pre-trained Weights (Adapted to Positive Constraint).")

    optimizer = torch.optim.Adam(model.parameters(), lr=FT_CONFIG["lr"])
    loss_fn = nn.MSELoss()

    # --- Validation Set (Targeted) ---
    logging.info("Generating Targeted Validation Set...")
    N_VAL = FT_CONFIG["n_val_samples"]
    X_val_norm = get_diff_data(N_VAL)
    
    t_val = denormalize_val(X_val_norm[:, 0], t_min_glob, t_max_glob)
    S_val = denormalize_val(X_val_norm[:, 1], S_min_glob, S_max_glob)
    sig_val = denormalize_val(X_val_norm[:, 2], sig_min_glob, sig_max_glob)
    r_val = denormalize_val(X_val_norm[:, 3], r_min_glob, r_max_glob)
    K_val = denormalize_val(X_val_norm[:, 4], K_min_glob, K_max_glob)
    
    # คำนวณค่าจริง (Price)
    V_val_true = analytical_solution(S_val, K_val, t_val, r_val, sig_val)
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(DEVICE)

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

        S_pde = denormalize_val(X_pde_tensor[:, 1:2], S_min_glob, S_max_glob)
        sigma_pde = denormalize_val(X_pde_tensor[:, 2:3], sig_min_glob, sig_max_glob)
        r_pde = denormalize_val(X_pde_tensor[:, 3:4], r_min_glob, r_max_glob)
        K_pde = denormalize_val(X_pde_tensor[:, 4:5], K_min_glob, K_max_glob)
        V_real = v_pred_norm * K_pde

        grads = torch.autograd.grad(v_pred_norm, X_pde_tensor, grad_outputs=torch.ones_like(v_pred_norm), create_graph=True)[0]
        dv_dt_n, dv_ds_n = grads[:, 0:1], grads[:, 1:2]
        grads2 = torch.autograd.grad(dv_ds_n, X_pde_tensor, grad_outputs=torch.ones_like(dv_ds_n), create_graph=True)[0]
        d2v_ds2_n = grads2[:, 1:2]

        dV_dt = (K_pde / (t_max_glob - t_min_glob)) * dv_dt_n
        dV_dS = (K_pde / (S_max_glob - S_min_glob)) * dv_ds_n
        d2V_dS2 = (K_pde / (S_max_glob - S_min_glob)**2) * d2v_ds2_n

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
            writer.add_scalar('Loss_Detail/IVP', loss_ivp.item(), i)
            writer.add_scalar('Loss_Detail/BVP_Total', loss_bvp_total.item(), i)
            writer.add_scalar('Loss_Detail/BVP1_Min', loss_bvp1.item(), i)
            writer.add_scalar('Loss_Detail/BVP2_Max', loss_bvp2.item(), i)

        # --- D. Validation Metrics (Adjusted to Ratio Scale) ---
        if (i + 1) % FT_CONFIG["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                # 1. Prediction (V/K)
                v_val_pred_ratio = model(X_val_tensor).cpu().numpy().flatten()
                
                # 2. Ground Truth (Convert V -> V/K)
                v_val_true_ratio = V_val_true.flatten() / K_val.flatten()
                
                # 3. Calculate Metrics on Ratio Scale
                diff_ratio = v_val_pred_ratio - v_val_true_ratio
                
                rmse_r = np.sqrt(np.mean(diff_ratio**2))
                mae_r = np.mean(np.abs(diff_ratio))
                max_err_r = np.max(np.abs(diff_ratio))
                bias_r = np.mean(diff_ratio)
                
                # Correlation
                if np.std(v_val_true_ratio) == 0 or np.std(v_val_pred_ratio) == 0:
                    r_val = 0.0
                else:
                    r_val = np.corrcoef(v_val_true_ratio, v_val_pred_ratio)[0, 1]
                    
                smape_r = calculate_smape(v_val_true_ratio, v_val_pred_ratio)

                # Log to TensorBoard
                writer.add_scalar('Metrics_Ratio/RMSE', rmse_r, i)
                writer.add_scalar('Metrics_Ratio/MAE', mae_r, i)
                writer.add_scalar('Metrics_Ratio/SMAPE', smape_r, i)
                writer.add_scalar('Metrics_Ratio/Bias', bias_r, i)
                writer.add_scalar('Metrics_Ratio/R', r_val, i)
                writer.add_scalar('Metrics_Ratio/Max_Error', max_err_r, i)

                # Log to Text File
                log_msg = (
                    f"Epoch {i+1:5d} | "
                    f"Loss: {total_loss.item():.12f} (PDE:{pde_loss.item():.12f} Data:{data_loss.item():.12f}) | "
                    f"Val(Ratio): [RMSE:{rmse_r:.4f} MAE:{mae_r:.4f} SMAPE:{smape_r:.2f}% Bias:{bias_r:.4f} R:{r_val:.4f} MaxErr:{max_err_r:.4f}]"
                )
                logging.info(log_msg)
                
        if (i + 1) % 10000 == 0:
            filename = f"checkpoint_epoch_{i+1}.pth"
            torch.save(model.state_dict(), os.path.join(ft_result_dir, filename))
            print(f"Saved checkpoint: {filename}")

            model.train()

    logging.info("--- Fine-Tuning Finished ---")
    writer.close()

    model_save_path = os.path.join(ft_result_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Fine-Tuned Model saved to: {model_save_path}")

if __name__ == "__main__":
    if torch.cuda.is_available(): torch.zeros(1).cuda()
    main()