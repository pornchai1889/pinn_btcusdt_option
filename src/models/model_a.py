import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import logging
import json # เพิ่มเพื่อ save config
from datetime import datetime

def main():
    # --- 1. Dynamic Directory Setup (Experiment Tracking) ---
    # ใช้ Timestamp เพื่อแยกแต่ละการทดลองออกจากกัน
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"train_{current_time}_Universal5Inputs" # ตั้งชื่อให้สื่อความหมาย
    base_output_dir = "runs"
    result_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(result_dir, exist_ok=True)

    # --- 1.5 Setup Logging & TensorBoard ---
    # ไฟล์ log จะอยู่ในโฟลเดอร์ experiment ของมันเอง ไม่ปนกับคนอื่น
    log_filename = os.path.join(result_dir, "training_log.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    
    # TensorBoard จะอ่านข้อมูลจากโฟลเดอร์นี้
    writer = SummaryWriter(log_dir=result_dir)
    
    logging.info(f"--- Started Experiment: {run_name} ---")
    logging.info(f"--- Artifacts (Logs, Models) saved to: {result_dir} ---")

    # --- 2. Configuration Dictionary (Centralized Config) ---
    # เก็บค่าคงที่ทั้งหมดไว้ใน Dict เพื่อให้ Save ง่ายๆ
    CONFIG = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "market": {
            "T_MAX": 1.0,
            "S_range": [0.0, 1000000.0],
            "K_range": [10000.0, 200000.0],
            "t_range": [0.0, 1.0],
            "sigma_range": [0.1, 2.0],
            "r_range": [0.0, 0.15]
        },
        "model": {
            "n_input": 5,
            "n_output": 1,
            "n_hidden": 128,
            "n_layers": 8
        },
        "training": {
            "epochs": 80000,
            "lr": 1e-4,
            "n_sample_data": 8000,        # สำหรับ IVP/BVP
            "n_sample_pde_multiplier": 4, # ตัวคูณสำหรับ PDE (ตามที่คุณขอ)
            "physics_loss_weight": 1.0
        }
    }
    
    DEVICE = torch.device(CONFIG["device"])
    logging.info(f"Using device: {DEVICE}")

    # *** สำคัญ: บันทึก Config ลงไฟล์ JSON ไว้ดูย้อนหลัง ***
    config_save_path = os.path.join(result_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    logging.info(f"Configuration saved to: {config_save_path}")

    # ดึงค่าจาก Config มาใช้ (เพื่อความสั้นในการเขียนโค้ดต่อจากนี้)
    c_market = CONFIG["market"]
    S_min, S_max = c_market["S_range"]
    K_min, K_max = c_market["K_range"]
    t_min, t_max = c_market["t_range"]
    sig_min, sig_max = c_market["sigma_range"]
    r_min, r_max = c_market["r_range"]

    # --- Normalization Utilities ---
    def normalize_val(val, v_min, v_max):
        return (val - v_min) / (v_max - v_min)

    def denormalize_val(val_norm, v_min, v_max):
        return val_norm * (v_max - v_min) + v_min

    # --- Data Generation Functions ---
    def get_diff_data(n):
        # 1. Random K
        K_points = np.random.uniform(K_min, K_max, (n, 1))

        # 2. Relational Sampling for S
        n_focus = int(n * 0.8)
        n_wide = n - n_focus
        
        moneyness = np.random.uniform(0.5, 1.5, (n_focus, 1)) 
        S_focus = K_points[:n_focus] * moneyness
        S_wide = np.random.uniform(S_min, S_max, (n_wide, 1))
        
        S_points = np.concatenate([S_focus, S_wide], axis=0)
        S_points = np.clip(S_points, S_min, S_max)

        # 3. Random others
        t_points = np.random.uniform(t_min, t_max, (n, 1))
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        # 4. Normalize
        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min, S_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        return np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)

    def get_ivp_data(n):
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

        # Lower Bound
        S1_norm = normalize_val(S_min * np.ones((n, 1)), S_min, S_max)
        X1_norm = np.concatenate([t_norm, S1_norm, sig_norm, r_norm, K_norm], axis=1)
        y1_val = np.zeros((n, 1))

        # Upper Bound
        S2_points = S_max * np.ones((n, 1))
        S2_norm = normalize_val(S2_points, S_min, S_max)
        X2_norm = np.concatenate([t_norm, S2_norm, sig_norm, r_norm, K_norm], axis=1)
        y2_val = (S2_points - K_points * np.exp(-r_points * t_points)).reshape(-1, 1)

        return X1_norm, y1_val / K_points, X2_norm, y2_val / K_points

    # --- 4. PINN Model Definition ---
    class UniversalPINN(nn.Module):
        def __init__(self, n_input, n_output, n_hidden, n_layers):
            super().__init__()
            activation = nn.Tanh()
            layers = [nn.Linear(n_input, n_hidden), activation]
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(activation)
            layers.append(nn.Linear(n_hidden, n_output))
            
            # Initialization
            for m in self.net.modules() if hasattr(self, 'net') else self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # --- 5. Training Setup ---
    # ดึงค่าจาก Config
    N_INPUT = CONFIG["model"]["n_input"]
    N_OUTPUT = CONFIG["model"]["n_output"]
    N_HIDDEN = CONFIG["model"]["n_hidden"]
    N_LAYERS = CONFIG["model"]["n_layers"]
    
    EPOCHS = CONFIG["training"]["epochs"]
    LR = CONFIG["training"]["lr"]
    
    # N_SAMPLE สำหรับ Data Loss
    N_SAMPLE_DATA = CONFIG["training"]["n_sample_data"]
    # N_SAMPLE สำหรับ PDE Loss (ใช้ตัวคูณ)
    N_SAMPLE_PDE = N_SAMPLE_DATA * CONFIG["training"]["n_sample_pde_multiplier"]
    
    PHYSICS_WEIGHT = CONFIG["training"]["physics_loss_weight"]

    logging.info(f"Model: {N_INPUT} inputs, {N_LAYERS} layers")
    logging.info(f"Data Samples: {N_SAMPLE_DATA}, PDE Samples: {N_SAMPLE_PDE}")
    
    model = UniversalPINN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- 6. Training Loop ---
    logging.info("\n--- Starting PINN Training ---")
    
    for i in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Data Loss (IVP + BVP)
        ivp_x, ivp_y = get_ivp_data(N_SAMPLE_DATA)
        ivp_pred = model(torch.from_numpy(ivp_x).float().to(DEVICE))
        mse_ivp = loss_fn(ivp_pred, torch.from_numpy(ivp_y).float().to(DEVICE))

        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = get_bvp_data(N_SAMPLE_DATA)
        pred_bvp1 = model(torch.from_numpy(bvp_x1).float().to(DEVICE))
        pred_bvp2 = model(torch.from_numpy(bvp_x2).float().to(DEVICE))
        mse_bvp = loss_fn(pred_bvp1, torch.from_numpy(bvp_y1).float().to(DEVICE)) + \
                  loss_fn(pred_bvp2, torch.from_numpy(bvp_y2).float().to(DEVICE))
        
        data_loss = mse_ivp + mse_bvp

        # 2. Physics Loss (PDE) - ใช้ N_SAMPLE_PDE ที่เยอะกว่า
        X_pde_norm = get_diff_data(N_SAMPLE_PDE)
        X_pde_tensor = torch.from_numpy(X_pde_norm).float().to(DEVICE).requires_grad_()
        
        v_pred_norm = model(X_pde_tensor)

        # Denormalize Inputs
        # col 0=t, 1=S, 2=sigma, 3=r, 4=K
        S_pde = denormalize_val(X_pde_tensor[:, 1:2], S_min, S_max)
        sigma_pde = denormalize_val(X_pde_tensor[:, 2:3], sig_min, sig_max)
        r_pde = denormalize_val(X_pde_tensor[:, 3:4], r_min, r_max)
        K_pde = denormalize_val(X_pde_tensor[:, 4:5], K_min, K_max)

        # Output V = v_norm * K
        V_real = v_pred_norm * K_pde

        # Gradients
        grads = torch.autograd.grad(v_pred_norm, X_pde_tensor, grad_outputs=torch.ones_like(v_pred_norm), create_graph=True)[0]
        dv_dt_n, dv_ds_n = grads[:, 0:1], grads[:, 1:2]
        
        grads2 = torch.autograd.grad(dv_ds_n, X_pde_tensor, grad_outputs=torch.ones_like(dv_ds_n), create_graph=True)[0]
        d2v_ds2_n = grads2[:, 1:2]

        # Chain Rule Scaling
        dV_dt = (K_pde / (t_max - t_min)) * dv_dt_n
        dV_dS = (K_pde / (S_max - S_min)) * dv_ds_n
        d2V_dS2 = (K_pde / (S_max - S_min)**2) * d2v_ds2_n

        # PDE Residual
        pde_residual = dV_dt - (0.5 * sigma_pde**2 * S_pde**2 * d2V_dS2 + r_pde * S_pde * dV_dS - r_pde * V_real)
        
        pde_loss = PHYSICS_WEIGHT * loss_fn(pde_residual / K_pde, torch.zeros_like(pde_residual))

        # Total Loss
        loss = data_loss + pde_loss
        loss.backward()
        optimizer.step()

        # Logging & TensorBoard
        writer.add_scalar('Loss/Total', loss.item(), i)
        writer.add_scalar('Loss/Data', data_loss.item(), i)
        writer.add_scalar('Loss/PDE', pde_loss.item(), i)

        if (i + 1) % 1000 == 0:
            logging.info(f"Epoch {i+1}/{EPOCHS} | Loss: {loss.item():.6f} (Data: {data_loss.item():.6f}, PDE: {pde_loss.item():.6f})")

    logging.info("--- Training Finished ---")
    writer.close()

    # --- 7. Save Model (ในโฟลเดอร์เดียวกับ Log) ---
    # ใช้ชื่อกลางๆ เช่น 'final_model.pth' เพราะโฟลเดอร์บอก timestamp อยู่แล้ว
    model_save_path = os.path.join(result_dir, "final_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()