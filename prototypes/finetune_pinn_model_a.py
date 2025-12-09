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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. CONFIGURATION FOR FINE-TUNING
# ==============================================================================
# ระบุ Path ของ Run เดิมที่จะดึงมาจูน
BASE_RUN_DIR = "runs/train_2025-12-07_08-17-11_DynamicBoundaries" 
MODEL_NAME = "model.pth" # หรือ checkpoint ที่ต้องการ

CHECKPOINT_EPOCHS = 10000 # จำนวนรอบที่ต้องเซฟโมเดลไว้ (สำรองเผื่อไฟดับ)

# ค่าที่อนุญาตให้ปรับ (นอกนั้นใช้ค่าเดิมจากโมเดลแม่)
FT_CONFIG = {
    "epochs": 100000,           # [Adjustable] รอบการจูน
    "lr": 1e-5,                 # [Adjustable] Learning Rate (ควรต่ำกว่าตอนเทรนแม่)
    "n_sample_data": 10000,     # [Adjustable] Batch Size
    "n_sample_pde_multiplier": 4, # [Adjustable] สัดส่วน PDE
    "physics_loss_weight": 1.0, # [Adjustable] ความสำคัญของ Physics
    "val_interval": 1000,       # [Adjustable] ความถี่ในการ Validate
    "n_val_samples": 100000,    # [Adjustable] จำนวนจุด Validate
    "K_step": 1000.0,           # [Adjustable] ความละเอียดของ Strike Price
    "time_sampling_power": 2.0  # [Adjustable] Power Law Factor (เน้นช่วงใกล้หมดอายุ)
}
# ==============================================================================

# --- Utility Functions ---
def analytical_solution(S, K, t, r, sigma):
    """
    Calculate Black-Scholes European Call Option Price.
    """
    # 1. กันค่า Time เป็น 0
    t = np.maximum(t, 1e-10) 
    
    # 2. กัน S และ K เป็น 0
    S = np.maximum(S, 1e-10)
    K = np.maximum(K, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def calculate_smape(true, pred):
    denominator = (np.abs(true) + np.abs(pred)) / 2.0
    diff = np.abs(true - pred)
    return np.mean(diff / (denominator + 1e-8)) * 100

# --- Visualization Functions (Modified: No Pre-training plots) ---
def plot_checkpoint_performance(model, config, save_dir, device):
    """
    Generate 3D Surface and Scatter plots for model validation.
    Saved in the specific checkpoint folder.
    """
    model.eval()
    
    # Extract params
    fix_sig = config["validation_params"]["fix_sigma"]
    fix_r = config["validation_params"]["fix_r"]
    
    # Fix K (Mid range)
    k_min, k_max = config["market"]["K_range"]
    k_step = config["market"]["K_step"]
    fix_K = np.round(((k_min + k_max) / 2) / k_step) * k_step
    
    m_min, m_max = config["sampling"]["moneyness_range"]
    t_min, t_max = config["market"]["t_range"]
    
    # Generate Grid for 3D Plot
    S_plot = np.linspace(fix_K * m_min, fix_K * m_max, 50)
    t_plot = np.linspace(t_min, t_max, 50)
    S_grid, t_grid = np.meshgrid(S_plot, t_plot)
    
    # Prepare Input
    X_flat = np.zeros((S_grid.size, 5))
    X_flat[:, 0] = t_grid.flatten() # t
    X_flat[:, 1] = S_grid.flatten() # S
    X_flat[:, 2] = fix_sig         # sigma
    X_flat[:, 3] = fix_r           # r
    X_flat[:, 4] = fix_K           # K
    
    # Normalization
    c_m = config["market"]
    t_norm = (X_flat[:, 0] - c_m["t_range"][0]) / (c_m["t_range"][1] - c_m["t_range"][0])
    S_norm = (X_flat[:, 1] - c_m["S_range"][0]) / (c_m["S_range"][1] - c_m["S_range"][0])
    sig_norm = (X_flat[:, 2] - c_m["sigma_range"][0]) / (c_m["sigma_range"][1] - c_m["sigma_range"][0])
    r_norm = (X_flat[:, 3] - c_m["r_range"][0]) / (c_m["r_range"][1] - c_m["r_range"][0])
    K_norm = (X_flat[:, 4] - c_m["K_range"][0]) / (c_m["K_range"][1] - c_m["K_range"][0])
    
    X_tensor = torch.tensor(np.stack([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        V_pred_norm = model(X_tensor).cpu().numpy().flatten()
        
    # Scaling back (Output of model is V/K)
    V_pred = V_pred_norm * fix_K
    V_pred_grid = V_pred.reshape(S_grid.shape)
    
    # Analytical Solution
    V_true = analytical_solution(S_grid, fix_K, t_grid, fix_r, fix_sig)
    
    # --- Plot 1: 3D Surface Comparison ---
    fig = plt.figure(figsize=(14, 7))
    param_text = rf"Fixed Parameters: $\sigma={fix_sig}, r={fix_r}, K={fix_K:,.0f}$"
    fig.suptitle(f"3D Surface Comparison\n({param_text})", fontsize=14)
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(t_grid, S_grid, V_true, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title('Analytical Solution', fontsize=12)
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Spot Price (S)')
    ax1.set_zlabel('Option Price (V)')
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(t_grid, S_grid, V_pred_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_title('PINN Prediction', fontsize=12)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Spot Price (S)')
    ax2.set_zlabel('Option Price (V)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(save_dir, "3d_surface_comparison.png"))
    plt.close()
    
    # --- Plot 2: Scatter Comparison ---
    plt.figure(figsize=(8, 8))
    
    v_true_flat = V_true.flatten()
    v_pred_flat = V_pred.flatten()
    
    plt.scatter(v_pred_flat, v_true_flat, alpha=0.5, s=10, label='Prediction Points')
    
    min_val = min(np.min(v_true_flat), np.min(v_pred_flat))
    max_val = max(np.max(v_true_flat), np.max(v_pred_flat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Match (y=x)')
    
    rmse = np.sqrt(np.mean((v_true_flat - v_pred_flat)**2))
    corr = np.corrcoef(v_true_flat, v_pred_flat)[0, 1]
    
    plt.title(f'PINN vs. Analytical Predictions\n(RMSE: {rmse:.4f}, R: {corr:.4f})\n{param_text}', fontsize=12)
    plt.xlabel('PINN Prediction')
    plt.ylabel('Analytical Solution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_comparison.png"))
    plt.close()

def plot_post_training(result_dir, history):
    """
    Generate loss curves AFTER training is complete.
    """
    logging.info("Generating Post-training visualizations...")
    plot_dir = result_dir
    
    if len(history['total']) > 0:
        epochs = range(1, len(history['total']) + 1)
        
        # --- Plot 1: Detailed Curves ---
        fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
        
        def plot_metric(ax, data, color, label, title, y_label):
            ax.plot(epochs, data, color=color, label=label, linewidth=1.0)
            ax.set_ylabel(y_label)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(loc='upper right')
            ax.set_title(title, fontsize=10, pad=2)

        plot_metric(axes[0], history['total'], '#1f77b4', 'Total Loss', 'Total Loss', 'Loss')
        plot_metric(axes[1], history['pde'], '#ff7f0e', 'PDE Loss', 'Physics (PDE) Loss', 'PDE Loss')
        plot_metric(axes[2], history['data'], '#2ca02c', 'Data Loss', 'Total Data Loss (IVP + BVP)', 'Data Loss')
        plot_metric(axes[3], history['ivp'], '#d62728', 'IVP Loss', 'Initial Value Problem (t=0) Loss', 'IVP Loss')
        plot_metric(axes[4], history['bvp1'], '#9467bd', 'BVP1 Loss', 'Lower Boundary Loss', 'BVP1 Loss')
        plot_metric(axes[5], history['bvp2'], '#8c564b', 'BVP2 Loss', 'Upper Boundary Loss', 'BVP2 Loss')

        axes[-1].set_xlabel('Epoch')
        fig.suptitle('Detailed Training Curves (Linear Scale)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(plot_dir, "detailed_training_curves.png"))
        plt.close()
        
        # --- Plot 2: Total Loss Only ---
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['total'], color='#d62728', linewidth=1.5, label='Total Loss')
        plt.title('Total Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "total_loss_curve.png"))
        plt.close()

# --- Main Logic ---

def main():
    # --- 1. Load Original Config (Mother Universe) ---
    config_path = os.path.join(BASE_RUN_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        MOTHER_CONFIG = json.load(f)
    
    # Apply Fine-Tuning Parameters (Override Mother Config)
    # Update Training Params
    for key in ["epochs", "lr", "n_sample_data", "n_sample_pde_multiplier", 
                "physics_loss_weight", "val_interval", "n_val_samples"]:
        MOTHER_CONFIG["training"][key] = FT_CONFIG[key]
        
    # Update Market Params (K_step)
    MOTHER_CONFIG["market"]["K_step"] = FT_CONFIG["K_step"]

    # Update Sampling Params (Power Law Time)
    if "time_sampling_power" in FT_CONFIG:
        MOTHER_CONFIG["sampling"]["time_sampling_power"] = FT_CONFIG["time_sampling_power"]

    # --- 2. Setup Directory (Universal Path Logic) ---
    # Logic: Avoid nested fine_tune folders
    normalized_base_path = BASE_RUN_DIR.replace("\\", "/")
    
    if "/fine_tune" in normalized_base_path:
        root_run_dir = normalized_base_path.split("/fine_tune")[0]
        root_run_dir = os.path.normpath(root_run_dir)
    else:
        root_run_dir = BASE_RUN_DIR

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ft_folder_name = f"ft_{current_time}_Adaptive_PowerLaw"
    
    ft_result_dir = os.path.join(root_run_dir, "fine_tune", ft_folder_name)
    os.makedirs(ft_result_dir, exist_ok=True)

    log_filename = os.path.join(ft_result_dir, "finetune_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    writer = SummaryWriter(log_dir=ft_result_dir)
    
    logging.info(f"--- Started Fine-Tuning (Universal) ---")
    logging.info(f"Source Model: {os.path.join(BASE_RUN_DIR, MODEL_NAME)}")
    logging.info(f"Output Directory: {ft_result_dir}")
    
    with open(os.path.join(ft_result_dir, "config.json"), 'w') as f:
        json.dump(MOTHER_CONFIG, f, indent=4)

    # Extract Global Params
    DEVICE = torch.device(MOTHER_CONFIG["device"])
    c_m = MOTHER_CONFIG["market"]
    c_s = MOTHER_CONFIG["sampling"]
    
    S_min_norm, S_max_norm = c_m["S_range"] 
    K_min, K_max = c_m["K_range"]
    K_step_val = c_m.get("K_step", 1)
    t_min, t_max = c_m["t_range"]
    sig_min, sig_max = c_m["sigma_range"]
    r_min, r_max = c_m["r_range"]
    
    M_min, M_max = c_s["moneyness_range"]
    
    # Calculate Real Sigma (Inherit Logic from Mother)
    ADAPTIVE_STD_FACTOR = c_s.get("adaptive_std", 1.0)
    RANGE_WIDTH = M_max - M_min
    REAL_STD = (RANGE_WIDTH / 6.0) * ADAPTIVE_STD_FACTOR
    
    # Extract Time Power
    TIME_POWER = c_s.get("time_sampling_power", 2.0)
    
    logging.info(f"Sampling Strategy:")
    logging.info(f" - Price: Gaussian (Range[{M_min}, {M_max}], StdFactor={ADAPTIVE_STD_FACTOR})")
    logging.info(f" - Time:  Power Law (Power={TIME_POWER}, Focus near t=0)")

    # --- 3. Normalization Utilities ---
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

    # --- 4. Data Generation Functions (Updated with Power Law) ---
    def get_diff_data(n):
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        
        # Adaptive Moneyness
        moneyness = np.clip(np.random.normal(1.0, REAL_STD, (n, 1)), M_min, M_max)
        S_points = np.clip(K_points * moneyness, S_min_norm, S_max_norm)
        
        # [UPDATED] Power Law Sampling for Time
        u_time = np.random.uniform(0, 1, (n, 1))
        t_points = t_min + (t_max - t_min) * (u_time ** TIME_POWER)
        
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min_norm, S_max_norm)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)
        return np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)

    def get_ivp_data(n):
        # IVP is strictly t=0, no sampling needed
        t_points = np.zeros((n, 1))
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        moneyness = np.clip(np.random.normal(1.0, REAL_STD, (n, 1)), M_min, M_max)
        S_points = np.clip(K_points * moneyness, S_min_norm, S_max_norm)
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))

        t_norm = normalize_val(t_points, t_min, t_max)
        S_norm = normalize_val(S_points, S_min_norm, S_max_norm)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)
        X_norm = np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)
        y_val = np.fmax(S_points - K_points, 0)
        return X_norm, y_val / K_points

    def get_bvp_data(n):
        # [UPDATED] Power Law Sampling for Time
        u_time = np.random.uniform(0, 1, (n, 1))
        t_points = t_min + (t_max - t_min) * (u_time ** TIME_POWER)
        
        sigma_points = np.random.uniform(sig_min, sig_max, (n, 1))
        r_points = np.random.uniform(r_min, r_max, (n, 1))
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        
        t_norm = normalize_val(t_points, t_min, t_max)
        sig_norm = normalize_val(sigma_points, sig_min, sig_max)
        r_norm = normalize_val(r_points, r_min, r_max)
        K_norm = normalize_val(K_points, K_min, K_max)

        # Lower Boundary
        S1_points = np.clip(K_points * M_min, S_min_norm, S_max_norm)
        S1_norm = normalize_val(S1_points, S_min_norm, S_max_norm)
        X1_norm = np.concatenate([t_norm, S1_norm, sig_norm, r_norm, K_norm], axis=1)
        y1_val = np.zeros((n, 1))

        # Upper Boundary
        S2_points = np.clip(K_points * M_max, S_min_norm, S_max_norm)
        S2_norm = normalize_val(S2_points, S_min_norm, S_max_norm)
        X2_norm = np.concatenate([t_norm, S2_norm, sig_norm, r_norm, K_norm], axis=1)
        y2_val = np.maximum(S2_points - K_points * np.exp(-r_points * t_points), 0)

        return X1_norm, y1_val / K_points, X2_norm, y2_val / K_points

    # --- 5. Model Definition ---
    class UniversalPINN(nn.Module):
        def __init__(self, n_input, n_output, n_hidden, n_layers):
            super().__init__()
            activation = nn.Tanh()
            layers = [nn.Linear(n_input, n_hidden), activation]
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(activation)
            layers.append(nn.Linear(n_hidden, n_output))
            layers.append(nn.Softplus())
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    # --- 6. Load Pre-trained Model ---
    N_INPUT = MOTHER_CONFIG["model"]["n_input"]
    
    model = UniversalPINN(N_INPUT, 1, MOTHER_CONFIG["model"]["n_hidden"], MOTHER_CONFIG["model"]["n_layers"]).to(DEVICE)
    
    model_path = os.path.join(BASE_RUN_DIR, MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    logging.info("Loaded Pre-trained Weights successfully.")

    optimizer = torch.optim.Adam(model.parameters(), lr=FT_CONFIG["lr"])
    loss_fn = nn.MSELoss()

    # --- 7. Validation Set ---
    # Generate new validation set based on K_step configuration
    N_VAL = FT_CONFIG["n_val_samples"]
    X_val_norm = get_diff_data(N_VAL)
    
    t_val = denormalize_val(X_val_norm[:, 0], t_min, t_max)
    S_val = denormalize_val(X_val_norm[:, 1], S_min_norm, S_max_norm)
    sig_val = denormalize_val(X_val_norm[:, 2], sig_min, sig_max)
    r_val = denormalize_val(X_val_norm[:, 3], r_min, r_max)
    K_val = denormalize_val(X_val_norm[:, 4], K_min, K_max)
    V_val_true = analytical_solution(S_val, K_val, t_val, r_val, sig_val)
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(DEVICE)
    
    history = {'total': [], 'pde': [], 'data': [], 'ivp': [], 'bvp1': [], 'bvp2': []}

    # --- 8. Fine-Tuning Loop ---
    logging.info("\n--- Starting Fine-Tuning Loop ---")
    
    EPOCHS = FT_CONFIG["epochs"]
    N_SAMPLE_DATA = FT_CONFIG["n_sample_data"]
    N_SAMPLE_PDE = N_SAMPLE_DATA * FT_CONFIG["n_sample_pde_multiplier"]
    PHYSICS_WEIGHT = FT_CONFIG["physics_loss_weight"]
    VAL_INTERVAL = FT_CONFIG["val_interval"]

    for i in tqdm(range(EPOCHS), desc="Fine-Tuning", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        
        # Losses
        ivp_x, ivp_y = get_ivp_data(N_SAMPLE_DATA)
        ivp_pred = model(torch.from_numpy(ivp_x).float().to(DEVICE))
        loss_ivp = loss_fn(ivp_pred, torch.from_numpy(ivp_y).float().to(DEVICE))

        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = get_bvp_data(N_SAMPLE_DATA)
        pred_bvp1 = model(torch.from_numpy(bvp_x1).float().to(DEVICE))
        pred_bvp2 = model(torch.from_numpy(bvp_x2).float().to(DEVICE))
        loss_bvp1 = loss_fn(pred_bvp1, torch.from_numpy(bvp_y1).float().to(DEVICE))
        loss_bvp2 = loss_fn(pred_bvp2, torch.from_numpy(bvp_y2).float().to(DEVICE))
        loss_bvp_total = loss_bvp1 + loss_bvp2
        
        X_pde_norm = get_diff_data(N_SAMPLE_PDE)
        X_pde_tensor = torch.from_numpy(X_pde_norm).float().to(DEVICE).requires_grad_()
        v_pred_norm = model(X_pde_tensor)
        
        # PDE logic
        S_pde = denormalize_val(X_pde_tensor[:, 1:2], S_min_norm, S_max_norm)
        sigma_pde = denormalize_val(X_pde_tensor[:, 2:3], sig_min, sig_max)
        r_pde = denormalize_val(X_pde_tensor[:, 3:4], r_min, r_max)
        K_pde = denormalize_val(X_pde_tensor[:, 4:5], K_min, K_max)
        V_real = v_pred_norm * K_pde
        
        grads = torch.autograd.grad(v_pred_norm, X_pde_tensor, grad_outputs=torch.ones_like(v_pred_norm), create_graph=True)[0]
        dv_dt_n, dv_ds_n = grads[:, 0:1], grads[:, 1:2]
        grads2 = torch.autograd.grad(dv_ds_n, X_pde_tensor, grad_outputs=torch.ones_like(dv_ds_n), create_graph=True)[0]
        d2v_ds2_n = grads2[:, 1:2]
        
        dV_dt = (K_pde / (t_max - t_min)) * dv_dt_n
        dV_dS = (K_pde / (S_max_norm - S_min_norm)) * dv_ds_n
        d2V_dS2 = (K_pde / (S_max_norm - S_min_norm)**2) * d2v_ds2_n
        
        pde_res = dV_dt - (0.5 * sigma_pde**2 * S_pde**2 * d2V_dS2 + r_pde * S_pde * dV_dS - r_pde * V_real)
        pde_loss = PHYSICS_WEIGHT * loss_fn(pde_res / K_pde, torch.zeros_like(pde_res))
        
        data_loss = loss_ivp + loss_bvp_total
        total_loss = data_loss + pde_loss
        total_loss.backward()
        optimizer.step()

        # Record History
        history['total'].append(total_loss.item())
        history['pde'].append(pde_loss.item())
        history['data'].append(data_loss.item())
        history['ivp'].append(loss_ivp.item())
        history['bvp1'].append(loss_bvp1.item())
        history['bvp2'].append(loss_bvp2.item())

        if i % 10 == 0:
            writer.add_scalar('Loss/Total', total_loss.item(), i)
            writer.add_scalar('Loss/PDE', pde_loss.item(), i)
            writer.add_scalar('Loss/Data_Total', data_loss.item(), i)
            writer.add_scalar('Loss_Detail/IVP', loss_ivp.item(), i)
            writer.add_scalar('Loss_Detail/BVP_Total', loss_bvp_total.item(), i)
            writer.add_scalar('Loss_Detail/BVP1_Min', loss_bvp1.item(), i)
            writer.add_scalar('Loss_Detail/BVP2_Max', loss_bvp2.item(), i)

        # Validation Log
        if (i + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                v_val_pred_ratio = model(X_val_tensor).cpu().numpy().flatten()
                v_val_true_ratio = V_val_true.flatten() / K_val.flatten()
                diff_ratio = v_val_pred_ratio - v_val_true_ratio
                rmse_r = np.sqrt(np.mean(diff_ratio**2))
                smape_r = calculate_smape(v_val_true_ratio, v_val_pred_ratio)
                
                writer.add_scalar('Metrics_Ratio/RMSE', rmse_r, i)
                writer.add_scalar('Metrics_Ratio/SMAPE', smape_r, i)
                
                log_msg = (
                    f"Epoch {i+1:5d} | "
                    f"Loss: {total_loss.item():.12f} (PDE:{pde_loss.item():.12f} Data:{data_loss.item():.12f}) | "
                    f"Val(Ratio): [RMSE:{rmse_r:.4f} SMAPE:{smape_r:.2f}%]"
                )
                logging.info(log_msg)
        
        # --- Checkpoint & Plots ---
        if (i + 1) % CHECKPOINT_EPOCHS == 0:
            ckpt_dir = os.path.join(ft_result_dir, "checkpoints", f"epoch_{i+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pth"))
            plot_checkpoint_performance(model, MOTHER_CONFIG, ckpt_dir, DEVICE)
            
            print(f"Saved checkpoint and plots to: {ckpt_dir}")
            model.train()

    logging.info("--- Fine-Tuning Finished ---")
    writer.close()

    # --- 9. Final Save ---
    torch.save(model.state_dict(), os.path.join(ft_result_dir, "model.pth"))
    
    logging.info("Generating Final Performance Plots...")
    plot_checkpoint_performance(model, MOTHER_CONFIG, ft_result_dir, DEVICE)
    
    # --- Post-Training Plots ---
    plot_post_training(ft_result_dir, history)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        print("GPU Warmed up and ready!")
    main()