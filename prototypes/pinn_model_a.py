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
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

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
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    Range: 0-100% (Stable for small values)
    """
    denominator = (np.abs(true) + np.abs(pred)) / 2.0
    diff = np.abs(true - pred)
    # Avoid division by zero
    return np.mean(diff / (denominator + 1e-8)) * 100

def sample_moneyness_mixed(n, m_min, m_max, std_val):
    """
    [NEW] Mixed Distribution Strategy:
    1. Sample from Gaussian (Normal Distribution).
    2. Check for outliers (values outside [m_min, m_max]).
    3. Re-sample outliers using Uniform Distribution (filling the gaps).
    """
    # 1. Generate Raw Gaussian
    data = np.random.normal(1.0, std_val, (n, 1))
    
    # 2. Identify Outliers
    # flatten() is used for boolean indexing safety, then reshape back
    flat_data = data.flatten()
    outliers_mask = (flat_data < m_min) | (flat_data > m_max)
    n_out = np.sum(outliers_mask)
    
    # 3. Resample Outliers (Uniformly)
    if n_out > 0:
        flat_data[outliers_mask] = np.random.uniform(m_min, m_max, n_out)
    
    return flat_data.reshape(n, 1)

# --- [MODIFIED] Visualization Functions ---
def plot_pre_training(result_dir, config):
    """
    Generate plots that visualize the data distribution BEFORE training starts.
    Saved in the root result directory.
    """
    logging.info("Generating Pre-training visualizations...")
    
    # Extract Configs
    user_std_factor = config["sampling"]["adaptive_std"]
    time_power = config["sampling"].get("time_sampling_power", 2.0) # Power Law Factor
    m_min, m_max = config["sampling"]["moneyness_range"]
    t_min, t_max = config["market"]["t_range"]
    
# --- Dynamic Sigma Calculation ---
    range_width = m_max - m_min
    std_val = (range_width / 6.0) * user_std_factor
    
    # --- Plot 1: Moneyness Density & Zones (UPDATED) ---
    plt.figure(figsize=(12, 7))
    
    mu = 1.0
    x = np.linspace(m_min, m_max, 1000)
    y = norm.pdf(x, mu, std_val)
    max_y = np.max(y)
    
    # 1. Calculate Stats (In-Bound vs Outliers)
    prob_below = norm.cdf(m_min, mu, std_val)
    prob_above = 1.0 - norm.cdf(m_max, mu, std_val)
    total_prob_outliers = prob_below + prob_above
    total_prob_gaussian_in = 1.0 - total_prob_outliers
    
    # Calculate "Water Level" (Uniform density from recycled outliers)
    # Density = Probability / Width
    water_level = total_prob_outliers / range_width

    # 2. Plot Main Gaussian Curve
    plt.plot(x, y, color='#333333', linewidth=2.5, label=rf'Base Gaussian ($\sigma$={std_val:.4f})', zorder=10)
    
    # 3. Zones Visualization (Restored & Active)
    zones = [
        (0, 1, '#2ca02c', 0.45),   # Green (Core)
        (1, 2, '#ff7f0e', 0.15),   # Orange
        (2, 3, '#d62728', 0.04),   # Red
        (3, 10, '#999999', 0.0)    # Grey (Tail - visually minimal inside range)
    ]
    
    zone_legend_handles = []
    
    # Plot Water Level Line (Blue Dashed) - "ระดับน้ำ"
    plt.hlines(water_level, m_min, m_max, colors='#0055A4', linestyles='-.', linewidth=2.0, zorder=15)
    
    # Loop to draw zones
    for start_sd, end_sd, color, h_ratio in zones:
        # Calculate prob for label
        prob_one_side = norm.cdf(end_sd) - norm.cdf(start_sd)
        pct_band_total = prob_one_side * 2 * 100
        
        label_txt = rf"{start_sd}-{end_sd}$\sigma$: {pct_band_total:.2f}%"
        if end_sd > 3: label_txt = rf">{start_sd}$\sigma$ (Tails)"
        
        # Add to legend list
        from matplotlib.patches import Patch
        zone_legend_handles.append(Patch(facecolor=color, edgecolor='none', label=label_txt))

        # Fill Areas (Right Side)
        right_start = mu + start_sd * std_val
        right_end = mu + end_sd * std_val
        plot_start = max(right_start, m_min)
        plot_end = min(right_end, m_max)
        
        if plot_start < plot_end:
            x_fill = np.linspace(plot_start, plot_end, 200)
            y_fill = norm.pdf(x_fill, mu, std_val)
            plt.fill_between(x_fill, y_fill, color=color, alpha=0.6)
            # Text Label
            if end_sd <= 3:
                text_x = (plot_start + plot_end) / 2
                text_y = norm.pdf(text_x, mu, std_val) * h_ratio
                if m_min < text_x < m_max:
                    plt.text(text_x, text_y, f"{prob_one_side*100:.1f}%", 
                             color='white' if start_sd < 1 else 'black', 
                             ha='center', va='center', fontsize=9, fontweight='bold')

        # Fill Areas (Left Side)
        left_start = mu - start_sd * std_val
        left_end = mu - end_sd * std_val
        plot_start = max(left_end, m_min)
        plot_end = min(left_start, m_max)
        
        if plot_start < plot_end:
            x_fill = np.linspace(plot_start, plot_end, 200)
            y_fill = norm.pdf(x_fill, mu, std_val)
            plt.fill_between(x_fill, y_fill, color=color, alpha=0.6)
            # Text Label
            if end_sd <= 3:
                text_x = (plot_start + plot_end) / 2
                text_y = norm.pdf(text_x, mu, std_val) * h_ratio
                if m_min < text_x < m_max:
                    plt.text(text_x, text_y, f"{prob_one_side*100:.1f}%", 
                             color='white' if start_sd < 1 else 'black', 
                             ha='center', va='center', fontsize=9, fontweight='bold')

    # 4. Axis Labels & Markers
    label_y_pos = -max_y * 0.04
    for i in range(1, 4):
        sd_pos_r = mu + i*std_val
        if m_min < sd_pos_r < m_max:
            plt.axvline(sd_pos_r, color='grey', linestyle=':', alpha=0.5, linewidth=1)
            plt.text(sd_pos_r, label_y_pos, rf"+{i}$\sigma$", ha='center', color='#333333', fontsize=9)
        sd_pos_l = mu - i*std_val
        if m_min < sd_pos_l < m_max:
            plt.axvline(sd_pos_l, color='grey', linestyle=':', alpha=0.5, linewidth=1)
            plt.text(sd_pos_l, label_y_pos, rf"-{i}$\sigma$", ha='center', color='#333333', fontsize=9)

    plt.axvline(mu, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.text(mu, label_y_pos, rf"$\mu$", ha='center', color='black', fontsize=10, fontweight='bold')
    
    # 5. Build Custom Legend
    from matplotlib.lines import Line2D
    
    # Add Divider and Summary Stats to Legend
    separator = Line2D([0], [0], color='white', label='__________________')
    gaussian_summary = Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333', 
                              label=f'In-Bound Gaussian: {total_prob_gaussian_in*100:.2f}%')
    water_summary = Line2D([0], [0], color='#0055A4', linestyle='-.', linewidth=2,
                           label=f'Recycled Tails (Water): {total_prob_outliers*100:.2f}%')
    
    # Combine handles
    final_handles = zone_legend_handles[:3] + [separator, gaussian_summary, water_summary]
    
    plt.legend(handles=final_handles, loc='upper right', framealpha=0.95, title="Data Distribution Stats")
    
    title_text = f'Moneyness Distribution: Gaussian Zones + Recycled Outliers (Water Level)\n(Adaptive Factor={user_std_factor}, Range Width={range_width:.2f})'
    plt.title(title_text, fontsize=14, pad=20)
    plt.xlabel('Moneyness (S/K)')
    plt.ylabel('Probability Density')
    
    # Adjust Y limit to see water level clearly if it's low, or fit the mountain if it's high
    plt.ylim(bottom=-max_y*0.08, top=max_y * 1.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "moneyness_density_mixed.png"))
    plt.close()
    # --- Plot 2: Data Sampling Distribution (Unchanged) ---
    fix_K_mid = (config["market"]["K_range"][0] + config["market"]["K_range"][1]) / 2
    fix_K = np.round(fix_K_mid / config["market"]["K_step"]) * config["market"]["K_step"]
    
    n_data = config["training"]["n_sample_data"]
    n_pde = n_data * config["training"]["n_sample_pde_multiplier"]
    total_points = n_pde + n_data + (n_data*2)

    plt.figure(figsize=(12, 8))
    
    # 1. PDE (Grey) - Using Mixed Sampling
    u_pde = np.random.uniform(0, 1, n_pde)
    t_pde = t_min + (t_max - t_min) * (u_pde ** time_power) 
    m_pde = sample_moneyness_mixed(n_pde, m_min, m_max, std_val).flatten()
    S_pde = fix_K * m_pde
    
    plt.scatter(t_pde, S_pde, c='#cccccc', s=10, alpha=0.4, label='PDE Collocation Points')
    
    # 2. IVP (Blue)
    t_ivp = np.zeros(n_data)
    m_ivp = sample_moneyness_mixed(n_data, m_min, m_max, std_val).flatten()
    S_ivp = fix_K * m_ivp
    plt.scatter(t_ivp, S_ivp, c='blue', s=20, alpha=0.6, label='IVP (t=0)')

    # 3. BVP Upper (Green)
    u_bvp2 = np.random.uniform(0, 1, n_data)
    t_bvp2 = t_min + (t_max - t_min) * (u_bvp2 ** time_power)
    S_bvp2 = np.full(n_data, fix_K * m_max)
    plt.scatter(t_bvp2, S_bvp2, c='green', marker='x', s=25, alpha=0.6, label=f'BVP Upper (S={fix_K * m_max:.0f})')

    # 4. BVP Lower (Red)
    u_bvp1 = np.random.uniform(0, 1, n_data)
    t_bvp1 = t_min + (t_max - t_min) * (u_bvp1 ** time_power)
    S_bvp1 = np.full(n_data, fix_K * m_min)
    plt.scatter(t_bvp1, S_bvp1, c='red', marker='x', s=25, alpha=0.6, label=f'BVP Lower (S={fix_K * m_min:.0f})')

    plt.axhline(fix_K, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Strike K={fix_K:,.0f}')
    
    plt.title(rf'Data Sampling Distribution (Mixed Moneyness + Power Law Time)', fontsize=14)
    plt.xlabel('Time to Maturity (t)')
    plt.ylabel('Spot Price (S)')
    
    info_text = (f"Point Counts (Total: {total_points:,}):\n"
                 f"PDE: {n_pde:,}\n"
                 f"IVP: {n_data:,}\n"
                 f"BVP: {n_data*2:,}\n"
                 f"Time Strategy: Power Law ($p={time_power}$)\n"
                 f"Moneyness: Mixed (Normal + Uniform)")
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.9))

    plt.legend(loc='center right', framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "data_sampling_distribution.png"))
    plt.close()


def plot_checkpoint_performance(model, config, save_dir, device):
    """
    Generate 3D Surface and Scatter plots for model validation.
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
    S_plot = np.linspace(fix_K * m_min, fix_K * m_max, 100)
    t_plot = np.linspace(t_min, t_max, 100)
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
    Saved in the root result directory.
    """
    logging.info("Generating Post-training visualizations...")
    plot_dir = result_dir 
    
    if len(history['total']) > 0:
        epochs = range(1, len(history['total']) + 1)
        
        # --- Plot 1: Detailed Curves (Linear Scale) ---
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
        
        # --- Plot 2: Total Loss Only (Standalone) ---
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
    # --- 2. Setup Directory & Logging ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"train_{current_time}_DynamicBoundaries_MixedDist"
    
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
            "T_MAX": 0.25,            
            "S_range": [0.0, 1000000.0],
            "K_range": [10000.0, 500000.0],
            "K_step": 1000.0,
            "t_range": [0.0, 0.25],
            "sigma_range": [0.1, 2.0],
            "r_range": [0.0, 0.15]
        },
        "sampling": {
            "moneyness_range": [0.5, 1.5],
            "adaptive_std": 1.0,
            "time_sampling_power": 2.0 
        },
        "validation_params": {
            "fix_sigma": 0.5,
            "fix_r": 0.05
        },
        "model": {
            "n_input": 5, "n_output": 1, "n_hidden": 256, "n_layers": 4
        },
        "training": {
            "epochs": 600000,
            "lr": 1e-4,
            "n_sample_data": 10000,
            "n_sample_pde_multiplier": 4,
            "physics_loss_weight": 1.0,
            "val_interval": 1000,
            "n_val_sample": 100000,
            "checkpoint_epochs": 10000
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
    
    S_min_norm, S_max_norm = c_m["S_range"] 
    K_min, K_max = c_m["K_range"]
    K_step_val = c_m.get("K_step", 1)
    t_min, t_max = c_m["t_range"]
    sig_min, sig_max = c_m["sigma_range"]
    r_min, r_max = c_m["r_range"]
    
    M_min, M_max = c_s["moneyness_range"]
    TIME_POWER = c_s.get("time_sampling_power", 2.0)
    
    # --- [CALCULATE REAL SIGMA] ---
    ADAPTIVE_STD_FACTOR = c_s.get("adaptive_std", 1.0)
    RANGE_WIDTH = M_max - M_min
    REAL_STD = (RANGE_WIDTH / 6.0) * ADAPTIVE_STD_FACTOR
    
    logging.info(f"Moneyness Range: [{M_min}, {M_max}] (Width: {RANGE_WIDTH})")
    logging.info(f"Time Sampling: Power Law (Power={TIME_POWER}, Focus near t=0)")
    logging.info(f"Adaptive Factor: {ADAPTIVE_STD_FACTOR} => Calculated Real Sigma: {REAL_STD:.6f}")
    logging.info("Sampling Strategy: Gaussian + Uniform Resampling for Outliers (Mixed Distribution)")

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
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        
        # [MODIFIED] Use Mixed Distribution
        moneyness = sample_moneyness_mixed(n, M_min, M_max, REAL_STD)
        
        S_points = np.clip(K_points * moneyness, S_min_norm, S_max_norm)
        
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
        # IVP is strictly t=0
        t_points = np.zeros((n, 1))
        K_points = get_discrete_K(n, K_min, K_max, K_step_val)
        
        # [MODIFIED] Use Mixed Distribution
        moneyness = sample_moneyness_mixed(n, M_min, M_max, REAL_STD)
        
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
            layers.append(nn.Softplus())
            
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

    # --- 7.5 Validation Set ---
    n_val = CONFIG["training"]["n_val_sample"]
    # Generate validation data (Mixed distribution applied here too for fairness, or use pure uniform for strict test)
    # Using the same generator as training ensures distributions match.
    X_val_norm = get_diff_data(n_val)
    t_val = denormalize_val(X_val_norm[:, 0], t_min, t_max)
    S_val = denormalize_val(X_val_norm[:, 1], S_min_norm, S_max_norm)
    sig_val = denormalize_val(X_val_norm[:, 2], sig_min, sig_max)
    r_val = denormalize_val(X_val_norm[:, 3], r_min, r_max)
    K_val = denormalize_val(X_val_norm[:, 4], K_min, K_max)
    V_val_true = analytical_solution(S_val, K_val, t_val, r_val, sig_val)
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(DEVICE)
    
    # History
    history = {'total': [], 'pde': [], 'data': [], 'ivp': [], 'bvp1': [], 'bvp2': []}

    # --- [STEP 1] GENERATE PRE-TRAINING PLOTS ---
    plot_pre_training(result_dir, CONFIG)

    # --- 8. Training Loop ---
    logging.info("\n--- Starting Training ---")
    
    try:
        for i in tqdm(range(EPOCHS), desc="Training PINN", unit="epoch"):
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
            if (i + 1) % CONFIG["training"]["val_interval"] == 0:
                model.eval()
                with torch.no_grad():
                    v_val_pred_ratio = model(X_val_tensor).cpu().numpy().flatten()
                    v_val_true_ratio = V_val_true.flatten() / K_val.flatten()
                    
                    # Metrics Calculation
                    diff_ratio = v_val_pred_ratio - v_val_true_ratio
                    abs_diff = np.abs(diff_ratio)
                    
                    rmse_r = np.sqrt(np.mean(diff_ratio**2))
                    mae_r = np.mean(abs_diff)
                    bias_r = np.mean(diff_ratio)
                    max_err_r = np.max(abs_diff)
                    smape_r = calculate_smape(v_val_true_ratio, v_val_pred_ratio)
                    
                    # R (Correlation Coefficient)
                    corr_matrix = np.corrcoef(v_val_true_ratio, v_val_pred_ratio)
                    r_score = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    
                    # Log to TensorBoard
                    writer.add_scalar('Metrics_Ratio/RMSE', rmse_r, i)
                    writer.add_scalar('Metrics_Ratio/MAE', mae_r, i)
                    writer.add_scalar('Metrics_Ratio/SMAPE', smape_r, i)
                    writer.add_scalar('Metrics_Ratio/Bias', bias_r, i)
                    writer.add_scalar('Metrics_Ratio/R', r_score, i)
                    writer.add_scalar('Metrics_Ratio/Max_Error', max_err_r, i)
                    
                    # Log Message
                    log_msg = (
                        f"Epoch {i+1:5d} | "
                        f"Loss: {total_loss.item():.12f} (PDE:{pde_loss.item():.12f} Data:{data_loss.item():.12f}) | "
                        f"Val(Ratio): [RMSE:{rmse_r:.4f} MAE:{mae_r:.4f} SMAPE:{smape_r:.2f}% "
                        f"Bias:{bias_r:.4f} R:{r_score:.4f} Max_Err:{max_err_r:.4f}]"
                    )
                    logging.info(log_msg)
            
            # --- [STEP 2] CHECKPOINT & PLOTS ---
            if (i + 1) % CONFIG["training"]["checkpoint_epochs"] == 0:
                ckpt_dir = os.path.join(result_dir, "checkpoints", f"epoch_{i+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pth"))
                plot_checkpoint_performance(model, CONFIG, ckpt_dir, DEVICE)
                
                print(f"Saved checkpoint and plots to: {ckpt_dir}")
                model.train()

    except KeyboardInterrupt:
        logging.warning("\n!!! Training Interrupted by User (Ctrl+C) !!!")
        logging.info("Attempting to save current model state before exiting...")
    
    except Exception as e:
        logging.error(f"\n!!! An unexpected error occurred: {e} !!!")
        raise e
        
    finally:
        # --- 9. Final Save (Executed on Finish, Interrupt, or Error) ---
        logging.info("--- Saving Final Artifacts ---")
        writer.close()
        
        # Save Model
        final_model_path = os.path.join(result_dir, "model.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Model saved to: {final_model_path}")
        
        # Plot Performance
        logging.info("Generating Final Performance Plots...")
        try:
            plot_checkpoint_performance(model, CONFIG, result_dir, DEVICE)
        except Exception as e:
            logging.error(f"Failed to plot checkpoint performance: {e}")
            
        # Plot History
        logging.info("Generating Loss History Plots...")
        try:
            plot_post_training(result_dir, history)
        except Exception as e:
            logging.error(f"Failed to plot training history: {e}")

        logging.info("--- Process Complete ---")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        print("GPU Warmed up and ready!")
    main()