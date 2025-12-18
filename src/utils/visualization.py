import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import norm

class Visualizer:
    """
    Handles all visualization tasks for the PINN project.
    
    This class replicates the exact visualization logic from the original 'pinn_model_a.py',
    ensuring consistency in:
    1. Pre-training Data Distribution (Gaussian Zones + Mixed Sampling).
    2. Checkpoint Validation (3D Surfaces + Scatter Parity).
    3. Post-training Loss Curves.
    """

    def __init__(self, config, physics_engine, run_dir=None):
        """
        Args:
            config (dict): Global configuration dictionary.
            physics_engine (OptionPhysics): Instance used for analytical solutions.
            run_dir (str, optional): Root directory for saving default plots. Defaults to None.
        """
        self.config = config
        self.physics = physics_engine
        self.run_dir = run_dir # เก็บค่าไว้ใช้ (ถ้ามี)
        
        # Cache specific config sections
        self.market = config['market']
        self.sampling = config['sampling']
        self.train_conf = config['training']

    def plot_pre_training(self, save_dir):
        """
        Generate plots that visualize the data distribution BEFORE training starts.
        Replicates 'plot_pre_training' from the original script.
        
        Args:
            save_dir (str): Directory to save the images.
        """
        logging.info("Visualizer: Generating Pre-training visualizations...")
        
        # --- Extract Configs ---
        # Note: Handling key name differences if any, mapping to original logic
        user_std_factor = self.sampling.get("adaptive_std", 1.0)
        time_power = self.sampling.get("time_sampling_power", 2.0)
        m_min, m_max = self.sampling["moneyness_range"]
        t_min, t_max = self.market["t_range"]
        
        # --- Dynamic Sigma Calculation (Logic from pinn_model_a.py) ---
        range_width = m_max - m_min
        std_val = (range_width / 6.0) * user_std_factor
        
        # =================================================================
        # Plot 1: Moneyness Density & Zones (Mixed Distribution)
        # =================================================================
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
        water_level = total_prob_outliers / range_width

        # 2. Plot Main Gaussian Curve
        plt.plot(x, y, color='#333333', linewidth=2.5, label=rf'Base Gaussian ($\sigma$={std_val:.4f})', zorder=10)
        
        # 3. Zones Visualization
        zones = [
            (0, 1, '#2ca02c', 0.50),   # Green (Core)
            (1, 2, '#ff7f0e', 0.50),   # Orange
            (2, 3, '#d62728', 0.50),   # Red
            (3, 10, '#999999', 0.0)    # Grey (Tail)
        ]
        
        zone_legend_handles = []
        
        # Plot Water Level Line (Blue Dashed)
        plt.hlines(water_level, m_min, m_max, colors='#0055A4', linestyles='-.', linewidth=2.0, zorder=15)
        
        # Loop to draw zones
        for start_sd, end_sd, color, h_ratio in zones:
            prob_one_side = norm.cdf(end_sd) - norm.cdf(start_sd)
            pct_band_total = prob_one_side * 2 * 100
            
            label_txt = rf"{start_sd}-{end_sd}$\sigma$: {pct_band_total:.2f}%"
            if end_sd > 3: label_txt = rf">{start_sd}$\sigma$ (Tails)"
            
            zone_legend_handles.append(Patch(facecolor=color, edgecolor='none', label=label_txt))

            # Fill Areas (Right & Left)
            for sign in [1, -1]:
                start = mu + (start_sd * std_val * sign)
                end = mu + (end_sd * std_val * sign)
                
                # Ensure ordered for linspace
                p_start, p_end = min(start, end), max(start, end)
                
                # Clip to plot range
                plot_start = max(p_start, m_min)
                plot_end = min(p_end, m_max)
                
                if plot_start < plot_end:
                    x_fill = np.linspace(plot_start, plot_end, 200)
                    y_fill = norm.pdf(x_fill, mu, std_val)
                    plt.fill_between(x_fill, y_fill, color=color, alpha=0.6)
                    
                    # Text Label (only for significant zones)
                    if end_sd <= 3 and sign == 1: # Label on right side only
                        text_x = (plot_start + plot_end) / 2
                        text_y = norm.pdf(text_x, mu, std_val) * h_ratio
                        if m_min < text_x < m_max:
                            plt.text(text_x, text_y, f"{prob_one_side*100:.1f}%", 
                                     color='black', 
                                     ha='center', va='center', fontsize=9, fontweight='bold')

        # 4. Axis Labels & Markers
        label_y_pos = -max_y * 0.04
        for i in range(1, 4):
            for sign in [1, -1]:
                sd_pos = mu + (i * std_val * sign)
                if m_min < sd_pos < m_max:
                    plt.axvline(sd_pos, color='grey', linestyle=':', alpha=0.5, linewidth=1)
                    txt = rf"+{i}$\sigma$" if sign > 0 else rf"-{i}$\sigma$"
                    plt.text(sd_pos, label_y_pos, txt, ha='center', color='#333333', fontsize=9)

        plt.axvline(mu, color='black', linestyle='--', alpha=0.3, linewidth=1)
        plt.text(mu, label_y_pos, rf"$\mu$", ha='center', color='black', fontsize=10, fontweight='bold')
        
        # 5. Build Custom Legend
        separator = Line2D([0], [0], color='white', label='__________________')
        gaussian_summary = Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333', 
                                  label=f'Gaussian Core Mass: {total_prob_gaussian_in*100:.2f}%')
        water_summary = Line2D([0], [0], color='#0055A4', linestyle='-.', linewidth=2,
                               label=f'Redistributed Tail Mass: {total_prob_outliers*100:.2f}%')
        
        final_handles = zone_legend_handles[:3] + [separator, gaussian_summary, water_summary]
        plt.legend(handles=final_handles, loc='upper right', framealpha=0.95, title="Data Distribution Statistics")
        
        plt.title(
            'Moneyness Distribution: Gaussian Core with Uniform Tail Redistribution \n'
            rf'(Adaptive $\sigma$={user_std_factor}, Interval Width={range_width:.2f})', 
            fontsize=14, pad=20
        )
        plt.xlabel('Moneyness ($S/K$)')
        plt.ylabel('Probability Density')
        plt.ylim(bottom=-max_y*0.08, top=max_y * 1.25)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "moneyness_density_mixed.png"))
        plt.close()

        # =================================================================
        # Plot 2: Data Sampling Distribution (Updated)
        # =================================================================
        fix_K_mid = (self.market["K_range"][0] + self.market["K_range"][1]) / 2
        fix_K = np.round(fix_K_mid / self.market["K_step"]) * self.market["K_step"]
        
        n_data = self.train_conf["n_sample_data"]
        
        pde_multiplier = self.train_conf.get('n_sample_pde_multiplier', 4.0)
        n_pde = int(n_data * pde_multiplier)
        kink_multiplier = self.train_conf.get('n_sample_kink_multiplier', 0.5)
        n_kink = int(n_data * kink_multiplier)
        total_points = n_pde + n_data + (n_data*2) + n_kink

        plt.figure(figsize=(12, 8))
        
        # 1. PDE (Grey)
        u_pde = np.random.uniform(0, 1, n_pde)
        t_pde = t_min + (t_max - t_min) * (u_pde ** time_power) 
        m_pde = self._sample_moneyness_mixed(n_pde, m_min, m_max, std_val).flatten()
        S_pde = fix_K * m_pde
        
        plt.scatter(t_pde, S_pde, c="#858484", s=10, alpha=0.4, label='PDE Collocation Points')
        
        # 2. IVP (Blue)
        t_ivp = np.zeros(n_data)
        m_ivp = self._sample_moneyness_mixed(n_data, m_min, m_max, std_val).flatten()
        S_ivp = fix_K * m_ivp
        # ใส่ r นำหน้าเพื่อความปลอดภัย
        plt.scatter(t_ivp, S_ivp, c='blue', s=20, alpha=0.6, label=r'IVP ($\tau=0$)')

        # [NEW] 3. Kink Focus (Gold Star) - Hard Attention
        # สร้างจุดจำลองที่ S=K, t=0
        t_kink = np.zeros(n_kink)
        S_kink = np.full(n_kink, fix_K)
        # ใช้ zorder เยอะๆ เพื่อให้ลอยอยู่บนสุด, ใช้ marker='*' รูปดาว
        plt.scatter(t_kink, S_kink, c='gold', marker='*', s=150, edgecolors='black', linewidth=0.5, 
                    zorder=10, label=r'Kink Focus ($S=K, \tau=0$)')

        # 4. BVP Upper (Green)
        u_bvp2 = np.random.uniform(0, 1, n_data)
        t_bvp2 = t_min + (t_max - t_min) * (u_bvp2 ** time_power)
        S_bvp2 = np.full(n_data, fix_K * m_max)
        plt.scatter(t_bvp2, S_bvp2, c='green', marker='x', s=25, alpha=0.6, 
                    label=rf'BVP Upper ($S={fix_K * m_max:,.0f}$)')

        # 5. BVP Lower (Red)
        u_bvp1 = np.random.uniform(0, 1, n_data)
        t_bvp1 = t_min + (t_max - t_min) * (u_bvp1 ** time_power)
        S_bvp1 = np.full(n_data, fix_K * m_min)
        plt.scatter(t_bvp1, S_bvp1, c='red', marker='x', s=25, alpha=0.6, 
                    label=rf'BVP Lower ($S={fix_K * m_min:,.0f}$)')

        # Strike Line
        plt.axhline(fix_K, color='black', linestyle='--', linewidth=1.5, alpha=0.5, 
                    label=rf'Strike ($K={fix_K:,.0f}$)')
        
        plt.title(rf'Data Sampling Distribution', fontsize=14)
        plt.xlabel(r'Time to Maturity ($\tau$)', fontsize=12)
        plt.ylabel(r'Spot Price ($S$)', fontsize=12)
        
        # Info Box Update
        info_text = (f"Total Points: {total_points:,}\n"
                     f"PDE: {n_pde:,}\n"
                     f"IVP: {n_data:,}\n"
                     f"Kink Focus: {n_kink:,}\n"  # เพิ่มบรรทัดนี้
                     f"BVP: {n_data*2:,}\n"
                     f"Adaptive $\\sigma$: {user_std_factor}\n"              
                     f"Time Power: {time_power}")

        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                 fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.9))

        plt.legend(loc='center right', framealpha=0.95)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "data_sampling_distribution.png"))
        plt.close()

    def plot_checkpoint_performance(self, model, epoch, device, save_dir):
        """
        Generate comprehensive validation plots: 3D Surface, Scatter Parity, and 2D Payoff Slice.
        Saved to the specific checkpoint directory (or root if final).
        
        Updates:
        - Added [Plot 3]: 2D Payoff at Maturity (t=0) to visualize the "Kink" behavior.
        - Includes 'Kink Error' metric to quantify sharpness deviation at Strike Price.
        - Maintains Put/Call visual alignment by inverting axes where appropriate.
        """
        model.eval()
        
        # --- 1. Preparation & Configuration ---
        fix_sig = self.config["validation_params"]["fix_sigma"]
        fix_r = self.config["validation_params"]["fix_r"]
        
        # Determine Experiment Type (Call or Put)
        exp_name = self.config['experiment']['name'].lower()
        is_put_option = "put" in exp_name
        
        # Fix K (Mid range, rounded to nearest step)
        k_min, k_max = self.market["K_range"]
        k_step = self.market["K_step"]
        fix_K = np.round(((k_min + k_max) / 2) / k_step) * k_step
        
        # Ranges
        m_min, m_max = self.sampling["moneyness_range"]
        t_min, t_max = self.market["t_range"]

        # Param Text for Plots
        param_text = rf"Fixed Parameters: $\sigma={fix_sig}, r={fix_r}, K={fix_K:,.0f}$"

        # --- 2. Data Generation for 3D & Scatter ---
        S_plot = np.linspace(fix_K * m_min, fix_K * m_max, 100)
        t_plot = np.linspace(t_min, t_max, 100)
        S_grid, t_grid = np.meshgrid(S_plot, t_plot)
        
        # Prepare Input Vector (Batch Processing)
        X_flat = np.zeros((S_grid.size, 5))
        X_flat[:, 0] = t_grid.flatten() # t
        X_flat[:, 1] = S_grid.flatten() # S
        X_flat[:, 2] = fix_sig
        X_flat[:, 3] = fix_r
        X_flat[:, 4] = fix_K
        
        # Manual Normalization (Matches training logic)
        c_m = self.market
        t_norm = (X_flat[:, 0] - c_m["t_range"][0]) / (c_m["t_range"][1] - c_m["t_range"][0])
        S_norm = (X_flat[:, 1] - c_m["S_range"][0]) / (c_m["S_range"][1] - c_m["S_range"][0])
        sig_norm = (X_flat[:, 2] - c_m["sigma_range"][0]) / (c_m["sigma_range"][1] - c_m["sigma_range"][0])
        r_norm = (X_flat[:, 3] - c_m["r_range"][0]) / (c_m["r_range"][1] - c_m["r_range"][0])
        K_norm = (X_flat[:, 4] - c_m["K_range"][0]) / (c_m["K_range"][1] - c_m["K_range"][0])
        
        X_tensor = torch.tensor(
            np.stack([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1), 
            dtype=torch.float32
        ).to(device)
        
        # Inference 3D
        with torch.no_grad():
            V_pred_norm = model(X_tensor).cpu().numpy().flatten()
            
        V_pred = V_pred_norm * fix_K
        V_pred_grid = V_pred.reshape(S_grid.shape)
        V_true = self.physics.analytical_solution(t_grid, S_grid, fix_K, fix_r, fix_sig)
        
        # =================================================================
        # Plot 1: 3D Surface Comparison
        # =================================================================
        fig = plt.figure(figsize=(16, 8)) 
        fig.suptitle(f"Option Price Surface Comparison (Epoch {epoch})\n{param_text}", fontsize=14, y=0.98)
        
        def style_3d_axis(ax, title):
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel(rf'Time to Maturity ($\tau$)', labelpad=12)
            ax.set_ylabel('Spot Price (S)', labelpad=15)
            ax.set_zlabel('Option Price (V)', labelpad=18)
            ax.tick_params(axis='z', pad=10)
            ax.view_init(elev=30, azim=-60)
            
            # Put Option Adjustment: Invert Spot Price axis to match Call visual flow
            if is_put_option:
                ax.invert_yaxis() 

        # Analytical
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(t_grid, S_grid, V_true, cmap='viridis', edgecolor='none', alpha=0.9)
        style_3d_axis(ax1, 'Analytical Solution')
        
        # Prediction
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(t_grid, S_grid, V_pred_grid, cmap='viridis', edgecolor='none', alpha=0.9)
        style_3d_axis(ax2, 'PINN Prediction')
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.15)
        plt.savefig(os.path.join(save_dir, "3d_surface_comparison.png"), dpi=120)
        plt.close()
        
        # =================================================================
        # Plot 2: Scatter Comparison
        # =================================================================
        plt.figure(figsize=(8, 8))
        v_true_flat = V_true.flatten()
        v_pred_flat = V_pred.flatten()
        
        plt.scatter(v_pred_flat, v_true_flat, alpha=0.5, s=10, label='Prediction Points')
        
        min_val = min(np.min(v_true_flat), np.min(v_pred_flat))
        max_val = max(np.max(v_true_flat), np.max(v_pred_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Match (y=x)')
        
        rmse = np.sqrt(np.mean((v_true_flat - v_pred_flat)**2))
        corr = np.corrcoef(v_true_flat, v_pred_flat)[0, 1] if np.std(v_true_flat) > 0 and np.std(v_pred_flat) > 0 else 0
        
        plt.title(f'Parity Plot (Epoch {epoch})\nRMSE: {rmse:.4f}, R: {corr:.4f}\n({param_text})', fontsize=12)
        plt.xlabel('PINN Prediction')
        plt.ylabel('Analytical Solution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "scatter_comparison.png"))
        plt.close()

        # =================================================================
        # Plot 3: 2D Payoff Slice at Maturity (The "Kink" Analysis)
        # =================================================================
        # We generate a dense slice exactly at t=0 (Maturity) to observe the hinge behavior
        
        n_slice = 200
        S_slice_raw = np.linspace(fix_K * 0.5, fix_K * 1.5, n_slice) # Focus around K
        t_slice_raw = np.zeros(n_slice) # t=0 (Maturity)
        
        # Prepare inputs for slice
        X_slice = np.zeros((n_slice, 5))
        X_slice[:, 0] = t_slice_raw
        X_slice[:, 1] = S_slice_raw
        X_slice[:, 2] = fix_sig
        X_slice[:, 3] = fix_r
        X_slice[:, 4] = fix_K
        
        # Normalize Slice
        t_norm_s = (X_slice[:, 0] - c_m["t_range"][0]) / (c_m["t_range"][1] - c_m["t_range"][0])
        S_norm_s = (X_slice[:, 1] - c_m["S_range"][0]) / (c_m["S_range"][1] - c_m["S_range"][0])
        # Note: other params (sig, r, K) are constant, reused from normalized values above
        sig_norm_s = np.full(n_slice, sig_norm[0]) 
        r_norm_s = np.full(n_slice, r_norm[0])
        K_norm_s = np.full(n_slice, K_norm[0])
        
        X_tensor_s = torch.tensor(
            np.stack([t_norm_s, S_norm_s, sig_norm_s, r_norm_s, K_norm_s], axis=1), 
            dtype=torch.float32
        ).to(device)
        
        with torch.no_grad():
            V_slice_pred = model(X_tensor_s).cpu().numpy().flatten() * fix_K
            
        # Analytical Payoff
        # For t=0, Analytical should be exactly Max(S-K, 0) or Max(K-S, 0)
        V_slice_true = self.physics.analytical_solution(t_slice_raw, S_slice_raw, fix_K, fix_r, fix_sig)
        
        # --- Metric: Kink Error (Deviation at Strike) ---
        # We find the closest point to K and measure absolute error
        idx_K = (np.abs(S_slice_raw - fix_K)).argmin()
        val_pred_at_K = V_slice_pred[idx_K]
        val_true_at_K = V_slice_true[idx_K]
        kink_error = abs(val_pred_at_K - val_true_at_K)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        plt.plot(S_slice_raw, V_slice_true, label='Analytical (Payoff)', color='black', linewidth=1.5, alpha=0.7)
        plt.plot(S_slice_raw, V_slice_pred, label='PINN Prediction', color='#ff7f0e', linestyle='--', linewidth=2.0)
        
        # Highlight the Kink
        plt.scatter([S_slice_raw[idx_K]], [val_pred_at_K], color='red', zorder=5, s=50)
        
        # Styling
        title_text = rf"Payoff at Maturity ($\tau=0$) & Kink Analysis"
        metric_text = f"Pointwise Abs. Error at Strike ($S=K$): {kink_error:.5f}\n({param_text})"
        
        plt.title(f"{title_text}\n{metric_text}", fontsize=13)
        plt.xlabel('Spot Price (S)')
        plt.ylabel('Option Price (V)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Put Option Adjustment: Invert X-axis
        # For Put: High S (OTM) is on Right, Low S (ITM) is on Left.
        # Graph shape: \___
        # By inverting X, High S goes to Left. Shape becomes ___/ which aligns with Call visual.
        if is_put_option:
            plt.gca().invert_xaxis()

        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "payoff_at_maturity_kink.png"))
        plt.close()

    def plot_loss_history(self, history, save_dir=None):
        """
        Generate detailed loss curves.
        [Update]: Added 'Kink Loss' tracking to visualize hard attention learning.
        
        Args:
            history (dict): Loss history.
            save_dir (str, optional): Override directory. Defaults to None.
        """
        logging.info("Visualizer: Generating Post-training visualizations...")
        
        plot_dir = save_dir if save_dir else (self.run_dir if self.run_dir else ".")
        
        if len(history['total']) > 0:
            epochs = range(1, len(history['total']) + 1)
            
            # =================================================================
            # Plot 1: Detailed Curves (Linear Scale)
            # =================================================================
            # [Update]: Increased subplots to 7 to accommodate Kink Loss
            # Adjusted figsize height to maintain readability (12x21)
            fig, axes = plt.subplots(7, 1, figsize=(12, 21), sharex=True)
            
            # Helper to map Trainer keys to Plot labels
            def plot_metric(ax, data, color, label, title, y_label):
                if len(data) == len(epochs):
                    ax.plot(epochs, data, color=color, label=label, linewidth=1.0)
                    ax.set_ylabel(y_label)
                    ax.grid(True, which="both", ls="-", alpha=0.2)
                    ax.legend(loc='upper right')
                    ax.set_title(title, fontsize=10, pad=2)

            # 1-4. Standard Losses
            plot_metric(axes[0], history['total'], '#1f77b4', 'Total Loss', 'Total Loss', 'Loss')
            plot_metric(axes[1], history['pde'], '#ff7f0e', 'PDE Loss', 'Physics (PDE) Loss', 'PDE Loss')
            plot_metric(axes[2], history['data'], '#2ca02c', 'Data Loss', 'Total Data Loss (IVP + BVP + Kink)', 'Data Loss')
            plot_metric(axes[3], history['ivp'], '#d62728', 'IVP Loss', 'Initial Value Problem (t=0) Loss', 'IVP Loss')
            
            # 5-6. Boundary Conditions
            bvp1 = history.get('bvp_min', history.get('bvp1', []))
            bvp2 = history.get('bvp_max', history.get('bvp2', []))
            
            plot_metric(axes[4], bvp1, '#9467bd', 'BVP1 Loss', 'Lower Boundary Loss', 'BVP1 Loss')
            plot_metric(axes[5], bvp2, '#8c564b', 'BVP2 Loss', 'Upper Boundary Loss', 'BVP2 Loss')

            # [Update]: 7. Kink Loss (Hard Attention)
            # Using .get() ensures backward compatibility if 'kink' is missing from history
            kink_loss = history.get('kink', [])
            plot_metric(axes[6], kink_loss, "#840082ff", 'Kink Loss', 'Kink Hard Attention Loss (S=K)', 'Kink Loss')

            axes[-1].set_xlabel('Epoch')
            fig.suptitle('Detailed Training Curves (Linear Scale)', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(os.path.join(plot_dir, "detailed_training_curves.png"))
            plt.close()
            
            # =================================================================
            # Plot 2: Total Loss Only (Standalone)
            # =================================================================
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

    def _sample_moneyness_mixed(self, n, m_min, m_max, std_val):
        """
        Internal Helper: Mixed Distribution Strategy.
        Used strictly for generating the visualization scatter plot.
        Replicates logic from 'pinn_model_a.py' / generator.
        """
        # 1. Generate Raw Gaussian
        data = np.random.normal(1.0, std_val, (n, 1))
        
        # 2. Identify Outliers
        flat_data = data.flatten()
        outliers_mask = (flat_data < m_min) | (flat_data > m_max)
        n_out = np.sum(outliers_mask)
        
        # 3. Resample Outliers (Uniformly)
        if n_out > 0:
            flat_data[outliers_mask] = np.random.uniform(m_min, m_max, n_out)
        
        return flat_data.reshape(n, 1)