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

    def __init__(self, config, physics_engine):
        """
        Args:
            config (dict): Global configuration dictionary.
            physics_engine (OptionPhysics): Instance used for analytical solutions.
        """
        self.config = config
        self.physics = physics_engine
        
        # Cache specific config sections for easier access
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
            (0, 1, '#2ca02c', 0.45),   # Green (Core)
            (1, 2, '#ff7f0e', 0.15),   # Orange
            (2, 3, '#d62728', 0.04),   # Red
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
                                     color='white' if start_sd < 1 else 'black', 
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
                                  label=f'In-Bound Gaussian: {total_prob_gaussian_in*100:.2f}%')
        water_summary = Line2D([0], [0], color='#0055A4', linestyle='-.', linewidth=2,
                               label=f'Recycled Tails (Water): {total_prob_outliers*100:.2f}%')
        
        final_handles = zone_legend_handles[:3] + [separator, gaussian_summary, water_summary]
        plt.legend(handles=final_handles, loc='upper right', framealpha=0.95, title="Data Distribution Stats")
        
        plt.title(f'Moneyness Distribution: Gaussian Zones + Recycled Outliers (Water Level)\n(Adaptive SD={user_std_factor}, Range Width={range_width:.2f})', fontsize=14, pad=20)
        plt.xlabel('Moneyness (S/K)')
        plt.ylabel('Probability Density')
        plt.ylim(bottom=-max_y*0.08, top=max_y * 1.25)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "moneyness_density_mixed.png"))
        plt.close()

        # =================================================================
        # Plot 2: Data Sampling Distribution
        # =================================================================
        fix_K_mid = (self.market["K_range"][0] + self.market["K_range"][1]) / 2
        fix_K = np.round(fix_K_mid / self.market["K_step"]) * self.market["K_step"]
        
        n_data = self.train_conf["n_sample_data"]
        n_pde = n_data * self.train_conf["n_sample_pde_multiplier"]
        total_points = n_pde + n_data + (n_data*2)

        plt.figure(figsize=(12, 8))
        
        # Generate dummy points for visualization using local helper
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
        
        plt.title(rf'Data Sampling Distribution', fontsize=14)
        plt.xlabel('Time to Maturity (t)')
        plt.ylabel('Spot Price (S)')
        
        info_text = (f"Total Point: {total_points:,}\n"
                     f"PDE: {n_pde:,}\n"
                     f"IVP: {n_data:,}\n"
                     f"BVP: {n_data*2:,}\n"
                     f"Adaptive SD: {user_std_factor}\n"              
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
        Generate 3D Surface and Scatter plots for model validation.
        Saved to the specific checkpoint directory (or root if final).
        
        Updates:
        - Logic to invert Spot Price Axis (Y-axis) for Put Options to align visual orientation with Call Options.
        - Adjusted label padding and figure layout to prevent text clipping on the edges.
        """
        model.eval()
        
        # Extract validation params
        fix_sig = self.config["validation_params"]["fix_sigma"]
        fix_r = self.config["validation_params"]["fix_r"]
        
        # Determine Experiment Type (Call or Put)
        # Used to adjust the visual orientation of the 3D plot
        exp_name = self.config['experiment']['name'].lower()
        is_put_option = "put" in exp_name
        
        # Fix K (Mid range)
        k_min, k_max = self.market["K_range"]
        k_step = self.market["K_step"]
        fix_K = np.round(((k_min + k_max) / 2) / k_step) * k_step
        
        m_min, m_max = self.sampling["moneyness_range"]
        t_min, t_max = self.market["t_range"]
        
        # Generate Grid for 3D Plot
        # S_plot: Spot Price axis
        # t_plot: Time axis
        S_plot = np.linspace(fix_K * m_min, fix_K * m_max, 100)
        t_plot = np.linspace(t_min, t_max, 100)
        S_grid, t_grid = np.meshgrid(S_plot, t_plot)
        
        # Prepare Input Vector
        X_flat = np.zeros((S_grid.size, 5))
        X_flat[:, 0] = t_grid.flatten() # t
        X_flat[:, 1] = S_grid.flatten() # S
        X_flat[:, 2] = fix_sig          # sigma
        X_flat[:, 3] = fix_r            # r
        X_flat[:, 4] = fix_K            # K
        
        # Manual Normalization
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
        
        # Inference
        with torch.no_grad():
            V_pred_norm = model(X_tensor).cpu().numpy().flatten()
            
        # Scaling back (Output of model is V/K)
        V_pred = V_pred_norm * fix_K
        V_pred_grid = V_pred.reshape(S_grid.shape)
        
        # Analytical Solution
        V_true = self.physics.analytical_solution(t_grid, S_grid, fix_K, fix_r, fix_sig)
        
        # =================================================================
        # Plot 1: 3D Surface Comparison
        # =================================================================
        # Increased figure width (16) to prevent label clipping on the right side
        fig = plt.figure(figsize=(16, 8)) 
        
        param_text = rf"Fixed Parameters: $\sigma={fix_sig}, r={fix_r}, K={fix_K:,.0f}$"
        fig.suptitle(f"3D Surface Comparison (Epoch {epoch})\n({param_text})\n", fontsize=14, y=0.98)
        
        # Helper to style 3D axes
        def style_3d_axis(ax, title):
            ax.set_title(title, fontsize=12, pad=10)
            
            # Label Padding: Pushes text away from the axis numbers to prevent overlap
            ax.set_xlabel(rf'Time to Maturity ($\tau$)', labelpad=12)
            ax.set_ylabel('Spot Price (S)', labelpad=15)
            ax.set_zlabel('Option Price (V)', labelpad=18)
            ax.tick_params(axis='z', pad=10)
            # View Adjustment: Standard isometric-like view
            ax.view_init(elev=30, azim=-60)
            
            # Put Option Adjustment:
            # We invert the Y-axis (Spot Price) because ax.plot_surface(X, Y, Z) maps S to Y.
            # Inverting S makes the Put Payoff ramp visual align with the Call Payoff ramp (Low S = High V).
            if is_put_option:
                ax.invert_yaxis() 

        # Subplot 1: Analytical
        ax1 = fig.add_subplot(121, projection='3d')
        # Note: t_grid maps to X-axis, S_grid maps to Y-axis
        ax1.plot_surface(t_grid, S_grid, V_true, cmap='viridis', edgecolor='none', alpha=0.9)
        style_3d_axis(ax1, 'Analytical Solution')
        
        # Subplot 2: Prediction
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(t_grid, S_grid, V_pred_grid, cmap='viridis', edgecolor='none', alpha=0.9)
        style_3d_axis(ax2, 'PINN Prediction')
        
        # Layout Adjustment:
        # Instead of tight_layout, we manually adjust margins to ensure labels aren't cut off.
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
        
        # Reference Line (y=x)
        min_val = min(np.min(v_true_flat), np.min(v_pred_flat))
        max_val = max(np.max(v_true_flat), np.max(v_pred_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Match (y=x)')
        
        rmse = np.sqrt(np.mean((v_true_flat - v_pred_flat)**2))
        corr = np.corrcoef(v_true_flat, v_pred_flat)[0, 1] if np.std(v_true_flat) > 0 and np.std(v_pred_flat) > 0 else 0
        
        plt.title(f'PINN vs. Analytical Predictions\n(RMSE: {rmse:.4f}, R: {corr:.4f})\n{param_text}', fontsize=12)
        plt.xlabel('PINN Prediction')
        plt.ylabel('Analytical Solution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "scatter_comparison.png"))
        plt.close()

    def plot_loss_history(self, history, save_dir=None):
        """
        Generate detailed loss curves.
        
        Args:
            history (dict): Loss history.
            save_dir (str, optional): Override directory. Defaults to None.
        """
        logging.info("Visualizer: Generating Post-training visualizations...")
        
        plot_dir = save_dir if save_dir else self.run_dir
        
        if len(history['total']) > 0:
            epochs = range(1, len(history['total']) + 1)
            
            # =================================================================
            # Plot 1: Detailed Curves (Linear Scale)
            # =================================================================
            fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
            
            # Helper to map Trainer keys to Plot labels
            # Trainer keys: 'total', 'pde', 'data', 'ivp', 'bvp_total', 'bvp_min', 'bvp_max'
            # We map 'bvp_min' -> BVP1, 'bvp_max' -> BVP2
            
            def plot_metric(ax, data, color, label, title, y_label):
                if len(data) == len(epochs):
                    ax.plot(epochs, data, color=color, label=label, linewidth=1.0)
                    ax.set_ylabel(y_label)
                    ax.grid(True, which="both", ls="-", alpha=0.2)
                    ax.legend(loc='upper right')
                    ax.set_title(title, fontsize=10, pad=2)

            plot_metric(axes[0], history['total'], '#1f77b4', 'Total Loss', 'Total Loss', 'Loss')
            plot_metric(axes[1], history['pde'], '#ff7f0e', 'PDE Loss', 'Physics (PDE) Loss', 'PDE Loss')
            plot_metric(axes[2], history['data'], '#2ca02c', 'Data Loss', 'Total Data Loss (IVP + BVP)', 'Data Loss')
            plot_metric(axes[3], history['ivp'], '#d62728', 'IVP Loss', 'Initial Value Problem (t=0) Loss', 'IVP Loss')
            
            # Handle key mapping safely
            bvp1 = history.get('bvp_min', history.get('bvp1', []))
            bvp2 = history.get('bvp_max', history.get('bvp2', []))
            
            plot_metric(axes[4], bvp1, '#9467bd', 'BVP1 Loss', 'Lower Boundary Loss', 'BVP1 Loss')
            plot_metric(axes[5], bvp2, '#8c564b', 'BVP2 Loss', 'Upper Boundary Loss', 'BVP2 Loss')

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

    # Note: We need to ensure run_dir is stored if we want plot_loss_history to work without args.
    # Re-defining init to capture run_dir passed from Trainer.
    def __init__(self, config, physics_engine, run_dir):
        """
        Args:
            config (dict): Global configuration.
            physics_engine (OptionPhysics): For analytical solutions.
            run_dir (str): Root directory for saving default plots.
        """
        self.config = config
        self.physics = physics_engine
        self.run_dir = run_dir
        
        # Cache
        self.market = config['market']
        self.sampling = config['sampling']
        self.train_conf = config['training']