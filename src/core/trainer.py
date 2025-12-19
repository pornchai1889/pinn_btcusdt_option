# src/core/trainer.py
import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Internal modules
from src.models.pinn_net import UniversalPINN
from src.physics.base_option import OptionPhysics
from src.data.generator import DataGenerator
from src.utils.visualization import Visualizer
from src.utils.metrics import MetricsCalculator
from src.utils.logger import TrainingLogger

class Trainer:
    """
    The Engine Class responsible for managing the training lifecycle.
    
    Refactored to align with 'pinn_model_a.py' directory structure:
    - Periodic Checkpoints -> runs/run_name/checkpoints/epoch_N/ (Model + Plots)
    - Final/Interrupted Model -> runs/run_name/ (Model + Plots)
    
    [Update]: 
    - Incorporated 'Kink Loss' with configurable sampling multiplier.
    - Refactored Loss Weights to be fully configurable via config.yaml.
    """
    def __init__(self, config, physics_engine: OptionPhysics, 
                 data_generator: DataGenerator, visualizer: Visualizer,
                 run_dir, mode='scratch', checkpoint_path=None):
        
        self.config = config
        self.physics = physics_engine
        self.data_gen = data_generator
        self.viz = visualizer
        self.run_dir = run_dir
        
        # Setup Device
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        
        # Initialize Components
        self._init_model(mode, checkpoint_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.loss_fn = nn.MSELoss()
        
        # Initialize Logger (TensorBoard + Console)
        self.logger = TrainingLogger(run_dir)
        
        # Pre-fetch validation set (Fixed for consistency)
        self._prepare_validation_set()
        
        # Track history for post-training plots
        # Added 'kink' to track the specific loss for the strike price point
        self.history = {
            'total': [], 'pde': [], 'data': [], 
            'ivp': [], 'bvp_total': [], 
            'bvp_min': [], 'bvp_max': [],
            'kink': [] 
        }
        
    def _init_model(self, mode, checkpoint_path):
        """Initialize model architecture and load weights if needed."""
        self.model = UniversalPINN(self.config).to(self.device)
        if mode == 'finetune':
            if checkpoint_path and os.path.exists(checkpoint_path):
                logging.info(f"Loading weights from: {checkpoint_path}")
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
    def _prepare_validation_set(self):
        """
        Pre-calculate Validation Sets (General & Kink) and their Ground Truths.
        Executed once at initialization to minimize overhead during training loops.
        """
        conf_train = self.config['training']
        
        # --- 1. General Validation Set (Monte Carlo) ---
        n_val = conf_train.get('n_val_sample', 20000)
        logging.info(f"Generating fixed General validation set ({n_val} samples)...")
        
        # Generate & Transfer to Device
        self.val_data_norm = self.data_gen.get_validation_batch(n_val)
        self.val_tensor = torch.from_numpy(self.val_data_norm).float().to(self.device)
        
        # Calculate Ground Truth (General)
        t, S, sigma, r, K = self.data_gen.norm.denormalize_batch(self.val_data_norm)
        self.val_K = K
        val_V_true = self.physics.analytical_solution(t, S, K, r, sigma)
        
        # Store Ratio for metrics (V/K)
        # Flattening here ensures consistency with prediction output
        self.val_ratio_true = (val_V_true / (self.val_K + 1e-8)).flatten()

        # --- 2. Kink Validation Set (Targeted Hard Attention) ---
        # Defaults to 2000 if 'n_val_kink' is not yet in config
        n_kink_val = conf_train.get('n_kink_val_sample', 2000)
        logging.info(f"Generating fixed Kink validation set ({n_kink_val} samples)...")
        
        kink_x_np = self.data_gen.get_kink_batch(n_kink_val)
        self.val_kink_tensor = torch.from_numpy(kink_x_np).float().to(self.device)
        
        # Ground Truth for Kink (S=K, t=0) is always 0.
        # Pre-allocating array avoids recreating it every epoch.
        self.val_kink_true = np.zeros(n_kink_val, dtype=np.float32)

    def train(self):
        """Main execution loop."""
        conf_train = self.config['training']
        epochs = conf_train['epochs']
        val_interval = conf_train['val_interval']
        ckpt_interval = conf_train['checkpoint_epochs']

        logging.info(f"Starting training on {self.device} for {epochs} epochs...")
        
        try:
            for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
                loss_dict = self._train_step()
                
                # 1. Update History
                self._update_history(loss_dict)
                
                # 2. TensorBoard Logging
                if epoch % 10 == 0:
                    self.logger.log_training_loss(epoch, loss_dict)

                # 3. Validation
                if epoch % val_interval == 0:
                    self._validate(epoch, loss_dict)

                # 4. Checkpointing (Model + Plots in subfolder)
                if epoch % ckpt_interval == 0:
                    self._save_snapshot(epoch, is_final=False)

        except KeyboardInterrupt:
            logging.warning("\nTraining interrupted by user! Saving emergency state...")
        
        except Exception as e:
            logging.error(f"\nCritical error during training: {e}")
            raise e
            
        finally:
            # 5. Final Save (Model + Plots in ROOT folder)
            self._save_snapshot("final", is_final=True)
            self.logger.close()
            
            # 6. Generate Loss History Plot
            logging.info("Generating Loss History Plots...")
            self.viz.plot_loss_history(self.history)
            logging.info("Training workflow complete.")

    def _train_step(self):
        """Single optimization step with configurable weighted losses."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # --- Configuration & Weights (From Config) ---
        conf_train = self.config['training']
        n_data = conf_train['n_sample_data']
        pde_multiplier = conf_train.get('n_sample_pde_multiplier', 4.0)
        n_pde = int(n_data * pde_multiplier)
        kink_multiplier = conf_train.get('n_sample_kink_multiplier', 0.5)
        n_kink = int(n_data * kink_multiplier)
        
        # Extract weights from config (with defaults if missing)
        w_physics = conf_train.get('physics_loss_weight', 1.0)
        w_ivp = conf_train.get('ivp_loss_weight', 10.0)
        w_bvp = conf_train.get('bvp_loss_weight', 10.0)   
        w_kink = conf_train.get('kink_loss_weight', 100.0) 
        
        # --- 1. Data Loss (Boundary Conditions) ---
        # IVP (t=0)
        ivp_x, ivp_y = self.data_gen.get_ivp_batch(n_data)
        pred_ivp = self.model(self._to_tensor(ivp_x))
        loss_ivp = self.loss_fn(pred_ivp, self._to_tensor(ivp_y))

        # BVP (Upper/Lower S)
        bvp_lx, bvp_ly, bvp_ux, bvp_uy = self.data_gen.get_bvp_batch(n_data)
        loss_bvp_l = self.loss_fn(self.model(self._to_tensor(bvp_lx)), self._to_tensor(bvp_ly))
        loss_bvp_u = self.loss_fn(self.model(self._to_tensor(bvp_ux)), self._to_tensor(bvp_uy))
        loss_bvp_total = loss_bvp_l + loss_bvp_u
        
        # --- 2. Kink Loss (Hard Attention) ---
        # Specifically target S=K at t=0 to force sharp turn
        # Using dynamic batch size calculated from config multiplier
        kink_x_np = self.data_gen.get_kink_batch(n_kink)
        kink_x = self._to_tensor(kink_x_np)        

        kink_target = torch.zeros(kink_x.shape[0], 1).to(self.device) # Payoff at ATM is exactly 0
        pred_kink = self.model(kink_x)
        loss_kink = self.loss_fn(pred_kink, kink_target)

        # Combined Data Loss with Configurable Weights
        loss_data = (w_ivp * loss_ivp) + (w_bvp * loss_bvp_total) + (w_kink * loss_kink)

        # --- 3. Physics Loss (PDE) ---
        pde_x = self._to_tensor(self.data_gen.get_pde_batch(n_pde), requires_grad=True)
        pde_res = self.physics.compute_pde_residual(self.model, pde_x)
        loss_physics = w_physics * self.loss_fn(pde_res, torch.zeros_like(pde_res))

        # --- Backprop ---
        total_loss = loss_data + loss_physics
        total_loss.backward()
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'pde': loss_physics.item(),
            'data': loss_data.item(),
            'ivp': loss_ivp.item(),
            'bvp_total': loss_bvp_total.item(),
            'bvp_min': loss_bvp_l.item(),
            'bvp_max': loss_bvp_u.item(),
            'kink': loss_kink.item()
        }

    def _validate(self, epoch, loss_dict):
        """
        Validation routine using MetricsCalculator.
        Uses pre-calculated tensors for efficiency.
        """
        self.model.eval()
        with torch.no_grad():
            # ---------------------------------------------------------
            # 1. Standard Global Validation
            # ---------------------------------------------------------
            # Inference
            val_pred_ratio = self.model(self.val_tensor).cpu().numpy().flatten()
            
            # Compute Standard Metrics (Comparing against pre-calc true values)
            metrics = MetricsCalculator.compute_all_metrics(self.val_ratio_true, val_pred_ratio)
            
            # ---------------------------------------------------------
            # 2. Kink-Specific Validation
            # ---------------------------------------------------------
            # Inference on specific Kink batch
            pred_kink_ratio = self.model(self.val_kink_tensor).cpu().numpy().flatten()
            
            # Compute Kink Metrics (Comparing against pre-calc zeros)
            kink_metrics = MetricsCalculator.compute_kink_metrics(self.val_kink_true, pred_kink_ratio)
            
            # Merge & Log
            metrics.update(kink_metrics)
            self.logger.log_validation_metrics(epoch, metrics, loss_dict)

    def _save_snapshot(self, tag, is_final=False):
        """
        Saves the model and generates performance plots.
        Args:
            tag (str|int): Identifier for the save (e.g., epoch number or "final").
            is_final (bool): If True, saves to root dir. If False, saves to checkpoints/epoch_X.
        """
        # 1. Determine Directory
        if is_final:
            save_dir = self.run_dir
            logging.info(f"Saving FINAL model to root: {save_dir}")
        else:
            save_dir = os.path.join(self.run_dir, "checkpoints", f"epoch_{tag}")
            os.makedirs(save_dir, exist_ok=True)

        # 2. Save Model State
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

        # 3. Generate Performance Plots
        try:
            self.viz.plot_checkpoint_performance(
                model=self.model, 
                epoch=tag, 
                device=self.device, 
                save_dir=save_dir
            )
        except Exception as e:
            logging.error(f"Failed to generate checkpoint plots: {e}")

    def _update_history(self, losses):
        """Update internal history list for final plotting."""
        self.history['total'].append(losses['total'])
        self.history['pde'].append(losses['pde'])
        self.history['data'].append(losses['data'])
        self.history['ivp'].append(losses['ivp'])
        self.history['bvp_total'].append(losses['bvp_total'])
        self.history['bvp_min'].append(losses['bvp_min'])
        self.history['bvp_max'].append(losses['bvp_max'])
        self.history['kink'].append(losses['kink']) 

    def _to_tensor(self, array, requires_grad=False):
        t = torch.from_numpy(array).float().to(self.device)
        if requires_grad:
            t.requires_grad = True
        return t