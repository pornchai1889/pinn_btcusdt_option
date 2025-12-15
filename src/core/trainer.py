import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

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
        self.history = {
            'total': [], 'pde': [], 'data': [], 
            'ivp': [], 'bvp_total': [], 
            'bvp_min': [], 'bvp_max': []
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
        """Pre-calculate Validation Ground Truth."""
        logging.info("Generating fixed validation set...")
        n_val = self.config['training']['n_val_sample']
        self.val_data_norm = self.data_gen.get_validation_batch(n_val)
        self.val_tensor = torch.from_numpy(self.val_data_norm).float().to(self.device)
        
        # Denormalize to calculate True V
        t, S, sigma, r, K = self.data_gen.norm.denormalize_batch(self.val_data_norm)
        self.val_K = K
        
        # Analytical Solution
        self.val_V_true = self.physics.analytical_solution(t, S, K, r, sigma)
        
        # Ratio for metrics (V/K)
        self.val_ratio_true = self.val_V_true / (self.val_K + 1e-8)

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
        """Single optimization step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack Config
        n_data = self.config['training']['n_sample_data']
        n_pde = n_data * self.config['training']['n_sample_pde_multiplier']
        phys_weight = self.config['training']['physics_loss_weight']
        
        # --- Data Loss ---
        ivp_x, ivp_y = self.data_gen.get_ivp_batch(n_data)
        loss_ivp = self.loss_fn(self.model(self._to_tensor(ivp_x)), self._to_tensor(ivp_y))

        bvp_lx, bvp_ly, bvp_ux, bvp_uy = self.data_gen.get_bvp_batch(n_data)
        loss_bvp_l = self.loss_fn(self.model(self._to_tensor(bvp_lx)), self._to_tensor(bvp_ly))
        loss_bvp_u = self.loss_fn(self.model(self._to_tensor(bvp_ux)), self._to_tensor(bvp_uy))
        loss_bvp_total = loss_bvp_l + loss_bvp_u
        
        loss_data = loss_ivp + loss_bvp_total

        # --- Physics Loss ---
        pde_x = self._to_tensor(self.data_gen.get_pde_batch(n_pde), requires_grad=True)
        pde_res = self.physics.compute_pde_residual(self.model, pde_x)
        loss_physics = phys_weight * self.loss_fn(pde_res, torch.zeros_like(pde_res))

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
            'bvp_max': loss_bvp_u.item()
        }

    def _validate(self, epoch, loss_dict):
        """Validation routine using MetricsCalculator."""
        self.model.eval()
        with torch.no_grad():
            val_pred_ratio = self.model(self.val_tensor).cpu().numpy().flatten()
            val_true_ratio = self.val_ratio_true.flatten()
            
            # Compute Metrics
            metrics = MetricsCalculator.compute_all_metrics(val_true_ratio, val_pred_ratio)
            
            # Log
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
            #logging.info(f"Saving Checkpoint to: {save_dir}")

        # 2. Save Model State
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

        # 3. Generate Performance Plots (3D Surface & Scatter)
        # We pass the specific save_dir so plots end up next to the model file
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


    def _to_tensor(self, array, requires_grad=False):
        t = torch.from_numpy(array).float().to(self.device)
        if requires_grad:
            t.requires_grad = True
        return t