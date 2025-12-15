import os
import yaml
import logging
import torch
import sys
from datetime import datetime

# Ensure python can find 'src' if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import Custom Modules
from src.physics.call_option import CallOption
from src.data.normalizer import MarketNormalizer
from src.data.generator import DataGenerator
from src.core.trainer import Trainer
from src.utils.visualization import Visualizer

def load_config(path):
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # ==========================================
    # 1. Configuration & Environment Setup
    # ==========================================
    config_path = "configs/config_call.yaml"
    config = load_config(config_path)
    
    # Setup Run Directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = config['experiment']['name']
    run_dir = os.path.join("runs", f"train_{timestamp}_{exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config for reproducibility
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"--- Starting Experiment: {exp_name} ---")
    logging.info(f"Output Directory: {run_dir}")
    
    # Check Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Active Device: {device}")
    if torch.cuda.is_available():
        torch.zeros(1).cuda() # Warmup
        logging.info("GPU Warmed up and ready.")

    try:
        # ==========================================
        # 2. Initialize Core Components
        # ==========================================
        
        # Physics Engine (Handles PDE logic and analytical solutions)
        physics_engine = CallOption(config)
        
        # Data Pipeline (Normalization & Generation)
        normalizer = MarketNormalizer(config)
        data_gen = DataGenerator(config, normalizer)
        
        # Visualization Engine
        # Note: 'run_dir' is passed here to set the default save path
        viz = Visualizer(config, physics_engine, run_dir)
        
        # Generate Pre-training Plots (Distribution Analysis)
        # Saves to the root run directory
        viz.plot_pre_training(save_dir=run_dir)

        # ==========================================
        # 3. Initialize Trainer
        # ==========================================
        trainer = Trainer(
            config=config,
            physics_engine=physics_engine,
            data_generator=data_gen,
            visualizer=viz,     # Inject visualizer for checkpoints
            run_dir=run_dir,
            mode='scratch'      # Use 'finetune' and provide checkpoint_path if needed
        )

        # ==========================================
        # 4. Execute Training
        # ==========================================
        trainer.train()

    except Exception as e:
        logging.critical(f"Fatal error during execution: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()