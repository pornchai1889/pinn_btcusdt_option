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
from src.physics.put_option import PutOption
from src.data.normalizer import MarketNormalizer
from src.data.generator import DataGenerator
from src.core.trainer import Trainer
from src.utils.visualization import Visualizer

def load_config(path):
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def locate_mother_config_and_root(base_search_dir):
    """
    Traverses up the directory tree to find the original 'config.yaml' (Mother Config).
    Determines the true Root Experiment Directory to prevent nested fine-tuning folders.
    
    Returns:
        tuple: (root_run_dir, config_path)
    """
    config_path = None
    current_search_dir = base_search_dir
    
    # 1. Search upwards for config.yaml (Limit to 5 levels)
    for _ in range(5):
        candidate_path = os.path.join(current_search_dir, "config.yaml")
        if os.path.exists(candidate_path):
            config_path = candidate_path
            break
        
        # Move up one level
        parent_dir = os.path.dirname(current_search_dir)
        if parent_dir == current_search_dir: # Reached System Root
            break
        current_search_dir = parent_dir
        
    if config_path is None:
        raise FileNotFoundError(f"Error: 'config.yaml' not found in hierarchy starting from: {base_search_dir}")

    # 2. Determine Root Directory (Universal Path Logic)
    # Use the directory where config.yaml was found as the anchor
    anchor_dir = os.path.dirname(config_path)
    normalized_anchor = anchor_dir.replace("\\", "/") # Normalize for consistent splitting
    
    if "/fine_tune" in normalized_anchor:
        # If config is inside a fine_tune folder, strip back to the real mother root.
        # This prevents: runs/Exp1/fine_tune/ft_1/fine_tune/ft_2...
        root_run_dir = normalized_anchor.split("/fine_tune")[0]
    else:
        # If config is at the root, use it directly.
        root_run_dir = normalized_anchor
        
    return os.path.normpath(root_run_dir), config_path

def main():
    # ==========================================
    # 1. Setup Logic & Directory Structure
    # ==========================================
    ft_config_path = "configs/ft_config.yaml"
    ft_config = load_config(ft_config_path)
    
    # The directory we want to fine-tune FROM
    target_base_dir = ft_config['base_run_dir']
    
    try:
        # Locate the true root and the mother config
        root_run_dir, mother_config_path = locate_mother_config_and_root(target_base_dir)
        
        # Setup New Output Directory (Sibling Structure)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ft_folder_name = f"ft_{timestamp}"
        ft_run_dir = os.path.join(root_run_dir, "fine_tune", ft_folder_name)
        os.makedirs(ft_run_dir, exist_ok=True)

        # ==========================================
        # 2. Logging Setup
        # ==========================================
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(ft_run_dir, "finetune.log")),
                logging.StreamHandler()
            ]
        )
         
        logging.info(f"--- Starting Fine-tuning (Universal Structure) ---")
        logging.info(f"Target Base Dir: {target_base_dir}")
        logging.info(f"Mother Config Found: {mother_config_path}")
        logging.info(f"Root Experiment Dir: {root_run_dir}")
        logging.info(f"Output Directory: {ft_run_dir}")

        # Check Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Active Device: {device}")

        # ==========================================
        # 3. Configuration Integration
        # ==========================================
        # Load Mother Config
        config = load_config(mother_config_path)
        
        # Override with Finetune Params from ft_config.yaml
        logging.info("Overriding configuration with finetune parameters...")
        ft_params = ft_config['fine_tune_params']
        
        for k, v in ft_params.items():
            if k in config['training']:
                config['training'][k] = v
            elif k in config['sampling']:
                 config['sampling'][k] = v
            else:
                logging.warning(f"Key '{k}' not found in training/sampling config. Adding to 'training'.")
                config['training'][k] = v
        
        # Enforce Finetune Mode
        config['training']['mode'] = 'finetune'
        
        # Save combined config for reproducibility
        with open(os.path.join(ft_run_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        # ==========================================
        # 4. Initialize Components (AUTO DETECT LOGIC)
        # ==========================================
        exp_name = config['experiment']['name'].lower()
        
        if "put" in exp_name:
            logging.info("Detected Experiment Type: PUT Option")
            physics_engine = PutOption(config)
        else:
            logging.info("Detected Experiment Type: CALL Option")
            physics_engine = CallOption(config)

        normalizer = MarketNormalizer(config)
        data_gen = DataGenerator(config, normalizer, physics_engine)
        
        # Initialize Visualizer (Plots go to the new fine_tune folder)
        viz = Visualizer(config, physics_engine, ft_run_dir) 
        
        # ==========================================
        # 5. Initialize Trainer
        # ==========================================
        # Construct path to the model weights we are fine-tuning FROM
        # Note: model_checkpoint in ft_config is usually "model.pth" or "checkpoints/..."
        # We append it to the target_base_dir specified in ft_config
        checkpoint_path = os.path.join(target_base_dir, ft_config['model_checkpoint'])
        
        if not os.path.exists(checkpoint_path):
             # Fallback: check if it's inside the 'checkpoints' folder of target dir
             fallback_path = os.path.join(target_base_dir, "checkpoints", ft_config['model_checkpoint'])
             if os.path.exists(fallback_path):
                 checkpoint_path = fallback_path

        trainer = Trainer(
            config=config,
            physics_engine=physics_engine,
            data_generator=data_gen,
            visualizer=viz,
            run_dir=ft_run_dir,
            mode='finetune',
            checkpoint_path=checkpoint_path
        )
        
        # ==========================================
        # 6. Start Training
        # ==========================================
        trainer.train()

    except Exception as e:
        logging.critical(f"Fatal error during finetuning: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()