# src/utils/logger.py
import logging
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """
    Wrapper for TensorBoard and Standard Logging.
    
    [Update]: Refactored for performance and stability.
    - Configuration lists moved to __init__ for better state management.
    - Helper functions extracted to private methods to reduce overhead.
    - Maintains dynamic 'Catch-all' logic for future-proofing.
    """
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        
        # 1. Acronyms: These keys will always be converted to UPPERCASE.
        self.acronyms = {
            'rmse', 'mae', 'smape', 'pde', 'ivp', 'bvp', 'l2', 'h1', 'mse', 'nn', 'dl', 'ft'
        }
        
        # 2. Tag Overrides: Specific mapping for keys.
        self.tag_overrides = {
            'data': 'Data Total',
            'bvp_min': 'BVP1 Min',
            'bvp_max': 'BVP2 Max',
            'r_score': 'Corr',
            'bvp_total': 'BVP Total',
            
        }
        
        # 3. Display Order Configuration (Moved from method to init)
        # Defines the priority order for console logging.
        self.main_loss_order = ['total', 'data', 'pde']
        self.detail_loss_order = ['ivp', 'bvp_total', 'bvp_min', 'bvp_max', 'kink']
        self.metric_order = ['smape', 'rmse', 'kink_mae', 'r_score', 'mae', 'bias', 'max_error']

    def _format_tag_name(self, key):
        """
        Helper to format keys into professional Chart Titles.
        Priority: Overrides > Acronyms > Title Case.
        """
        if key in self.tag_overrides:
            return self.tag_overrides[key]
            
        if key.lower() in self.acronyms or len(key) <= 3:
            return key.upper()
            
        return key.replace('_', ' ').title()

    def _build_string_parts(self, data_dict, priority_keys, precision=".8f"):
        """
        Internal helper to format a subset of dictionary items into string parts.
        Returns:
            parts (list): List of formatted strings ["Key:Val", ...]
            processed (set): Set of keys that were processed (to handle leftovers later)
        """
        parts = []
        processed = set()
        
        # Process Priority Keys first
        for key in priority_keys:
            if key in data_dict:
                name = self._format_tag_name(key)
                val = data_dict[key]
                
                # Apply specific formatting rules
                if key == 'smape':
                    parts.append(f"{name}:{val:.2f}%")
                else:
                    parts.append(f"{name}:{val:{precision}}")
                
                processed.add(key)
        
        return parts, processed

    def log_training_loss(self, epoch, losses):
        """
        Log training losses to TensorBoard using a dynamic loop.
        """
        # Define keys that belong to the Main "Loss" group
        main_keys_set = set(self.main_loss_order)

        for key, value in losses.items():
            # Determine Group based on importance
            group = "Loss" if key in main_keys_set else "Loss Detail"
            
            # Format Name
            pretty_name = self._format_tag_name(key)
            
            # Log to TensorBoard
            self.writer.add_scalar(f'{group}/{pretty_name}', value, epoch)

    def log_validation_metrics(self, epoch, metrics, losses):
        """
        Log validation metrics to TensorBoard and Console.
        Uses optimized class-level configurations and helper methods.
        """
        # --- 1. TensorBoard Logging (Dynamic Loop) ---
        for key, value in metrics.items():
            pretty_name = self._format_tag_name(key)
            self.writer.add_scalar(f'Metrics Ratio/{pretty_name}', value, epoch)

        # --- 2. Console Logging (Dynamic & Ordered) ---
        
        # A. Build Main Losses String
        main_parts, processed_main = self._build_string_parts(
            losses, self.main_loss_order, precision=".8f"
        )

        # B. Build Detailed Losses String
        # B1. Priority Details
        detail_parts, processed_details = self._build_string_parts(
            losses, self.detail_loss_order, precision=".8f"
        )
        
        # B2. Catch-all for leftovers (Dynamic Loss Logging)
        # Adds any loss that wasn't in Main OR Detail priority lists.
        for key, val in losses.items():
            if key not in processed_main and key not in processed_details:
                name = self._format_tag_name(key)
                detail_parts.append(f"{name}:{val:.8f}")

        # C. Build Metrics String
        # C1. Priority Metrics
        metric_parts, processed_metrics = self._build_string_parts(
            metrics, self.metric_order, precision=".4f"
        )
        
        # C2. Dynamic Metrics (Catch-all for experimental metrics)
        for key, val in metrics.items():
            if key not in processed_metrics:
                name = self._format_tag_name(key)
                metric_parts.append(f"{name}:{val:.4f}")

        # Construct the final log message using join() for clean spacing
        log_msg = (
            f"Epoch {epoch:5d} | "
            f"Main Losses: [{' '.join(main_parts)}] | "
            f"Detailed: [{' '.join(detail_parts)}] | "
            f"Val(Ratio): [{' '.join(metric_parts)}]"
        )
        logging.info(log_msg)

    def close(self):
        self.writer.close()