import logging
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """
    Wrapper for TensorBoard and Standard Logging.
    
    [Update]: Added logging support for 'Kink Loss' to track 
    sharpness learning progress at the strike price.
    """
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

    def log_training_loss(self, epoch, losses):
        """
        Log training losses to TensorBoard.
        Args:
            epoch (int): Current epoch.
            losses (dict): Dictionary of loss components.
        """
        # Main Losses
        self.writer.add_scalar('Loss/Total', losses['total'], epoch)
        self.writer.add_scalar('Loss/PDE', losses['pde'], epoch)
        self.writer.add_scalar('Loss/Data_Total', losses['data'], epoch)
        
        # Detailed Components
        self.writer.add_scalar('Loss_Detail/IVP', losses['ivp'], epoch)
        self.writer.add_scalar('Loss_Detail/BVP_Total', losses['bvp_total'], epoch)
        self.writer.add_scalar('Loss_Detail/BVP1_Min', losses['bvp_min'], epoch)
        self.writer.add_scalar('Loss_Detail/BVP2_Max', losses['bvp_max'], epoch)
        
        # [Added] Track Kink Loss specifically
        if 'kink' in losses:
            self.writer.add_scalar('Loss_Detail/Kink', losses['kink'], epoch)

    def log_validation_metrics(self, epoch, metrics, losses):
        """
        Log validation metrics to TensorBoard and Console.
        """
        # TensorBoard
        for key, value in metrics.items():
            # Format key (e.g., rmse -> RMSE)
            tag = f'Metrics_Ratio/{key.upper() if len(key) <= 3 else key.replace("_", " ").title()}'
            self.writer.add_scalar(tag, value, epoch)

        # Console Log
        # [Update]: Added 'Kink' to the loss breakdown for real-time monitoring
        kink_loss_str = f" Kink:{losses['kink']:.8f}" if 'kink' in losses else ""
        
        log_msg = (
            f"Epoch {epoch:5d} | "
            f"Loss: {losses['total']:.8f} (PDE:{losses['pde']:.8f} Data:{losses['data']:.8f}{kink_loss_str}) | "
            f"Val(Ratio): [RMSE:{metrics['rmse']:.4f} MAE:{metrics['mae']:.4f} SMAPE:{metrics['smape']:.2f}% "
            f"R:{metrics['r_score']:.4f}]"
        )
        logging.info(log_msg)

    def close(self):
        self.writer.close()