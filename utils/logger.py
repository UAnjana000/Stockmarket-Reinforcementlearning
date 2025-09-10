import os
import logging
import json
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tensorboard import program
import tensorflow as tf

def setup_logger(
    name: str = "rl_trading",
    log_dir: str = "results/logs",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration"""
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{timestamp}.log')
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class MetricsLogger:
    """Logger for tracking and saving metrics during training"""
    
    def __init__(
        self,
        log_dir: str = "results/metrics",
        use_wandb: bool = False,
        use_tensorboard: bool = True
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize storage
        self.metrics = {}
        self.episode_metrics = []
        
        # Setup TensorBoard
        if use_tensorboard:
            self.tb_writer = tf.summary.create_file_writer(log_dir)
        
        # CSV file for metrics
        self.csv_file = os.path.join(log_dir, 'metrics.csv')
        self.csv_initialized = False
    
    def log(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics"""
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        if step is not None:
            metrics['step'] = step
        
        # Store in memory
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log to W&B
        if self.use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard and step is not None:
            with self.tb_writer.as_default():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tf.summary.scalar(key, value, step=step)
        
        # Log to CSV
        self._log_to_csv(metrics)
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """Log episode-level data"""
        
        self.episode_metrics.append(episode_data)
        
        # Calculate running statistics
        if len(self.episode_metrics) >= 10:
            recent_returns = [ep['return'] for ep in self.episode_metrics[-10:]]
            recent_trades = [ep['trades'] for ep in self.episode_metrics[-10:]]
            
            stats = {
                'mean_return_10ep': np.mean(recent_returns),
                'std_return_10ep': np.std(recent_returns),
                'mean_trades_10ep': np.mean(recent_trades)
            }
            
            self.log(stats, step=len(self.episode_metrics))
    
    def _log_to_csv(self, metrics: Dict[str, Any]):
        """Save metrics to CSV file"""
        
        # Initialize CSV with headers
        if not self.csv_initialized:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
            self.csv_initialized = True
        
        # Append metrics
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
    
    def plot_metrics(self, keys: list = None, save_path: str = None):
        """Plot logged metrics"""
        
        if keys is None:
            keys = [k for k in self.metrics.keys() 
                   if isinstance(self.metrics[k][0], (int, float))]
        
        n_plots = len(keys)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, key in zip(axes, keys):
            values = self.metrics[key]
            ax.plot(values)
            ax.set_title(key)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_metrics(self, filename: str = None):
        """Save all metrics to JSON file"""
        
        if filename is None:
            filename = os.path.join(
                self.log_dir,
                f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
        
        with open(filename, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'episode_metrics': self.episode_metrics
            }, f, indent=2, default=str)
    
    def load_metrics(self, filename: str):
        """Load metrics from JSON file"""
        
        with open(filename, 'r') as f:
            data = json.load(f)
            self.metrics = data['metrics']
            self.episode_metrics = data['episode_metrics']

def log_metrics(metrics: Dict[str, float], step: int = None, logger_name: str = "metrics"):
    """Quick function to log metrics"""
    
    logger = logging.getLogger(logger_name)
    
    message = f"Step {step}: " if step else ""
    message += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    logger.info(message)

def launch_tensorboard(log_dir: str = "results/logs", port: int = 6006):
    """Launch TensorBoard server"""
    
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port)])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    return tb

class ProgressTracker:
    """Track and display training progress"""
    
    def __init__(self, total_steps: int, update_freq: int = 100):
        self.total_steps = total_steps
        self.update_freq = update_freq
        self.current_step = 0
        self.start_time = datetime.now()
        
    def update(self, step: int, metrics: Dict[str, float] = None):
        """Update progress"""
        
        self.current_step = step
        
        if step % self.update_freq == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            progress = step / self.total_steps
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            print(f"\rProgress: {progress:.1%} | "
                  f"Step: {step}/{self.total_steps} | "
                  f"ETA: {eta/60:.1f} min", end="")
            
            if metrics:
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(f" | {metrics_str}", end="")

class ExperimentTracker:
    """Track experiments and hyperparameters"""
    
    def __init__(self, experiment_name: str, base_dir: str = "results/experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create experiment metadata
        self.metadata = {
            'name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'status': 'running'
        }
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        
        self.metadata['hyperparameters'] = hyperparams
        
        # Save to file
        with open(os.path.join(self.experiment_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def log_results(self, results: Dict[str, Any]):
        """Log final results"""
        
        self.metadata['results'] = results
        self.metadata['completed_at'] = datetime.now().isoformat()
        self.metadata['status'] = 'completed'
        
        # Save to file
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def save_metadata(self):
        """Save experiment metadata"""
        
        with open(os.path.join(self.experiment_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @staticmethod
    def list_experiments(base_dir: str = "results/experiments"):
        """List all experiments"""
        
        experiments = []
        
        if os.path.exists(base_dir):
            for exp_dir in os.listdir(base_dir):
                metadata_path = os.path.join(base_dir, exp_dir, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        experiments.append(metadata)
        
        return experiments
    
    @staticmethod
    def compare_experiments(experiment_names: list, base_dir: str = "results/experiments"):
        """Compare multiple experiments"""
        
        comparison = {}
        
        for name in experiment_names:
            results_path = os.path.join(base_dir, name, 'results.json')
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    comparison[name] = results
        
        return comparison