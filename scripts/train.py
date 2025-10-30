import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.trading_env import TradingEnv
from models.agent import TradingRLAgent, EarlyStopping
from utils.preprocessing import DataProcessor
from utils.logger import setup_logger, log_metrics

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_env(df: pd.DataFrame, config: dict) -> TradingEnv:
    """Create trading environment"""
    env = TradingEnv(
        df=df,
        window_size=config['environment']['window_size'],
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        max_position_size=config['environment']['max_position_size'],
        trading_window=config['environment']['trading_window']
    )
    return env

def setup_wandb(config: dict, run_name: str = None):
    """Initialize Weights & Biases logging"""
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project_name'],
            name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=[config['agent']['algorithm'], config['environment']['ticker']]
        )

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds for reproducibility
    seed = config['project']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup logging
    logger = setup_logger(
        name="training",
        log_dir=config['logging']['log_dir']
    )
    logger.info("Starting training process...")
    
    # Initialize W&B if enabled
    run_name = f"{config['agent']['algorithm']}_{config['environment']['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    setup_wandb(config, run_name)
    
    # Prepare data
    logger.info("Preparing data...")
    data_processor = DataProcessor(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        indicators=config['data']['indicators'],
        normalize=config['data']['normalize'],
        scaler_type=config['data']['scaler_type']
    )
    
    train_df, val_df, test_df = data_processor.prepare_data(
        train_split=config['data']['train_test_split'],
        val_split=config['data']['validation_split']
    )
    
    # Create environments
    logger.info("Creating environments...")
    
    # Training environment
    train_env = create_env(train_df, config)
    train_env = Monitor(train_env, filename=f"{config['logging']['log_dir']}/train")
    train_env = DummyVecEnv([lambda: train_env])
    
    # Validation environment
    val_env = create_env(val_df, config)
    val_env = Monitor(val_env, filename=f"{config['logging']['log_dir']}/val")
    val_env = DummyVecEnv([lambda: val_env])
    
    # Optionally normalize observations
    if args.normalize_obs:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
        val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create agent
    logger.info(f"Creating {config['agent']['algorithm']} agent...")
    
    agent_config = {
        'lstm_units': config['agent']['network']['lstm_units'],
        'lstm_layers': config['agent']['network']['lstm_layers'],
        'policy_layers': config['agent']['network']['fc_layers'],
        'value_layers': config['agent']['network']['fc_layers'],
        'dropout': config['agent']['network']['dropout'],
        'verbose': config['training']['verbose'],
        'tensorboard_log': config['logging']['log_dir']
    }
    
    # Add algorithm-specific parameters
    if config['agent']['algorithm'] == 'PPO':
        ppo_config = config['agent']['ppo'].copy()
        # Ensure learning_rate is a float
        if 'learning_rate' in ppo_config:
            ppo_config['learning_rate'] = float(ppo_config['learning_rate'])
        agent_config.update(ppo_config)
    elif config['agent']['algorithm'] == 'SAC':
        sac_config = config['agent']['sac'].copy()
        # Ensure learning_rate is a float
        if 'learning_rate' in sac_config:
            sac_config['learning_rate'] = float(sac_config['learning_rate'])
        agent_config.update(sac_config)
    
    agent = TradingRLAgent(
        env=train_env,
        algorithm=config['agent']['algorithm'],
        config=agent_config,
        device=config['project']['device']
    )
    
    # Setup callbacks
    callbacks = []
    
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Train the agent
    logger.info("Starting training...")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']}")
    
    try:
        agent.train(
            total_timesteps=config['training']['total_timesteps'],
            eval_env=val_env,
            eval_freq=config['training']['eval_freq'],
            n_eval_episodes=config['training']['n_eval_episodes'],
            save_freq=config['training']['save_freq'],
            save_path=f"models/checkpoints/{run_name}",
            callbacks=callbacks
        )
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_env = create_env(test_df, config)
    test_env = Monitor(test_env, filename=f"{config['logging']['log_dir']}/test")
    test_env = DummyVecEnv([lambda: test_env])
    
    if args.normalize_obs and 'train_env' in locals():
        test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, training=False)
        test_env.obs_rms = train_env.obs_rms  # Use training statistics
    
    try:
        test_results = agent.evaluate(
            env=test_env,
            n_episodes=config['training']['n_eval_episodes'],
            deterministic=True,
            render=args.render
        )
        
        logger.info(f"Test Results:")
        logger.info(f"  Mean Return: {test_results['mean_return']:.4f}")
        logger.info(f"  Mean Reward: {test_results['mean_reward']:.4f}")
        logger.info(f"  Mean Trades: {test_results['mean_trades']:.1f}")
        
        # Log final results to W&B
        if config['logging']['use_wandb']:
            wandb.log({
                'test/mean_return': test_results['mean_return'],
                'test/mean_reward': test_results['mean_reward'],
                'test/mean_trades': test_results['mean_trades']
            })
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        logger.info("Training completed successfully, but evaluation failed.")
        test_results = {'mean_return': 0.0, 'mean_reward': 0.0, 'mean_trades': 0.0}
    
    if config['logging']['use_wandb']:
        wandb.finish()
    
    # Save final model
    final_model_path = f"models/{run_name}_final.zip"
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save normalization statistics if used
    if args.normalize_obs:
        import pickle
        with open(f"models/{run_name}_vec_normalize.pkl", 'wb') as f:
            pickle.dump(train_env, f)
    
    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--normalize-obs",
        action="store_true",
        help="Normalize observations using running statistics"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu/cuda/auto)"
    )
    
    args = parser.parse_args()
    
    # Override device if specified
    if args.device != "auto":
        # Load config to modify it
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['project']['device'] = args.device
        # Save temporary config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            args.config = f.name
    
    # Run training
    results = main(args)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final Test Performance:")
    print(f"  Return: {results['mean_return']:.2%}")
    print(f"  Trades: {results['mean_trades']:.0f}")
    print("="*50)