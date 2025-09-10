import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Tuple, Optional, List
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

from models.network import TradingActorNetwork, TradingCriticNetwork

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for Stable Baselines3"""
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        lstm_units: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        # Calculate the actual features dimension
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        if isinstance(observation_space, spaces.Dict):
            market_shape = observation_space['market_data'].shape
            portfolio_shape = observation_space['portfolio_state'].shape
            
            self.market_input_dim = market_shape[1]  # features per timestep
            self.window_size = market_shape[0]
            self.portfolio_dim = portfolio_shape[0]
        
        # LSTM for market data
        self.lstm = nn.LSTM(
            input_size=self.market_input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Update features_dim to match actual output
        self._features_dim = lstm_units + self.portfolio_dim
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        market_data = observations['market_data']
        portfolio_state = observations['portfolio_state']
        
        # Process market data through LSTM
        lstm_out, (hidden, cell) = self.lstm(market_data)
        market_features = hidden[-1]  # Last hidden state
        
        # Concatenate with portfolio state
        combined = torch.cat([market_features, portfolio_state], dim=-1)
        
        return combined

class TradingRLAgent:
    """Main RL Agent for trading"""
    
    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        config: Dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.env = env
        self.algorithm = algorithm
        self.config = config or {}
        self.device = device
        self.model = None
        
        # Initialize the model based on algorithm
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the RL model"""
        
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(
                lstm_units=self.config.get('lstm_units', 128),
                lstm_layers=self.config.get('lstm_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            ),
            net_arch=dict(
                pi=self.config.get('policy_layers', [256, 128]),
                vf=self.config.get('value_layers', [256, 128])
            ),
            activation_fn=nn.ReLU
        )
        
        if self.algorithm == "PPO":
            self.model = PPO(
                policy="MultiInputPolicy",
                env=self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_steps=self.config.get('n_steps', 2048),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                ent_coef=self.config.get('ent_coef', 0.01),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs,
                verbose=self.config.get('verbose', 1),
                tensorboard_log=self.config.get('tensorboard_log', './results/logs'),
                device=self.device
            )
            
        elif self.algorithm == "SAC":
            self.model = SAC(
                policy="MultiInputPolicy",
                env=self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                buffer_size=self.config.get('buffer_size', 100000),
                learning_starts=self.config.get('learning_starts', 1000),
                batch_size=self.config.get('batch_size', 256),
                tau=self.config.get('tau', 0.005),
                gamma=self.config.get('gamma', 0.99),
                train_freq=self.config.get('train_freq', 1),
                gradient_steps=self.config.get('gradient_steps', 1),
                policy_kwargs=policy_kwargs,
                verbose=self.config.get('verbose', 1),
                tensorboard_log=self.config.get('tensorboard_log', './results/logs'),
                device=self.device
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_freq: int = 50000,
        save_path: str = "models/checkpoints",
        callbacks: List[BaseCallback] = None
    ):
        """Train the agent"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Create callbacks
        callbacks_list = callbacks or []
        
        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{save_path}/best",
                log_path=f"{save_path}/eval",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks_list.append(eval_callback)
        
        # Add custom callbacks
        callbacks_list.append(TradingMetricsCallback(save_freq=save_freq))
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks_list,
            progress_bar=True
        )
        
        # Save final model
        self.save(f"{save_path}/final_model")
        
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, ...]]]:
        """Predict action given observation"""
        
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        return action, _states
    
    def save(self, path: str):
        """Save the model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a saved model"""
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env, device=self.device)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=self.env, device=self.device)
        print(f"Model loaded from {path}")
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """Evaluate the agent"""
        
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Collect episode metrics from info
            if 'portfolio_value' in info:
                initial_value = env.initial_balance
                final_value = info['portfolio_value']
                episode_return = (final_value - initial_value) / initial_value
                
                episode_metrics.append({
                    'return': episode_return,
                    'final_value': final_value,
                    'total_trades': info.get('total_trades', 0)
                })
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_return': np.mean([m['return'] for m in episode_metrics]) if episode_metrics else 0,
            'mean_trades': np.mean([m['total_trades'] for m in episode_metrics]) if episode_metrics else 0
        }
        
        return results


class TradingMetricsCallback(BaseCallback):
    """Custom callback for tracking trading-specific metrics"""
    
    def __init__(self, save_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_trades = []
        
    def _on_step(self) -> bool:
        # Log metrics if episode ended
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            if 'portfolio_value' in info:
                initial_value = 10000  # Should get from env
                final_value = info['portfolio_value']
                episode_return = (final_value - initial_value) / initial_value
                
                self.episode_returns.append(episode_return)
                self.episode_trades.append(info.get('total_trades', 0))
                
                # Log to tensorboard
                self.logger.record("trading/episode_return", episode_return)
                self.logger.record("trading/total_trades", info.get('total_trades', 0))
                self.logger.record("trading/final_value", final_value)
        
        # Save checkpoint
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"models/checkpoints/model_{self.n_calls}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.episode_returns:
            mean_return = np.mean(self.episode_returns)
            std_return = np.std(self.episode_returns)
            
            print(f"\nTraining completed!")
            print(f"Mean episode return: {mean_return:.4f} ± {std_return:.4f}")
            print(f"Mean trades per episode: {np.mean(self.episode_trades):.1f}")


class EarlyStopping(BaseCallback):
    """Early stopping callback"""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.counter = 0
        
    def _on_step(self) -> bool:
        # Check if we have evaluation results
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            
            if mean_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = mean_reward
                self.counter = 0
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.4f}")
            else:
                self.counter += 1
                
            if self.counter >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered after {self.counter} episodes without improvement")
                return False
        
        return True