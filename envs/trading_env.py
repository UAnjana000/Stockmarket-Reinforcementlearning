import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TradingEnv(gym.Env):
    """Custom Trading Environment for RL Agent"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 180,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        trading_window: int = 720,
        reward_scaling: float = 1e-4
    ):
        super().__init__()
        
        self.df = df
        # Columns used as model inputs (exclude execution-only columns)
        self.feature_columns = [col for col in df.columns if col != 'close_price']
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.trading_window = trading_window
        self.reward_scaling = reward_scaling
        
        # Episode variables
        self.current_step = 0
        self.start_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.avg_cost = 0
        self.total_trades = 0
        self.episode_trades = []
        
        # Define action and observation spaces
        # Actions: 0=Hold, 1=Buy, 2=Sell (discrete)
        # Or continuous: [-1, 1] where negative=sell, positive=buy, 0=hold
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: [features + portfolio state], exclude execution-only columns
        n_features = len(self.feature_columns)
        n_portfolio_features = 5  # cash_ratio, position_ratio, unrealized_pnl, time_remaining, trades_count
        obs_shape = (window_size, n_features) + (n_portfolio_features,)
        
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(low=-np.inf, high=np.inf, 
                                     shape=(window_size, n_features), dtype=np.float32),
            'portfolio_state': spaces.Box(low=-np.inf, high=np.inf, 
                                         shape=(n_portfolio_features,), dtype=np.float32)
        })
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Reset episode variables
        self.balance = self.initial_balance
        self.shares_held = 0
        self.avg_cost = 0
        self.total_trades = 0
        self.episode_trades = []
        
        # Random start point ensuring enough room for trading window
        max_start = len(self.df) - self.trading_window - self.window_size - 1
        # Ensure we have enough data for at least one step
        if max_start <= self.window_size:
            max_start = self.window_size + 1
        self.start_step = np.random.randint(self.window_size, max_start)
        self.current_step = self.start_step
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        # Execute action
        action_value = float(action[0])
        current_price = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value()
        
        # Process action with more sensitive thresholds
        if action_value > 0.0:  # Buy on any positive signal
            self._execute_buy(action_value, current_price)
            if self.current_step < self.start_step + 10:  # Debug first 10 steps
                print(f"BUY: action={action_value:.4f}, price={current_price:.2f}")
        elif action_value < 0.0:  # Sell on any negative signal
            self._execute_sell(abs(action_value), current_price)
            if self.current_step < self.start_step + 10:  # Debug first 10 steps
                print(f"SELL: action={action_value:.4f}, price={current_price:.2f}")
        # else: Hold
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        new_portfolio_value = self._get_portfolio_value()
        step_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Reward function combining profit and risk
        reward = step_return * self.reward_scaling
        
        # Add risk penalty for large positions
        position_ratio = (self.shares_held * current_price) / new_portfolio_value if new_portfolio_value > 0 else 0
        if position_ratio > 0.8:  # Penalize over-concentration
            reward -= 0.001 * self.reward_scaling
        
        # Check if episode is done
        done = False
        truncated = False
        
        if self.current_step >= self.start_step + self.trading_window:
            truncated = True
        elif self.current_step >= len(self.df) - 1:
            done = True
        elif new_portfolio_value <= self.initial_balance * 0.5:  # Stop loss at 50%
            done = True
            reward -= 1.0  # Large penalty for major loss
        
        info = {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'total_trades': self.total_trades,
            'step_return': step_return,
            'action_taken': 'buy' if action_value > 0.0 else 'sell' if action_value < 0.0 else 'hold',
            'action_value': action_value
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_buy(self, fraction: float, current_price: float):
        """Execute buy order"""
        # Skip if price is invalid (non-positive)
        if current_price <= 0 or not np.isfinite(current_price):
            return
        # Calculate maximum shares we can buy
        max_spend = self.balance * min(fraction, self.max_position_size)
        shares_to_buy = int(max_spend / (current_price * (1 + self.transaction_cost)))
        # Ensure at least 1 share if affordable and positive intent
        if shares_to_buy == 0 and fraction > 0 and self.balance >= current_price * (1 + self.transaction_cost):
            shares_to_buy = 1
        
        if shares_to_buy > 0:
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)
            if cost <= self.balance:
                # Update portfolio
                total_shares = self.shares_held + shares_to_buy
                if self.shares_held > 0:
                    self.avg_cost = ((self.avg_cost * self.shares_held) + 
                                   (current_price * shares_to_buy)) / total_shares
                else:
                    self.avg_cost = current_price
                
                self.shares_held = total_shares
                self.balance -= cost
                self.total_trades += 1
                
                self.episode_trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost
                })
    
    def _execute_sell(self, fraction: float, current_price: float):
        """Execute sell order"""
        if self.shares_held > 0:
            if current_price <= 0 or not np.isfinite(current_price):
                return
            shares_to_sell = int(self.shares_held * min(fraction, 1.0))
            if shares_to_sell == 0 and self.shares_held > 0:
                shares_to_sell = 1
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                
                self.shares_held -= shares_to_sell
                self.balance += revenue
                self.total_trades += 1
                
                self.episode_trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue
                })
                
                if self.shares_held == 0:
                    self.avg_cost = 0
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        # Ensure current_step is within bounds
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        current_price = self._get_current_price()
        return self.balance + (self.shares_held * current_price)

    def _get_current_price(self) -> float:
        """Get the current unnormalized price if available, else fallback to 'close'."""
        if self.current_step >= len(self.df):
            idx = len(self.df) - 1
        else:
            idx = self.current_step
        row = self.df.iloc[idx]
        if 'close_price' in self.df.columns:
            return float(row['close_price'])
        return float(row['close'])
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        # Get market data window
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        market_window = self.df[self.feature_columns].iloc[start_idx:end_idx]
        
        # Pad if necessary
        if len(market_window) < self.window_size:
            padding_size = self.window_size - len(market_window)
            padding = pd.DataFrame(
                np.zeros((padding_size, len(market_window.columns))),
                columns=market_window.columns
            )
            market_window = pd.concat([padding, market_window], ignore_index=True)
        
        # Portfolio state features
        # Ensure current_step is within bounds
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Cash ratio
            (self.shares_held * current_price) / portfolio_value if portfolio_value > 0 else 0,  # Position ratio
            ((current_price - self.avg_cost) / self.avg_cost) if self.avg_cost > 0 else 0,  # Unrealized P&L
            1 - ((self.current_step - self.start_step) / self.trading_window),  # Time remaining
            self.total_trades / 100.0  # Normalized trade count
        ], dtype=np.float32)
        
        return {
            'market_data': market_window.values.astype(np.float32),
            'portfolio_state': portfolio_state
        }
    
    def render(self, mode='human'):
        """Render the environment"""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        profit = portfolio_value - self.initial_balance
        
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares_held}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Profit/Loss: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)")
        print(f"Total Trades: {self.total_trades}")
        print("-" * 50)