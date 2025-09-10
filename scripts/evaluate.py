import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.trading_env import TradingEnv
from models.agent import TradingRLAgent
from utils.preprocessing import DataProcessor, calculate_portfolio_metrics
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class TradingEvaluator:
    """Comprehensive evaluation and backtesting for trading agents"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        
    def load_agent(self, model_path: str, env) -> TradingRLAgent:
        """Load a trained agent"""
        agent = TradingRLAgent(
            env=env,
            algorithm=self.config['agent']['algorithm'],
            config={},
            device=self.config['project']['device']
        )
        agent.load(model_path)
        return agent
    
    def evaluate_agent(
        self,
        agent: TradingRLAgent,
        env: TradingEnv,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """Evaluate agent performance"""
        
        all_trades = []
        all_portfolio_values = []
        all_returns = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            
            episode_portfolio_values = [env.envs[0].initial_balance]
            
            while not (done or truncated):
                action, _ = agent.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                
                if isinstance(info, list):
                    info = info[0]
                
                episode_portfolio_values.append(info['portfolio_value'])
            
            # Get final metrics
            final_value = episode_portfolio_values[-1]
            initial_value = episode_portfolio_values[0]
            episode_return = (final_value - initial_value) / initial_value
            
            all_returns.append(episode_return)
            all_portfolio_values.append(episode_portfolio_values)
            
            if hasattr(env.envs[0], 'episode_trades'):
                all_trades.extend(env.envs[0].episode_trades)
        
        # Calculate aggregate metrics
        metrics = {
            'mean_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'sharpe_ratio': np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252),
            'max_return': np.max(all_returns),
            'min_return': np.min(all_returns),
            'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns),
            'total_trades': len(all_trades),
            'avg_trades_per_episode': len(all_trades) / n_episodes
        }
        
        # Calculate max drawdown
        for values in all_portfolio_values:
            drawdown = self._calculate_max_drawdown(values)
            if 'max_drawdown' not in metrics or drawdown > metrics['max_drawdown']:
                metrics['max_drawdown'] = drawdown
        
        return metrics, all_portfolio_values, all_trades
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        values = np.array(values)
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        return abs(np.min(drawdown))
    
    def evaluate_baseline_strategies(self, df: pd.DataFrame) -> Dict:
        """Evaluate baseline trading strategies"""
        
        baselines = {}
        
        # Buy and Hold
        initial_price = df.iloc[0]['close']
        final_price = df.iloc[-1]['close']
        baselines['buy_and_hold'] = {
            'return': (final_price - initial_price) / initial_price,
            'trades': 1
        }
        
        # Random Trading
        np.random.seed(42)
        random_returns = []
        for _ in range(10):
            n_trades = np.random.randint(10, 50)
            trade_points = np.random.choice(len(df), n_trades, replace=False)
            trade_points.sort()
            
            position = 0
            cash = 10000
            
            for i in range(0, len(trade_points), 2):
                if i+1 < len(trade_points):
                    buy_price = df.iloc[trade_points[i]]['close']
                    sell_price = df.iloc[trade_points[i+1]]['close']
                    position = cash / buy_price
                    cash = position * sell_price
            
            random_returns.append((cash - 10000) / 10000)
        
        baselines['random'] = {
            'return': np.mean(random_returns),
            'trades': 30
        }
        
        # SMA Crossover
        df['sma_fast'] = df['close'].rolling(20).mean()
        df['sma_slow'] = df['close'].rolling(50).mean()
        
        signals = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
        signal_changes = np.diff(signals)
        trade_points = np.where(signal_changes != 0)[0]
        
        cash = 10000
        position = 0
        
        for i, point in enumerate(trade_points):
            if signals[point] == 1 and cash > 0:  # Buy signal
                position = cash / df.iloc[point]['close']
                cash = 0
            elif signals[point] == -1 and position > 0:  # Sell signal
                cash = position * df.iloc[point]['close']
                position = 0
        
        # Close any open position
        if position > 0:
            cash = position * df.iloc[-1]['close']
        
        baselines['sma_crossover'] = {
            'return': (cash - 10000) / 10000,
            'trades': len(trade_points)
        }
        
        return baselines
    
    def plot_results(
        self,
        portfolio_values: List[List[float]],
        trades: List[Dict],
        df: pd.DataFrame,
        save_path: str = "results/plots"
    ):
        """Create visualization plots"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Portfolio value over time
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot portfolio values
        for i, values in enumerate(portfolio_values[:5]):  # Plot first 5 episodes
            axes[0].plot(values, alpha=0.7, label=f'Episode {i+1}')
        
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Price chart with trade markers
        price_subset = df['close'].iloc[:len(portfolio_values[0])]
        axes[1].plot(price_subset.values, color='black', alpha=0.7, label='Price')
        
        # Add trade markers
        buy_steps = [t['step'] for t in trades if t['action'] == 'buy'][:50]
        sell_steps = [t['step'] for t in trades if t['action'] == 'sell'][:50]
        
        if buy_steps:
            buy_prices = [df.iloc[step]['close'] for step in buy_steps if step < len(df)]
            axes[1].scatter(buy_steps[:len(buy_prices)], buy_prices, 
                          color='green', marker='^', s=100, label='Buy', zorder=5)
        
        if sell_steps:
            sell_prices = [df.iloc[step]['close'] for step in sell_steps if step < len(df)]
            axes[1].scatter(sell_steps[:len(sell_prices)], sell_prices,
                          color='red', marker='v', s=100, label='Sell', zorder=5)
        
        axes[1].set_title('Trading Signals', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Returns distribution
        returns = [(values[-1] - values[0]) / values[0] for values in portfolio_values]
        axes[2].hist(returns, bins=20, edgecolor='black', alpha=0.7)
        axes[2].axvline(np.mean(returns), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(returns):.2%}')
        axes[2].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Return')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Performance comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if hasattr(self, 'baseline_results'):
            strategies = list(self.baseline_results.keys()) + ['RL Agent']
            returns = [self.baseline_results[s]['return'] for s in self.baseline_results.keys()]
            returns.append(np.mean([r for r in returns]))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(strategies, returns, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, ret in zip(bars, returns):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{ret:.2%}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Return (%)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(min(returns) * 1.2 if min(returns) < 0 else 0, max(returns) * 1.2)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/strategy_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = TradingEvaluator(config)
    
    # Prepare test data
    print("Loading test data...")
    data_processor = DataProcessor(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        indicators=config['data']['indicators'],
        normalize=config['data']['normalize'],
        scaler_type=config['data']['scaler_type']
    )
    
    # Load processed data
    test_df = pd.read_csv('data/processed/test.csv', index_col=0, parse_dates=True)
    
    # Create test environment
    test_env = TradingEnv(
        df=test_df,
        window_size=config['environment']['window_size'],
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        max_position_size=config['environment']['max_position_size'],
        trading_window=config['environment']['trading_window']
    )
    test_env = DummyVecEnv([lambda: test_env])
    
    # Load normalization if used
    if args.normalize_obs:
        with open(args.norm_stats, 'rb') as f:
            vec_normalize = pickle.load(f)
            test_env = VecNormalize(test_env, training=False)
            test_env.obs_rms = vec_normalize.obs_rms
    
    # Load agent
    print(f"Loading model from {args.model}...")
    agent = evaluator.load_agent(args.model, test_env)
    
    # Evaluate agent
    print("Evaluating agent...")
    metrics, portfolio_values, trades = evaluator.evaluate_agent(
        agent, test_env, n_episodes=args.n_episodes, deterministic=not args.stochastic
    )
    
    # Evaluate baselines
    print("Evaluating baseline strategies...")
    evaluator.baseline_results = evaluator.evaluate_baseline_strategies(test_df)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nRL Agent Performance:")
    print(f"  Mean Return: {metrics['mean_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Total Trades: {metrics['total_trades']}")
    
    print("\nBaseline Strategies:")
    for name, results in evaluator.baseline_results.items():
        print(f"  {name}: Return={results['return']:.2%}, Trades={results['trades']}")
    
    # Create plots
    if args.plot:
        print("\nGenerating plots...")
        evaluator.plot_results(portfolio_values, trades, test_df)
    
    # Save results
    if args.save_results:
        results_dict = {
            'agent_metrics': metrics,
            'baselines': evaluator.baseline_results,
            'config': config
        }
        
        import json
        with open(f'results/evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print("\nResults saved to results/ directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Trading Agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--normalize-obs", action="store_true", help="Use normalized observations")
    parser.add_argument("--norm-stats", type=str, help="Path to normalization statistics")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--save-results", action="store_true", help="Save evaluation results")
    
    args = parser.parse_args()
    main(args)