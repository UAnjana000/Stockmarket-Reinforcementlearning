#!/usr/bin/env python3
"""
RL Trading Agent - Main Entry Point
A reinforcement learning system for automated trading
"""

import argparse
import sys
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="RL Trading Agent - Reinforcement Learning for Automated Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data
  python main.py prepare --ticker AAPL --start 2023-01-01 --end 2024-01-01
  
  # Train a new agent
  python main.py train --config configs/config.yaml
  
  # Evaluate trained model
  python main.py evaluate --model models/best/model.zip --plot
  
  # Run live trading simulation
  python main.py live --model models/best/model.zip --ticker AAPL
  
  # Backtest on historical data
  python main.py backtest --model models/best/model.zip --start 2023-06-01 --end 2023-12-31
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data preparation command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare and preprocess data')
    prepare_parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    prepare_parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    prepare_parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    prepare_parser.add_argument('--interval', type=str, default='1d', help='Data interval (1m, 5m, 1h, 1d)')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    eval_parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # Live trading simulation
    live_parser = subparsers.add_parser('live', help='Run live trading simulation')
    live_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    live_parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    live_parser.add_argument('--interval', type=str, default='1m', help='Update interval')
    live_parser.add_argument('--duration', type=int, default=720, help='Duration in minutes')
    
    # Backtesting command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    backtest_parser.add_argument('--start', type=str, required=True, help='Start date')
    backtest_parser.add_argument('--end', type=str, required=True, help='End date')
    backtest_parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    
    # Hyperparameter optimization
    optimize_parser = subparsers.add_parser('optimize', help='Optimize hyperparameters')
    optimize_parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    optimize_parser.add_argument('--config', type=str, default='configs/config.yaml', help='Base configuration')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        from utils.preprocessing import DataProcessor
        
        print(f"Preparing data for {args.ticker}...")
        processor = DataProcessor(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval
        )
        
        train_df, val_df, test_df = processor.prepare_data()
        print(f"Data prepared successfully!")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
    elif args.command == 'train':
        from scripts.train import main as train_main
        
        print("Starting training...")
        train_main(args)
        
    elif args.command == 'evaluate':
        from scripts.evaluate import main as eval_main
        
        print("Starting evaluation...")
        eval_main(args)
        
    elif args.command == 'live':
        print("Live trading simulation...")
        run_live_simulation(args)
        
    elif args.command == 'backtest':
        print("Running backtest...")
        run_backtest(args)
        
    elif args.command == 'optimize':
        print("Optimizing hyperparameters...")
        run_hyperparameter_optimization(args)
        
    else:
        parser.print_help()

def run_live_simulation(args):
    """Run live trading simulation"""
    
    import time
    import yfinance as yf
    from envs.trading_env import TradingEnv
    from models.agent import TradingRLAgent
    from utils.preprocessing import DataProcessor, MarketDataBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print(f"Starting live simulation for {args.ticker}...")
    print(f"Duration: {args.duration} minutes")
    print(f"Update interval: {args.interval}")
    
    # Load model
    env = DummyVecEnv([lambda: TradingEnv(df=pd.DataFrame(), window_size=180)])
    agent = TradingRLAgent(env=env, algorithm='PPO')
    agent.load(args.model)
    
    # Initialize data buffer
    buffer = MarketDataBuffer(window_size=180)
    
    # Simulation loop
    start_time = datetime.now()
    portfolio_value = 10000
    
    while (datetime.now() - start_time).seconds < args.duration * 60:
        try:
            # Fetch latest data
            ticker = yf.Ticker(args.ticker)
            hist = ticker.history(period='1d', interval=args.interval)
            
            if not hist.empty:
                latest = hist.iloc[-1]
                
                # Add to buffer
                buffer.add(latest.values)
                
                if buffer.is_ready():
                    # Make trading decision
                    obs = {
                        'market_data': buffer.get_window(),
                        'portfolio_state': np.array([1.0, 0.0, 0.0, 0.5, 0.0])
                    }
                    
                    action, _ = agent.predict(obs)
                    
                    # Display decision
                    print(f"\n{datetime.now().strftime('%H:%M:%S')}")
                    print(f"Price: ${latest['Close']:.2f}")
                    print(f"Action: {['Hold', 'Buy', 'Sell'][int(action[0] * 2 + 1)]}")
                    print(f"Portfolio: ${portfolio_value:.2f}")
            
            # Wait for next update
            time.sleep(60 if args.interval == '1m' else 300)
            
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
    
    print("\nSimulation completed!")

def run_backtest(args):
    """Run backtesting on historical data"""
    
    import pandas as pd
    import yfinance as yf
    from envs.trading_env import TradingEnv
    from models.agent import TradingRLAgent
    from utils.preprocessing import DataProcessor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print(f"Running backtest from {args.start} to {args.end}...")
    
    # Fetch historical data
    processor = DataProcessor(
        ticker='SPY',  # Default to SPY for backtesting
        start_date=args.start,
        end_date=args.end,
        interval='1d'
    )
    
    df = processor.fetch_data()
    df = processor.add_technical_indicators(df)
    df = processor.normalize_data(df)
    
    # Create environment
    env = TradingEnv(
        df=df,
        initial_balance=args.initial_balance,
        trading_window=len(df)
    )
    env = DummyVecEnv([lambda: env])
    
    # Load model
    agent = TradingRLAgent(env=env, algorithm='PPO')
    agent.load(args.model)
    
    # Run backtest
    obs, _ = env.reset()
    done = False
    portfolio_values = [args.initial_balance]
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if isinstance(info, list):
            info = info[0]
        
        portfolio_values.append(info['portfolio_value'])
        done = done[0] if isinstance(done, (list, tuple)) else done
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - args.initial_balance) / args.initial_balance
    
    print(f"\nBacktest Results:")
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Value: ${max(portfolio_values):,.2f}")
    print(f"Min Value: ${min(portfolio_values):,.2f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.show()

def run_hyperparameter_optimization(args):
    """Run hyperparameter optimization"""
    
    print(f"Running {args.trials} optimization trials...")
    print("This feature is under development...")
    print("\nSuggested hyperparameter ranges:")
    print("- Learning rate: [1e-5, 1e-3]")
    print("- Batch size: [32, 256]")
    print("- LSTM units: [64, 256]")
    print("- Dropout: [0.1, 0.5]")
    print("- Gamma: [0.95, 0.99]")

if __name__ == "__main__":
    main()