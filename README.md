# RL Trading Agent 🤖📈

A sophisticated Reinforcement Learning system for automated trading, implementing state-of-the-art algorithms for continuous action space trading decisions.

## 🎯 Features

- **Multiple RL Algorithms**: PPO, SAC, and DQN implementations
- **LSTM/GRU Networks**: Time-series aware neural architectures
- **Real-time Trading**: Live simulation capabilities
- **Comprehensive Backtesting**: Historical performance evaluation
- **Advanced Preprocessing**: Technical indicators and feature engineering
- **Experiment Tracking**: Integration with Weights & Biases and TensorBoard
- **Modular Design**: Clean, extensible architecture

## 📋 Requirements

- Python 3.10.11
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM recommended

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RL_Project

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (platform specific)
# For Windows:
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# pip install TA_Lib-0.4.24-cp310-cp310-win_amd64.whl

# For Linux/Mac:
# brew install ta-lib  # Mac
# sudo apt-get install ta-lib  # Ubuntu
```

### 2. Prepare Data

```bash
# Download and preprocess data
python main.py prepare --ticker AAPL --start 2023-01-01 --end 2024-01-01
```

### 3. Train Agent

```bash
# Train with default configuration
python main.py train --config configs/config.yaml

# Train with GPU
python main.py train --config configs/config.yaml --device cuda

# Resume training from checkpoint
python main.py train --resume models/checkpoints/model_50000.zip
```

### 4. Evaluate Performance

```bash
# Evaluate trained model
python main.py evaluate --model models/best/model.zip --episodes 10 --plot

# Run backtesting
python main.py backtest --model models/best/model.zip --start 2023-06-01 --end 2023-12-31
```

## 📁 Project Structure

```
RL_Project/
│
├── data/                   # Data storage
│   ├── raw/               # Raw market data
│   └── processed/         # Preprocessed features
│
├── envs/                  # Trading environments
│   ├── __init__.py
│   └── trading_env.py     # Custom Gym environment
│
├── models/                # Model architectures and agents
│   ├── agent.py          # RL agent implementation
│   └── network.py        # Neural network architectures
│
├── scripts/              # Training and evaluation scripts
│   ├── train.py         # Main training loop
│   └── evaluate.py      # Evaluation and backtesting
│
├── utils/               # Utility functions
│   ├── logger.py       # Logging utilities
│   ├── replay_buffer.py # Experience replay
│   └── preprocessing.py # Data processing
│
├── configs/            # Configuration files
│   └── config.yaml    # Main configuration
│
├── results/           # Output directory
│   ├── checkpoints/  # Model checkpoints
│   ├── logs/        # Training logs
│   └── plots/       # Visualizations
│
├── main.py          # Main entry point
├── requirements.txt # Dependencies
└── README.md       # Documentation
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Environment**: Trading window, transaction costs, position limits
- **Data**: Ticker symbols, date ranges, technical indicators
- **Agent**: Algorithm choice (PPO/SAC), network architecture
- **Training**: Learning rates, batch sizes, training duration