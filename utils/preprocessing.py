import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict, Optional, List
import pickle
import os

class DataProcessor:
    """Data preprocessing and feature engineering for trading data"""
    
    def __init__(
        self,
        ticker: str = "AAPL",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        interval: str = "1m",
        indicators: List[str] = None,
        normalize: bool = True,
        scaler_type: str = "minmax"
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.indicators = indicators or ['sma_20', 'rsi_14', 'macd']
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.scaler = None
        
    def fetch_data(self, cache_dir: str = "data/raw") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance or load from cache"""
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{self.ticker}_{self.interval}_{self.start_date}_{self.end_date}.csv"
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print(f"Fetching data for {self.ticker} from Yahoo Finance...")
            stock = yf.Ticker(self.ticker)
            
            # For intraday data, yfinance limits to last 60 days
            if self.interval in ['1m', '2m', '5m', '15m', '30m']:
                # Fetch in chunks for longer periods
                df = self._fetch_intraday_data(stock)
            else:
                df = stock.history(start=self.start_date, end=self.end_date, interval=self.interval)
            
            # Save to cache
            df.to_csv(cache_file)
            print(f"Data saved to {cache_file}")
        
        # Clean column names
        df.columns = [col.lower() for col in df.columns]
        
        return df
    
    def _fetch_intraday_data(self, stock) -> pd.DataFrame:
        """Fetch intraday data in chunks (yfinance limitation)"""
        import datetime
        
        # For demo, fetch last 60 days of 1-minute data
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=59)
        
        df = stock.history(start=start, end=end, interval=self.interval)
        
        if df.empty:
            print("Warning: No intraday data available. Using daily data instead.")
            df = stock.history(period="1y", interval="1d")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Technical indicators using ta library
        if 'sma_20' in self.indicators:
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            
        if 'sma_50' in self.indicators:
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
        if 'ema_12' in self.indicators:
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            
        if 'ema_26' in self.indicators:
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
        if 'rsi_14' in self.indicators:
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            
        if 'macd' in self.indicators:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
        if 'bollinger_bands' in self.indicators:
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_pband'] = bb.bollinger_pband()
            
        if 'stochastic' in self.indicators:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
        if 'atr' in self.indicators:
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
        if 'obv' in self.indicators:
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
        # Price position features
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Trend features
        df['price_sma20_ratio'] = df['close'] / df['sma_20'] if 'sma_20' in self.indicators else 1
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize the dataframe using MinMax or Standard scaler"""
        
        if not self.normalize:
            return df
        
        df = df.copy()
        
        # Columns to not normalize (already in percentage or ratio form)
        exclude_cols = ['returns', 'log_returns', 'rsi_14', 'stoch_k', 'stoch_d', 
                       'bb_pband', 'price_position']
        
        # Get columns to normalize
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        
        if fit:
            # Initialize scaler
            if self.scaler_type == "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                self.scaler = StandardScaler()
            
            # Fit and transform
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        else:
            # Only transform (for test data)
            if self.scaler is not None:
                df[cols_to_normalize] = self.scaler.transform(df[cols_to_normalize])
        
        return df
    
    def prepare_data(
        self, 
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete data preparation pipeline"""
        
        # Fetch data
        df = self.fetch_data()
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Split data
        n = len(df)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        train_df = df[:train_end].copy()
        val_df = df[train_end:val_end].copy()
        test_df = df[val_end:].copy()
        
        # Normalize data
        train_df = self.normalize_data(train_df, fit=True)
        val_df = self.normalize_data(val_df, fit=False)
        test_df = self.normalize_data(test_df, fit=False)
        
        # Save scaler
        if self.scaler is not None:
            with open('data/processed/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv('data/processed/train.csv')
        val_df.to_csv('data/processed/val.csv')
        test_df.to_csv('data/processed/test.csv')
        
        print(f"Data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        window_size: int = 180,
        step_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        
        sequences = []
        targets = []
        
        for i in range(0, len(df) - window_size - 1, step_size):
            seq = df.iloc[i:i+window_size].values
            target = df.iloc[i+window_size]['close']
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    @staticmethod
    def load_scaler(path: str = 'data/processed/scaler.pkl'):
        """Load saved scaler"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class MarketDataBuffer:
    """Buffer for storing and managing market data during trading"""
    
    def __init__(self, window_size: int = 180):
        self.window_size = window_size
        self.buffer = []
        
    def add(self, data: np.ndarray):
        """Add new data point to buffer"""
        self.buffer.append(data)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def get_window(self) -> np.ndarray:
        """Get current window of data"""
        if len(self.buffer) < self.window_size:
            # Pad with zeros if not enough data
            padding = np.zeros((self.window_size - len(self.buffer), self.buffer[0].shape[0]))
            return np.vstack([padding, self.buffer])
        return np.array(self.buffer[-self.window_size:])
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data"""
        return len(self.buffer) >= self.window_size


def calculate_portfolio_metrics(
    trades: List[Dict],
    initial_balance: float,
    final_value: float
) -> Dict[str, float]:
    """Calculate portfolio performance metrics"""
    
    if not trades:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_trade_return': 0,
            'total_trades': 0
        }
    
    # Calculate returns for each trade
    trade_returns = []
    for i, trade in enumerate(trades):
        if trade['action'] == 'sell' and i > 0:
            # Find corresponding buy
            for j in range(i-1, -1, -1):
                if trades[j]['action'] == 'buy':
                    buy_price = trades[j]['price']
                    sell_price = trade['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
                    break
    
    # Calculate metrics
    total_return = (final_value - initial_balance) / initial_balance
    
    if trade_returns:
        sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252)
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        avg_trade_return = np.mean(trade_returns)
    else:
        sharpe_ratio = 0
        win_rate = 0
        avg_trade_return = 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': 0,  # Placeholder - would need price history
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'total_trades': len(trades)
    }