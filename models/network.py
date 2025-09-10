import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

class LSTMFeatureExtractor(nn.Module):
    """LSTM-based feature extractor for time series data"""
    
    def __init__(
        self,
        input_dim: int,
        lstm_units: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = lstm_units
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, lstm_units)
        
        return self.dropout(last_hidden)

class GRUFeatureExtractor(nn.Module):
    """GRU-based feature extractor for time series data"""
    
    def __init__(
        self,
        input_dim: int,
        gru_units: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_units,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = gru_units
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.dropout(last_hidden)

class AttentionLayer(nn.Module):
    """Self-attention layer for sequence data"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return attn_out

class TradingActorNetwork(nn.Module):
    """Actor network for continuous action space"""
    
    def __init__(
        self,
        market_input_dim: int,
        portfolio_input_dim: int,
        window_size: int,
        lstm_units: int = 128,
        lstm_layers: int = 2,
        fc_layers: list = [256, 128],
        dropout: float = 0.2,
        feature_extractor: str = 'lstm'
    ):
        super().__init__()
        
        # Feature extractor for market data
        if feature_extractor == 'lstm':
            self.market_extractor = LSTMFeatureExtractor(
                market_input_dim, lstm_units, lstm_layers, dropout
            )
        elif feature_extractor == 'gru':
            self.market_extractor = GRUFeatureExtractor(
                market_input_dim, lstm_units, lstm_layers, dropout
            )
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        
        # Combine market features with portfolio state
        combined_dim = self.market_extractor.output_dim + portfolio_input_dim
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_dim = combined_dim
        
        for fc_dim in fc_layers:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            prev_dim = fc_dim
        
        # Output layer for action (continuous: -1 to 1)
        self.action_mean = nn.Linear(prev_dim, 1)
        self.action_log_std = nn.Linear(prev_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        market_data: torch.Tensor, 
        portfolio_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features from market data
        market_features = self.market_extractor(market_data)
        
        # Concatenate with portfolio state
        x = torch.cat([market_features, portfolio_state], dim=-1)
        
        # Pass through FC layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        # Get action distribution parameters
        action_mean = torch.tanh(self.action_mean(x))  # [-1, 1]
        action_log_std = self.action_log_std(x)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        return action_mean, action_log_std

class TradingCriticNetwork(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(
        self,
        market_input_dim: int,
        portfolio_input_dim: int,
        window_size: int,
        lstm_units: int = 128,
        lstm_layers: int = 2,
        fc_layers: list = [256, 128],
        dropout: float = 0.2,
        feature_extractor: str = 'lstm'
    ):
        super().__init__()
        
        # Feature extractor for market data
        if feature_extractor == 'lstm':
            self.market_extractor = LSTMFeatureExtractor(
                market_input_dim, lstm_units, lstm_layers, dropout
            )
        elif feature_extractor == 'gru':
            self.market_extractor = GRUFeatureExtractor(
                market_input_dim, lstm_units, lstm_layers, dropout
            )
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        
        # Combine market features with portfolio state
        combined_dim = self.market_extractor.output_dim + portfolio_input_dim
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_dim = combined_dim
        
        for fc_dim in fc_layers:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            prev_dim = fc_dim
        
        # Output layer for value
        self.value_head = nn.Linear(prev_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        market_data: torch.Tensor, 
        portfolio_state: torch.Tensor
    ) -> torch.Tensor:
        # Extract features from market data
        market_features = self.market_extractor(market_data)
        
        # Concatenate with portfolio state
        x = torch.cat([market_features, portfolio_state], dim=-1)
        
        # Pass through FC layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        # Get value estimate
        value = self.value_head(x)
        
        return value

class TradingQNetwork(nn.Module):
    """Q-Network for DQN with discrete actions"""
    
    def __init__(
        self,
        market_input_dim: int,
        portfolio_input_dim: int,
        window_size: int,
        n_actions: int = 3,  # Hold, Buy, Sell
        lstm_units: int = 128,
        lstm_layers: int = 2,
        fc_layers: list = [256, 128],
        dropout: float = 0.2,
        feature_extractor: str = 'lstm'
    ):
        super().__init__()
        
        # Feature extractor for market data
        if feature_extractor == 'lstm':
            self.market_extractor = LSTMFeatureExtractor(
                market_input_dim, lstm_units, lstm_layers, dropout
            )
        elif feature_extractor == 'gru':
            self.market_extractor = GRUFeatureExtractor(
                market_input_dim, lstm_units, lstm_layers, dropout
            )
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        
        # Combine market features with portfolio state
        combined_dim = self.market_extractor.output_dim + portfolio_input_dim
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_dim = combined_dim
        
        for fc_dim in fc_layers:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            prev_dim = fc_dim
        
        # Dueling DQN architecture
        self.value_stream = nn.Linear(prev_dim, 1)
        self.advantage_stream = nn.Linear(prev_dim, n_actions)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        market_data: torch.Tensor, 
        portfolio_state: torch.Tensor
    ) -> torch.Tensor:
        # Extract features from market data
        market_features = self.market_extractor(market_data)
        
        # Concatenate with portfolio state
        x = torch.cat([market_features, portfolio_state], dim=-1)
        
        # Pass through FC layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        # Dueling architecture
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values

class CustomActorCriticPolicy(nn.Module):
    """Custom Actor-Critic Policy for Stable Baselines3 integration"""
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        *args,
        **kwargs
    ):
        super().__init__()
        
        # This is a placeholder for SB3 integration
        # The actual implementation would need to inherit from SB3's BasePolicy
        pass