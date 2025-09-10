import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
import random

class ReplayBuffer:
    """Basic replay buffer for experience replay"""
    
    def __init__(
        self,
        buffer_size: int = 100000,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        
    def add(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool
    ):
        """Add experience to buffer"""
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Separate and stack experiences
        states = {
            'market_data': torch.FloatTensor(
                np.stack([e['state']['market_data'] for e in batch])
            ).to(self.device),
            'portfolio_state': torch.FloatTensor(
                np.stack([e['state']['portfolio_state'] for e in batch])
            ).to(self.device)
        }
        
        actions = torch.FloatTensor(
            np.stack([e['action'] for e in batch])
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            [e['reward'] for e in batch]
        ).to(self.device)
        
        next_states = {
            'market_data': torch.FloatTensor(
                np.stack([e['next_state']['market_data'] for e in batch])
            ).to(self.device),
            'portfolio_state': torch.FloatTensor(
                np.stack([e['next_state']['portfolio_state'] for e in batch])
            ).to(self.device)
        }
        
        dones = torch.FloatTensor(
            [e['done'] for e in batch]
        ).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples"""
        return len(self.buffer) >= self.batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer"""
    
    def __init__(
        self,
        buffer_size: int = 100000,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu"
    ):
        super().__init__(buffer_size, batch_size, device)
        
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0
        
    def add(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool
    ):
        """Add experience with maximum priority"""
        
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample batch with prioritization"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Sample experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Process batch (same as parent class)
        states = {
            'market_data': torch.FloatTensor(
                np.stack([e['state']['market_data'] for e in batch])
            ).to(self.device),
            'portfolio_state': torch.FloatTensor(
                np.stack([e['state']['portfolio_state'] for e in batch])
            ).to(self.device)
        }
        
        actions = torch.FloatTensor(
            np.stack([e['action'] for e in batch])
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            [e['reward'] for e in batch]
        ).to(self.device)
        
        next_states = {
            'market_data': torch.FloatTensor(
                np.stack([e['next_state']['market_data'] for e in batch])
            ).to(self.device),
            'portfolio_state': torch.FloatTensor(
                np.stack([e['next_state']['portfolio_state'] for e in batch])
            ).to(self.device)
        }
        
        dones = torch.FloatTensor(
            [e['done'] for e in batch]
        ).to(self.device)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority + self.epsilon)


class SequenceReplayBuffer:
    """Replay buffer for sequences (for LSTM training)"""
    
    def __init__(
        self,
        buffer_size: int = 10000,
        sequence_length: int = 10,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        
        self.episodes = []
        self.current_episode = []
        
    def add(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool
    ):
        """Add experience to current episode"""
        
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        if done:
            # Episode ended, store it
            if len(self.current_episode) >= self.sequence_length:
                self.episodes.append(self.current_episode)
                
                # Maintain buffer size
                if len(self.episodes) > self.buffer_size:
                    self.episodes.pop(0)
            
            self.current_episode = []
    
    def sample_sequences(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample sequences of experiences"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        sequences = []
        
        for _ in range(batch_size):
            # Sample a random episode
            episode = random.choice(self.episodes)
            
            # Sample a random starting point
            max_start = len(episode) - self.sequence_length
            start_idx = random.randint(0, max_start)
            
            # Extract sequence
            sequence = episode[start_idx:start_idx + self.sequence_length]
            sequences.append(sequence)
        
        # Process sequences into tensors
        states = {
            'market_data': torch.FloatTensor(
                np.stack([[step['state']['market_data'] for step in seq] for seq in sequences])
            ).to(self.device),
            'portfolio_state': torch.FloatTensor(
                np.stack([[step['state']['portfolio_state'] for step in seq] for seq in sequences])
            ).to(self.device)
        }
        
        actions = torch.FloatTensor(
            np.stack([[step['action'] for step in seq] for seq in sequences])
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            [[step['reward'] for step in seq] for seq in sequences]
        ).to(self.device)
        
        next_states = {
            'market_data': torch.FloatTensor(
                np.stack([[step['next_state']['market_data'] for step in seq] for seq in sequences])
            ).to(self.device),
            'portfolio_state': torch.FloatTensor(
                np.stack([[step['next_state']['portfolio_state'] for step in seq] for seq in sequences])
            ).to(self.device)
        }
        
        dones = torch.FloatTensor(
            [[step['done'] for step in seq] for seq in sequences]
        ).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough episodes"""
        return len(self.episodes) >= self.batch_size