"""
Deep Q-Network Agent for LunarLander-v2

This module implements a DQN agent with the following features:
- Q-network with 8→256→256→4 architecture
- Xavier weight initialization
- Target network for stable training
- Experience replay integration
- Epsilon-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
import os


class QNetwork(nn.Module):
    """
    Deep Q-Network for approximating the action-value function.

    Architecture: Input(8) -> Hidden(256) -> Hidden(256) -> Output(4)
    Uses ReLU activations and Xavier initialization.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 256):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space (default: 8 for LunarLander)
            action_dim: Dimension of action space (default: 4 for LunarLander)
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super(QNetwork, self).__init__()

        # Define network layers
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.layers(x)


class DQNAgent:
    """
    Deep Q-Network Agent for LunarLander-v2.

    Implements DQN with target network, experience replay, and epsilon-greedy exploration.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000,
        device: str = "auto"
    ):
        """
        Initialize DQN Agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            lr: Learning rate for Adam optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay rate
            target_update_freq: Frequency to update target network
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Q-networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        print(f"DQN Agent initialized on device: {self.device}")
        print(f"Policy network: {self.policy_net}")
        print(f"Target network: {self.target_net}")

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Target network updated at step {self.step_count}")

    # Placeholder methods - will be implemented in subsequent commits
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action
        """
        if training and np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(self.action_dim)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
        """
        Update the policy network using a batch of experiences.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            Loss value
        """
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = F.huber_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)

        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str) -> None:
        """
        Save model state to disk.

        Args:
            path: Path to save the model
        """
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if path has a directory component
            os.makedirs(dir_path, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'target_update_freq': self.target_update_freq
            }
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model state from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        print(f"Model loaded from {path}")

    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """Set epsilon value (useful for evaluation)."""
        self.epsilon = epsilon
