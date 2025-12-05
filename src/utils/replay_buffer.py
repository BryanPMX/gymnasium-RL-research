import numpy as np
from typing import Tuple, List, Any
import random


class ReplayBuffer:
    """
    A cyclic replay buffer for storing and sampling transitions in reinforcement learning.

    Stores transitions as (state, action, reward, next_state, done) tuples.
    Uses a cyclic buffer that overwrites oldest experiences when full.
    """

    def __init__(self, capacity: int = 50000, batch_size: int = 128):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store (default: 50,000)
            batch_size: Size of batches to sample (default: 128)
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions uniformly at random.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each element is a numpy array with shape (batch_size, ...)
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"Not enough transitions in buffer. Have {len(self.buffer)}, need {self.batch_size}")

        batch = random.sample(self.buffer, self.batch_size)

        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if the buffer has reached maximum capacity."""
        return len(self.buffer) == self.capacity

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer = []
        self.position = 0