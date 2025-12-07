"""
Proximal Policy Optimization (PPO) Agent for LunarLander-v2.

Actor–Critic architecture:
- Shared MLP: 8 -> 256 -> 256
- Policy head: 256 -> 4 (discrete actions)
- Value head: 256 -> 1 (state value)

Implements:
- Clipped PPO objective
- Advantage = returns - values
- Mini-batch updates over one collected trajectory
"""

from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        logits = self.policy_head(x)          # [B, action_dim]
        value = self.value_head(x).squeeze(-1)  # [B]
        return logits, value


class PPOAgent:
    """
    PPO agent with actor-critic network.

    Public API used by training script:
    - select_action(state) -> (action, log_prob, value)
    - update_trajectory(trajectory) -> loss value
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.net = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        print(f"PPO Agent initialized on device: {self.device}")
        print(self.net)

    # ---------- Action selection ----------

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select an action given a single state.

        Returns:
            action (int)
            log_prob (float)
            value (float)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
        with torch.no_grad():
            logits, value = self.net(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze(0).item()),
        )

    # ---------- Training update on one trajectory ----------

    def _compute_returns_and_advantages(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        gamma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple returns and advantages:
        G_t = r_t + gamma * G_{t+1} * (1 - done)
        adv_t = G_t - V(s_t)
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0

        for t in reversed(range(T)):
            G = rewards[t] + gamma * G * (1.0 - dones[t])
            returns[t] = G

        advantages = returns - values
        # normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update_trajectory(self, trajectory: Dict[str, np.ndarray]) -> float:
        """
        Perform PPO update using one full-episode trajectory.

        trajectory keys (numpy arrays, shape [T]):
            'states':      float32 [T, state_dim]
            'actions':     int64   [T]
            'rewards':     float32 [T]
            'dones':       float32 [T]
            'log_probs':   float32 [T]
            'values':      float32 [T]
        """
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]
        old_log_probs = trajectory["log_probs"]
        values = trajectory["values"]

        returns, advantages = self._compute_returns_and_advantages(
            rewards, dones, values, self.gamma
        )

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        N = states_t.size(0)
        losses = []

        for _ in range(self.update_epochs):
            # mini-batch indices
            idx = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                mb_idx = idx[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]

                logits, values_pred = self.net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ratio = pi(a|s) / pi_old(a|s)
                ratios = torch.exp(log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(
                    ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.optimizer.step()

                losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    # ---------- Save / load ----------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {
                    "gamma": self.gamma,
                    "clip_epsilon": self.clip_epsilon,
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                    "update_epochs": self.update_epochs,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
