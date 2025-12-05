"""
Environment wrappers for LunarLander-v2

This module provides environment wrappers to enhance the learning process:
- Reward shaping for better learning signals
- Curriculum learning for progressive difficulty
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple


class BaseWrapper(gym.Wrapper):
    """
    Base wrapper class providing common functionality.

    This serves as a foundation for more specific environment wrappers.
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        self.episode_count = 0
        self.total_steps = 0

    def reset(self, **kwargs):
        """Reset the environment and update episode counter."""
        self.episode_count += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute action and update step counter."""
        result = self.env.step(action)
        self.total_steps += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            'episodes': self.episode_count,
            'total_steps': self.total_steps
        }


class RewardShapingWrapper(BaseWrapper):
    """
    Wrapper that modifies rewards for better learning.

    Implements reward shaping techniques for LunarLander:
    - Bonus for vertical velocity reduction (stability)
    - Penalty for excessive horizontal movement
    - Bonus for leg contact (successful landing)
    """

    def __init__(self, env: gym.Env, shaping_config: Optional[Dict[str, float]] = None):
        """
        Initialize reward shaping wrapper.

        Args:
            env: The environment to wrap
            shaping_config: Configuration for reward shaping parameters
        """
        super().__init__(env)

        # Default reward shaping parameters
        self.config = shaping_config or {
            'vertical_velocity_weight': 0.1,    # Bonus for reducing vertical velocity
            'horizontal_velocity_penalty': 0.05, # Penalty for horizontal movement
            'leg_contact_bonus': 0.5,           # Bonus for leg contact
            'stability_threshold': 0.1          # Velocity threshold for bonuses
        }

        # Track previous state for reward shaping
        self.prev_state = None

    def reset(self, **kwargs):
        """Reset environment and clear previous state."""
        self.prev_state = None
        return super().reset(**kwargs)

    def step(self, action):
        """Execute action with modified reward."""
        state, reward, terminated, truncated, info = super().step(action)

        # Apply reward shaping
        shaped_reward = self._shape_reward(state, reward, terminated, info)

        return state, shaped_reward, terminated, truncated, info

    def _shape_reward(self, state: np.ndarray, original_reward: float,
                     terminated: bool, info: Dict[str, Any]) -> float:
        """
        Apply reward shaping to the original reward.

        Args:
            state: Current state
            original_reward: Original reward from environment
            terminated: Whether episode ended
            info: Additional information

        Returns:
            Shaped reward
        """
        shaped_reward = original_reward

        if self.prev_state is not None:
            # Extract relevant state components for LunarLander-v2
            # State: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
            prev_vy = self.prev_state[3]  # Previous vertical velocity
            prev_vx = self.prev_state[2]  # Previous horizontal velocity

            curr_vy = state[3]  # Current vertical velocity
            curr_vx = state[2]  # Current horizontal velocity

            # Reward for reducing vertical velocity (stability)
            vy_change = prev_vy - curr_vy  # Positive if slowing down
            if vy_change > 0:
                shaped_reward += self.config['vertical_velocity_weight'] * vy_change

            # Penalty for excessive horizontal movement
            horizontal_movement = abs(curr_vx)
            if horizontal_movement > self.config['stability_threshold']:
                shaped_reward -= self.config['horizontal_velocity_penalty'] * horizontal_movement

        # Bonus for leg contact (successful landing setup)
        left_leg = state[6]
        right_leg = state[7]
        if left_leg or right_leg:  # At least one leg touching
            shaped_reward += self.config['leg_contact_bonus']

        # Store current state for next step
        self.prev_state = state.copy()

        return shaped_reward


class CurriculumWrapper(BaseWrapper):
    """
    Wrapper that implements curriculum learning by gradually increasing difficulty.

    Progressively modifies environment parameters based on agent performance.
    """

    def __init__(self, env: gym.Env, curriculum_config: Optional[Dict[str, Any]] = None):
        """
        Initialize curriculum wrapper.

        Args:
            env: The environment to wrap
            curriculum_config: Configuration for curriculum progression
        """
        super().__init__(env)

        # Default curriculum configuration
        self.config = curriculum_config or {
            'phases': [
                {'gravity': 9.8, 'wind_power': 0.0, 'episode_threshold': 100},  # Easy
                {'gravity': 9.8, 'wind_power': 5.0, 'episode_threshold': 200},  # Medium
                {'gravity': 12.0, 'wind_power': 10.0, 'episode_threshold': float('inf')}  # Hard
            ],
            'evaluation_window': 50,  # Episodes to evaluate performance
            'performance_threshold': 50.0  # Average reward to advance
        }

        self.current_phase = 0
        self.episode_rewards = []
        self.phase_start_episode = 0
        self.episode_reward_accumulator = 0.0  # Track cumulative episode reward

        # Store original environment parameters
        self.original_gravity = getattr(env, 'gravity', 9.8)
        self.original_wind_power = getattr(env, 'wind_power', 0.0)

    def reset(self, **kwargs):
        """Reset and potentially update environment difficulty."""
        result = super().reset(**kwargs)

        # Reset episode reward accumulator for new episode
        self.episode_reward_accumulator = 0.0

        # Update environment parameters based on current phase
        self._update_environment_parameters()

        return result

    def step(self, action):
        """Execute action and track performance for curriculum."""
        state, reward, terminated, truncated, info = super().step(action)

        # Accumulate reward throughout the episode
        self.episode_reward_accumulator += reward

        # Track episode rewards only when episode ends
        if terminated or truncated:
            # Store the cumulative episode reward, not just the final step reward
            self.episode_rewards.append(self.episode_reward_accumulator)

            # Check if we should advance to next phase
            if self._should_advance_phase():
                self._advance_phase()

        return state, reward, terminated, truncated, info

    def _update_environment_parameters(self):
        """Update environment parameters based on current curriculum phase."""
        phase = self.config['phases'][self.current_phase]

        # Modify environment parameters if supported
        if hasattr(self.env, 'gravity'):
            self.env.gravity = phase.get('gravity', self.original_gravity)
        if hasattr(self.env, 'wind_power'):
            self.env.wind_power = phase.get('wind_power', self.original_wind_power)

    def _should_advance_phase(self) -> bool:
        """Check if agent should advance to next phase."""
        if self.current_phase >= len(self.config['phases']) - 1:
            return False  # Already at final phase

        # Check episode threshold
        episodes_in_phase = self.episode_count - self.phase_start_episode
        if episodes_in_phase < self.config['evaluation_window']:
            return False

        # Check performance threshold
        recent_rewards = self.episode_rewards[-self.config['evaluation_window']:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0

        next_phase_threshold = self.config['phases'][self.current_phase + 1]['episode_threshold']
        performance_met = avg_reward >= self.config['performance_threshold']

        return episodes_in_phase >= next_phase_threshold and performance_met

    def _advance_phase(self):
        """Advance to the next curriculum phase."""
        if self.current_phase < len(self.config['phases']) - 1:
            self.current_phase += 1
            self.phase_start_episode = self.episode_count
            print(f"Advancing to curriculum phase {self.current_phase + 1}")
            print(f"New parameters: {self.config['phases'][self.current_phase]}")

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            'current_phase': self.current_phase + 1,
            'total_phases': len(self.config['phases']),
            'episodes_in_phase': self.episode_count - self.phase_start_episode,
            'recent_performance': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'phase_parameters': self.config['phases'][self.current_phase]
        }
