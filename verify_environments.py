#!/usr/bin/env python3
"""
Environment verification script for the RL project.
Tests LunarLander-v2 rendering and CartPole smoke tests.
"""

import gymnasium as gym
import numpy as np
import time
import sys


def test_cartpole_smoke():
    """Run a quick smoke test on CartPole environment."""
    print("Testing CartPole environment...")

    try:
        env = gym.make('CartPole-v1')

        # Reset environment
        state, info = env.reset()
        print(f"✓ CartPole environment initialized. Initial state shape: {state.shape}")

        total_reward = 0
        steps = 0
        max_steps = 100

        # Run random actions for smoke test
        for step in range(max_steps):
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        env.close()
        print(f"✓ CartPole smoke test completed. Steps: {steps}, Total reward: {total_reward:.2f}")
        return True

    except Exception as e:
        print(f"✗ CartPole test failed: {e}")
        return False


def test_lunar_lander_rendering():
    """Test LunarLander-v2 rendering capabilities."""
    print("Testing LunarLander-v2 rendering...")

    try:
        # Try to create environment with rendering
        env = gym.make('LunarLander-v2', render_mode='human')

        # Reset environment
        state, info = env.reset()
        print(f"✓ LunarLander-v2 environment initialized. State shape: {state.shape}")
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space: {env.observation_space}")

        # Test a few steps to ensure rendering works
        print("Testing rendering with a few random steps...")
        for step in range(5):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            time.sleep(0.1)  # Brief pause to see rendering

        env.close()
        print("✓ LunarLander-v2 rendering test completed successfully")
        print("Note: Close the rendering window to continue...")
        return True

    except Exception as e:
        print(f"✗ LunarLander-v2 rendering test failed: {e}")
        print("This might be expected if running in a headless environment.")
        return False


def test_lunar_lander_headless():
    """Test LunarLander-v2 in headless mode for basic functionality."""
    print("Testing LunarLander-v2 in headless mode...")

    try:
        env = gym.make('LunarLander-v2')  # No render_mode specified

        state, info = env.reset()
        print(f"✓ LunarLander-v2 headless mode initialized. State shape: {state.shape}")

        # Run a quick episode
        total_reward = 0
        steps = 0
        max_steps = 50

        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        env.close()
        print(f"✓ LunarLander-v2 headless test completed. Steps: {steps}, Total reward: {total_reward:.2f}")
        return True

    except Exception as e:
        print(f"✗ LunarLander-v2 headless test failed: {e}")
        return False


def main():
    """Run all environment verification tests."""
    print("=" * 60)
    print("Reinforcement Learning Environment Verification")
    print("=" * 60)

    results = []

    # Test CartPole
    results.append(test_cartpole_smoke())
    print()

    # Test LunarLander rendering (will work in GUI environments)
    results.append(test_lunar_lander_rendering())
    print()

    # Test LunarLander headless (should always work)
    results.append(test_lunar_lander_headless())
    print()

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("Your RL environment is ready for development!")
        return 0
    else:
        print(f"⚠ Some tests failed ({passed}/{total})")
        print("Check the error messages above. LunarLander rendering may fail in headless environments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())