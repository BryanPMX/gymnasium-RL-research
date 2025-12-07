import argparse
import os
import random
import csv
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.utils.replay_buffer import ReplayBuffer
from src.agents.dqn import DQNAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def setup_csv_logging(log_dir: str, experiment_name: str, seed: int):
    """Set up CSV logging for episode metrics."""
    csv_dir = os.path.join(log_dir, "csv_logs")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = f"{experiment_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow([
        'episode', 'episode_reward', 'moving_avg_reward', 'epsilon',
        'steps_in_episode', 'total_steps', 'loss'
    ])

    return csv_file, csv_writer


def setup_checkpoint_dir(log_dir: str, experiment_name: str, seed: int):
    """Set up directory for model checkpoints."""
    checkpoint_dir = os.path.join(log_dir, "checkpoints", f"{experiment_name}_seed{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def train_one_seed(config, seed: int, log_dir: str):
    env_id = config["env_id"]
    train_cfg = config["training"]

    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    set_seed(seed)

    # TensorBoard writer for this seed
    run_name = f"{config['experiment_name']}_seed{seed}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    # CSV logging setup
    csv_file, csv_writer = setup_csv_logging(log_dir, config['experiment_name'], seed)

    # Checkpoint directory setup
    checkpoint_dir = setup_checkpoint_dir(log_dir, config['experiment_name'], seed)

    # Agent + replay buffer
    buffer = ReplayBuffer(
        capacity=train_cfg["replay_capacity"],
        batch_size=train_cfg["batch_size"],
    )
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_cfg["learning_rate"],
        gamma=train_cfg["gamma"],
        epsilon_start=train_cfg["epsilon_start"],
        epsilon_end=train_cfg["epsilon_end"],
        epsilon_decay=train_cfg["epsilon_decay"],
        target_update_freq=train_cfg["target_update_freq"],
        device="auto",
    )

    episodes = train_cfg["episodes"]
    max_steps = train_cfg["max_steps_per_episode"]

    min_buffer_size = train_cfg["min_buffer_size"]
    eval_interval = train_cfg["eval_interval"]
    eval_episodes = train_cfg["eval_episodes"]
    log_interval = train_cfg["log_interval"]
    ma_window = train_cfg["moving_avg_window"]
    checkpoint_interval = train_cfg.get("checkpoint_interval", 100)  # Save checkpoint every 100 episodes

    episode_rewards = []
    moving_avg_rewards = deque(maxlen=ma_window)

    global_step = 0
    best_reward = float('-inf')

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed)
        state = np.array(state, dtype=np.float32)
        done = False
        truncated = False
        ep_reward = 0.0

        for t in range(max_steps):
            global_step += 1

            # ---- ACTION SELECTION ----
            # Agent internally uses epsilon for exploration.
            action = agent.select_action(state, training=True)

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            # store transition
            buffer.push(state, action, reward, next_state, done or truncated)
            ep_reward += reward
            state = next_state

            # ---- LEARNING STEP ----
            loss = None
            if len(buffer) >= min_buffer_size:
                batch = buffer.sample()          # (s, a, r, s', done)
                loss = agent.update(batch)

            if loss is not None:
                writer.add_scalar("train/loss", loss, global_step)

            if done or truncated:
                break

        episode_rewards.append(ep_reward)
        moving_avg_rewards.append(ep_reward)

        avg_reward = np.mean(moving_avg_rewards)
        current_eps = agent.get_epsilon()

        # per-episode logs
        writer.add_scalar("train/episode_reward", ep_reward, ep)
        writer.add_scalar("train/epsilon", current_eps, ep)
        writer.add_scalar("train/moving_avg_reward", avg_reward, ep)

        # CSV logging
        csv_writer.writerow([
            ep,
            f"{ep_reward:.4f}",
            f"{avg_reward:.4f}",
            f"{current_eps:.6f}",
            t + 1,  # steps_in_episode
            global_step,
            f"{loss:.6f}" if loss is not None else ""
        ])

        if ep % log_interval == 0:
            print(
                f"[Seed {seed}] Episode {ep}/{episodes} | "
                f"Reward: {ep_reward:.1f} | "
                f"MovingAvg({ma_window}): {avg_reward:.1f} | "
                f"Epsilon: {current_eps:.3f}"
            )

        # periodic evaluation (greedy policy, epsilon not used)
        if ep % eval_interval == 0:
            eval_mean, eval_std = evaluate_policy(
                env_id, agent, eval_episodes, seed=seed
            )
            writer.add_scalar("eval/mean_reward", eval_mean, ep)
            writer.add_scalar("eval/std_reward", eval_std, ep)
            print(
                f"[Seed {seed}] >>> Eval after ep {ep}: "
                f"mean={eval_mean:.1f} ± {eval_std:.1f}"
            )

            # Track best reward for checkpointing
            if eval_mean > best_reward:
                best_reward = eval_mean
                # Save best model checkpoint
                best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_ep{ep}_reward{eval_mean:.1f}.pth")
                agent.save(best_checkpoint_path)
                print(f"[Seed {seed}] >>> Saved best model: {best_checkpoint_path}")

        # Regular checkpointing
        if ep % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{ep}.pth")
            agent.save(checkpoint_path)
            print(f"[Seed {seed}] >>> Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    agent.save(final_checkpoint_path)
    print(f"[Seed {seed}] >>> Saved final model: {final_checkpoint_path}")

    env.close()
    writer.close()
    csv_file.close()

    print(f"[Seed {seed}] Training completed. CSV log saved.")


def evaluate_policy(env_id: str, agent: DQNAgent, episodes: int, seed: int = 0):
    """Run greedy evaluation episodes (no exploration)."""
    env = make_env(env_id, seed + 10_000)
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            # training=False -> no epsilon-based exploration
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            ep_reward += reward
            state = next_state

        rewards.append(ep_reward)

    env.close()
    rewards = np.array(rewards, dtype=np.float32)
    return float(rewards.mean()), float(rewards.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/dqn_baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="Base directory for TensorBoard logs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = config["seeds"]

    for seed in seeds:
        print(f"===== Training seed {seed} =====")
        train_one_seed(config, seed, args.logdir)


if __name__ == "__main__":
    main()
