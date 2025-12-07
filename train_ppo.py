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

from src.env.wrappers import RewardShapingWrapper, CurriculumWrapper
from src.agents.ppo import PPOAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_id: str, seed: int, use_reward_shaping: bool, use_curriculum: bool):
    env = gym.make(env_id)
    if use_reward_shaping:
        env = RewardShapingWrapper(env)
    if use_curriculum:
        env = CurriculumWrapper(env)
    env.reset(seed=seed)
    return env


def setup_csv_logging(log_dir: str, experiment_name: str, seed: int):
    csv_dir = os.path.join(log_dir, "csv_logs")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = f"{experiment_name}_ppo_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "episode",
            "episode_reward",
            "moving_avg_reward",
            "loss",
            "steps_in_episode",
            "total_steps",
        ]
    )
    return csv_file, csv_writer


def train_one_seed(config, seed: int, log_dir: str):
    env_id = config["env_id"]
    train_cfg = config["training"]

    use_reward_shaping = train_cfg.get("use_reward_shaping", True)
    use_curriculum = train_cfg.get("use_curriculum", True)

    env = make_env(env_id, seed, use_reward_shaping, use_curriculum)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    set_seed(seed)

    run_name = f"{config['experiment_name']}_ppo_seed{seed}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    csv_file, csv_writer = setup_csv_logging(log_dir, config["experiment_name"], seed)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_cfg.get("learning_rate", 3e-4),
        gamma=train_cfg.get("gamma", 0.99),
        clip_epsilon=train_cfg.get("clip_epsilon", 0.2),
        value_coef=train_cfg.get("value_coef", 0.5),
        entropy_coef=train_cfg.get("entropy_coef", 0.01),
        update_epochs=train_cfg.get("update_epochs", 10),
        batch_size=train_cfg.get("batch_size", 64),
        device="auto",
    )

    episodes = train_cfg["episodes"]
    max_steps = train_cfg["max_steps_per_episode"]
    log_interval = train_cfg.get("log_interval", 10)
    eval_interval = train_cfg.get("eval_interval", 50)
    eval_episodes = train_cfg.get("eval_episodes", 10)
    ma_window = train_cfg.get("moving_avg_window", 50)

    # curriculum logging optional
    use_curriculum_status = use_curriculum and hasattr(env, "get_curriculum_status")

    episode_rewards = []
    moving_avg_rewards = deque(maxlen=ma_window)
    global_step = 0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed)
        state = np.array(state, dtype=np.float32)

        done = False
        truncated = False
        ep_reward = 0.0

        # trajectory containers
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        for t in range(max_steps):
            global_step += 1

            action, log_prob, value = agent.select_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done or truncated))
            log_probs.append(log_prob)
            values.append(value)

            ep_reward += reward
            state = next_state

            if done or truncated:
                break

        # convert to arrays
        trajectory = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
        }

        loss = agent.update_trajectory(trajectory)

        episode_rewards.append(ep_reward)
        moving_avg_rewards.append(ep_reward)
        avg_reward = float(np.mean(moving_avg_rewards))

        writer.add_scalar("train/episode_reward", ep_reward, ep)
        writer.add_scalar("train/moving_avg_reward", avg_reward, ep)
        writer.add_scalar("train/loss", loss, ep)

        if use_curriculum_status:
            status = env.get_curriculum_status()
            writer.add_scalar("curriculum/phase", status["current_phase"], ep)
            writer.add_scalar(
                "curriculum/recent_performance", status["recent_performance"], ep
            )

        csv_writer.writerow(
            [
                ep,
                f"{ep_reward:.4f}",
                f"{avg_reward:.4f}",
                f"{loss:.6f}",
                t + 1,
                global_step,
            ]
        )

        if ep % log_interval == 0:
            print(
                f"[PPO Seed {seed}] Episode {ep}/{episodes} | "
                f"Reward: {ep_reward:.1f} | MA({ma_window}): {avg_reward:.1f}"
            )

        if ep % eval_interval == 0:
            eval_mean, eval_std = evaluate_policy(env_id, agent, eval_episodes, seed)
            writer.add_scalar("eval/mean_reward", eval_mean, ep)
            writer.add_scalar("eval/std_reward", eval_std, ep)
            print(
                f"[PPO Seed {seed}] >>> Eval after ep {ep}: "
                f"mean={eval_mean:.1f} ± {eval_std:.1f}"
            )

    env.close()
    writer.close()
    csv_file.close()
    print(f"[PPO Seed {seed}] Training completed.")


def evaluate_policy(env_id: str, agent: PPOAgent, episodes: int, seed: int = 0):
    env = gym.make(env_id)
    env.reset(seed=seed + 10000)

    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            # Greedy: take argmax of policy
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                logits, _ = agent.net(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()

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
        default="experiments/ppo_baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs_ppo",
        help="Base directory for TensorBoard logs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = config["seeds"]

    for seed in seeds:
        print(f"===== PPO training, seed {seed} =====")
        train_one_seed(config, seed, args.logdir)


if __name__ == "__main__":
    main()
