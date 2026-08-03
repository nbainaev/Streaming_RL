"""Reproducible random-policy reference for selected POPGym environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import popgym  # noqa: F401


def evaluate(env_id: str, seed: int, episodes: int):
    env = gym.make(env_id)
    env.action_space.seed(seed + 17)
    observation, _ = env.reset(seed=seed)
    del observation
    returns = []
    episode_return = 0.0
    while len(returns) < episodes:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_return += float(reward)
        if terminated or truncated:
            returns.append(episode_return)
            env.reset()
            episode_return = 0.0
    env.close()
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for env_id in args.env_ids:
        values = [evaluate(env_id, seed, args.episodes) for seed in args.seeds]
        result[env_id] = {
            "per_seed": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
