import gymnasium as gym
import numpy as np
import torch
import random
from pathlib import Path
import time
from stable_baselines3.common.monitor import ResultsWriter
from stream_rl.src.models.stream_ac_discrete import StreamAC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(num_episodes=500, seed=0, render=False):
    set_seed(seed)

    run_dir = Path("logs/stream_ac/CartPole-v1_1")
    run_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    monitor = ResultsWriter(
        filename=str(run_dir / "stream_ac"),
        header={
            "t_start": t_start,
            "env_id": "CartPole-v1",
            "algorithm": "StreamAC",
            "seed": seed,
        }
    )

    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = StreamAC(
        n_obs=n_obs,
        n_actions=n_actions,
        hidden_size=64,
        lr=1.0,
        gamma=0.99,
        lamda=0.8,
        kappa_policy=3.0,
        kappa_value=2.0,
        entropy_coeff=0.01,
    )

    episode_returns = []

    for episode in range(num_episodes):
        s, _ = env.reset(seed=seed + episode)
        done = False
        ep_return = 0.0
        ep_length = 0

        agent.optimizer_policy.reset()
        agent.optimizer_value.reset()

        while not done:
            a = agent.sample_action(s)
            s_prime, r, terminated, truncated, _ = env.step(int(a))
            done = terminated or truncated

            agent.update_params(s, a, r, s_prime, terminated)

            s = s_prime
            ep_return += r
            ep_length += 1

        episode_returns.append(ep_return)
        monitor.write_row({
            "r": ep_return,
            "l": ep_length,
            "t": round(time.time() - t_start, 6),
        })

        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Return: {ep_return:.1f} | Avg(20): {avg_return:.1f}")

    monitor.file_handler.close()
    env.close()
    return episode_returns


if __name__ == "__main__":
    returns = train(num_episodes=500, seed=42, render=False)