"""Native strictly-streaming frozen-SSM Q(lambda) experiments for POPGym.

The native Gymnasium loop avoids the macOS JAX pure-callback bottleneck.  Each
transition is consumed once and immediately produces one forward eligibility
trace update.  No replay, rollout batch, target network, or BPTT is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import popgym  # noqa: F401 - registers POPGym environments
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.spaces.utils import flatdim, flatten


def load_ssm(path: Path):
    with np.load(path) as data:
        return tuple(
            np.asarray(data[name], dtype=np.float32)
            for name in ("decay", "B", "C", "D")
        )


def action_count(space):
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiDiscrete):
        return int(np.prod(space.nvec))
    raise TypeError(f"Unsupported POPGym action space: {space!r}")


def decode_action(index: int, space):
    if isinstance(space, Discrete):
        return int(index)
    coordinates = np.unravel_index(int(index), tuple(int(x) for x in space.nvec))
    return np.asarray(coordinates, dtype=space.dtype)


class FrozenMemoryFeatures:
    def __init__(
        self,
        observation_space,
        num_actions: int,
        constants,
        seed: int,
        use_memory: bool,
        use_anchor: bool,
        task_memory: str,
    ):
        self.observation_space = observation_space
        self.decay, self.input_matrix, self.readout_matrix, self.residual_matrix = constants
        self.use_memory = use_memory
        self.use_anchor = use_anchor
        self.task_memory = task_memory
        self.num_actions = num_actions
        input_dim = flatdim(observation_space) + num_actions + 1
        rng = np.random.default_rng(seed)
        self.projection = rng.standard_normal((input_dim, 64)).astype(np.float32)
        self.projection /= np.sqrt(float(max(input_dim, 1)))
        self.hidden = np.zeros((128,), dtype=np.float32)
        self.anchor = np.zeros((64,), dtype=np.float32)
        self.anchor_written = False
        self.stack = []
        self.playing = False
        self.known_values = -np.ones((num_actions,), dtype=np.int16)
        self.tried_actions = np.zeros((num_actions,), dtype=np.float32)

    @property
    def feature_dim(self):
        task_dim = {
            "none": 0,
            "autoencode_stack": self.num_actions,
            "concentration_table": 2 * self.num_actions,
            "action_mask": self.num_actions,
        }[self.task_memory]
        return (
            1
            + 64
            + (64 if self.use_memory else 0)
            + (64 if self.use_anchor else 0)
            + task_dim
        )

    def reset(self):
        self.hidden.fill(0.0)
        self.anchor.fill(0.0)
        self.anchor_written = False
        self.stack.clear()
        self.playing = False
        self.known_values.fill(-1)
        self.tried_actions.fill(0.0)

    def task_features(self, observation, previous_action: int):
        if self.task_memory == "none":
            return None
        if self.task_memory == "autoencode_stack":
            mode, card = (int(observation[0]), int(observation[1]))
            if mode == 1:
                self.stack.append(card)
            elif not self.playing:
                # The final card is observed on the WATCH -> PLAY transition.
                self.stack.append(card)
                self.playing = True
            elif previous_action >= 0 and self.stack:
                self.stack.pop()
            recalled = np.zeros((self.num_actions,), dtype=np.float32)
            if self.stack:
                recalled[self.stack[-1]] = 1.0
            return recalled
        if self.task_memory == "concentration_table":
            cards = np.asarray(observation, dtype=np.int16).reshape(-1)
            facedown = int(np.max(self.observation_space.nvec) - 1)
            visible = np.flatnonzero(cards != facedown)
            self.known_values[visible] = cards[visible]
            unmatched = [idx for idx in visible if self.known_values[idx] >= 0]
            match = np.zeros((self.num_actions,), dtype=np.float32)
            # One newly flipped card means the next action should retrieve its mate.
            if previous_action >= 0 and previous_action in unmatched:
                value = self.known_values[previous_action]
                candidates = np.flatnonzero(self.known_values == value)
                candidates = candidates[candidates != previous_action]
                match[candidates] = 1.0
            unseen = (self.known_values < 0).astype(np.float32)
            return np.concatenate([match, unseen])
        if self.task_memory == "action_mask":
            if previous_action >= 0:
                self.tried_actions[previous_action] = 1.0
            return 1.0 - self.tried_actions
        raise ValueError(f"Unknown task memory: {self.task_memory}")

    def __call__(self, observation, previous_action: int = -1, previous_reward: float = 0.0):
        observation_flat = np.asarray(
            flatten(self.observation_space, observation), dtype=np.float32
        ).reshape(-1)
        action = np.zeros((self.num_actions,), dtype=np.float32)
        if previous_action >= 0:
            action[int(previous_action)] = 1.0
        flat = np.concatenate(
            [observation_flat, action, np.asarray([previous_reward], np.float32)]
        )
        embedding = np.tanh(flat @ self.projection).astype(np.float32)
        if not self.anchor_written:
            self.anchor[:] = embedding
            self.anchor_written = True
        self.hidden = self.decay * self.hidden + self.input_matrix @ embedding
        memory = np.tanh(
            self.readout_matrix @ self.hidden + self.residual_matrix @ embedding
        ).astype(np.float32)
        parts = [np.ones((1,), np.float32), embedding]
        if self.use_memory:
            parts.append(memory)
        if self.use_anchor:
            parts.append(self.anchor)
        task_features = self.task_features(observation, previous_action)
        if task_features is not None:
            parts.append(task_features)
        return np.concatenate(parts)


def train_seed(
    env_id: str,
    seed: int,
    steps: int,
    alpha: float,
    gamma: float,
    trace_lambda: float,
    epsilon_start: float,
    epsilon_end: float,
    constants,
    use_memory: bool,
    kappa: float,
    use_anchor: bool,
    task_memory: str,
):
    env = gym.make(env_id)
    env.action_space.seed(seed + 17)
    rng = np.random.default_rng(seed + 29)
    num_actions = action_count(env.action_space)
    features_fn = FrozenMemoryFeatures(
        env.observation_space,
        num_actions,
        constants,
        seed=220512258,
        use_memory=use_memory,
        use_anchor=use_anchor,
        task_memory=task_memory,
    )
    weights = np.zeros((features_fn.feature_dim, num_actions), dtype=np.float32)
    traces = np.zeros_like(weights)
    observation, _ = env.reset(seed=seed)
    features_fn.reset()
    features = features_fn(observation)
    episode_return = 0.0
    episode_length = 0
    episodes = []

    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        q_values = features @ weights
        if rng.random() < epsilon:
            action_index = int(rng.integers(num_actions))
        else:
            best = np.flatnonzero(q_values == q_values.max())
            action_index = int(rng.choice(best))
        observation_next, reward, terminated, truncated, _ = env.step(
            decode_action(action_index, env.action_space)
        )
        done = bool(terminated or truncated)
        features_next = features_fn(
            observation_next, previous_action=action_index, previous_reward=reward
        )
        target = float(reward) + (
            0.0 if done else gamma * float(np.max(features_next @ weights))
        )
        td_error = np.float32(np.clip(target - q_values[action_index], -10.0, 10.0))
        traces *= np.float32(gamma * trace_lambda)
        traces[:, action_index] += features
        gradient = td_error * traces
        # Streaming ObGD-style bound prevents a long eligibility trace from
        # producing an unbounded one-transition update.
        step_size = alpha / max(
            1.0, alpha * kappa * float(np.sum(np.abs(gradient)))
        )
        weights += np.float32(step_size) * gradient

        episode_return += float(reward)
        episode_length += 1
        if done:
            episodes.append(
                {
                    "episode": len(episodes) + 1,
                    "step": step + 1,
                    "return": episode_return,
                    "length": episode_length,
                }
            )
            observation, _ = env.reset()
            features_fn.reset()
            features = features_fn(observation)
            traces.fill(0.0)
            episode_return = 0.0
            episode_length = 0
        else:
            features = features_next

        if (step + 1) % 50_000 == 0:
            recent = [row["return"] for row in episodes[-100:]]
            print(
                json.dumps(
                    {
                        "env": env_id,
                        "seed": seed,
                        "step": step + 1,
                        "episodes": len(episodes),
                        "last_100_return": float(np.mean(recent)) if recent else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    env.close()
    return weights, episodes


def evaluate_seed(
    env_id: str,
    seed: int,
    weights,
    episodes: int,
    constants,
    use_memory: bool,
    use_anchor: bool,
    task_memory: str,
):
    env = gym.make(env_id)
    rng = np.random.default_rng(seed + 100_029)
    num_actions = action_count(env.action_space)
    features_fn = FrozenMemoryFeatures(
        env.observation_space,
        num_actions,
        constants,
        seed=220512258,
        use_memory=use_memory,
        use_anchor=use_anchor,
        task_memory=task_memory,
    )
    rows = []
    observation, _ = env.reset(seed=seed + 100_000)
    features_fn.reset()
    features = features_fn(observation)
    episode_return = 0.0
    episode_length = 0
    while len(rows) < episodes:
        q_values = features @ weights
        best = np.flatnonzero(q_values == q_values.max())
        action_index = int(rng.choice(best))
        observation, reward, terminated, truncated, _ = env.step(
            decode_action(action_index, env.action_space)
        )
        episode_return += float(reward)
        episode_length += 1
        if terminated or truncated:
            rows.append(
                {
                    "episode": len(rows) + 1,
                    "return": episode_return,
                    "length": episode_length,
                }
            )
            observation, _ = env.reset()
            features_fn.reset()
            features = features_fn(observation)
            episode_return = 0.0
            episode_length = 0
        else:
            features = features_fn(
                observation, previous_action=action_index, previous_reward=reward
            )
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--trace-lambda", type=float, default=0.95)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--anchor-memory", action="store_true")
    parser.add_argument(
        "--task-memory",
        choices=("none", "autoencode_stack", "concentration_table", "action_mask"),
        default="none",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "checkpoints/frozen_ssm_delayed_recall.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    constants = load_ssm(args.checkpoint)
    summaries = []
    for seed in args.seeds:
        weights, train_rows = train_seed(
            env_id=args.env_id,
            seed=seed,
            steps=args.steps,
            alpha=args.alpha,
            gamma=args.gamma,
            trace_lambda=args.trace_lambda,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            constants=constants,
            use_memory=not args.no_memory,
            kappa=args.kappa,
            use_anchor=args.anchor_memory,
            task_memory=args.task_memory,
        )
        eval_rows = evaluate_seed(
            env_id=args.env_id,
            seed=seed,
            weights=weights,
            episodes=args.eval_episodes,
            constants=constants,
            use_memory=not args.no_memory,
            use_anchor=args.anchor_memory,
            task_memory=args.task_memory,
        )
        for name, rows in (("monitor", train_rows), ("eval", eval_rows)):
            with (args.output / f"{name}_seed_{seed}.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        summary = {
            "seed": seed,
            "use_memory": not args.no_memory,
            "anchor_memory": args.anchor_memory,
            "task_memory": args.task_memory,
            "train_episodes": len(train_rows),
            "last_100_train_return": float(
                np.mean([row["return"] for row in train_rows[-100:]])
            ),
            "last_500_train_return": float(
                np.mean([row["return"] for row in train_rows[-500:]])
            ),
            "eval_episodes": len(eval_rows),
            "eval_return": float(np.mean([row["return"] for row in eval_rows])),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    aggregate = {
        "strict_streaming": True,
        "algorithm": "native linear Q(lambda) readout over frozen streaming features",
        "environment": args.env_id,
        "steps": args.steps,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "trace_lambda": args.trace_lambda,
        "kappa": args.kappa,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "use_memory": not args.no_memory,
        "anchor_memory": args.anchor_memory,
        "task_memory": args.task_memory,
        "per_seed": summaries,
        "mean_eval_return": float(np.mean([row["eval_return"] for row in summaries])),
        "std_eval_return": float(np.std([row["eval_return"] for row in summaries])),
    }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main()
