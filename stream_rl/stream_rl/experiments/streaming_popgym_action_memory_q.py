"""Strictly-streaming action-conditioned memory for POPGym tasks.

The memory is deterministic and frozen.  Only one shared linear action scorer
is learned, so a rule learned for one card/cell is immediately shared across
all positions.  Every transition causes one Q(lambda) update; there is no
replay, rollout batch, target network, or BPTT.
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


def action_count(space) -> int:
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiDiscrete):
        return int(np.prod(space.nvec))
    raise TypeError(f"Unsupported action space: {space!r}")


def decode_action(index: int, space):
    if isinstance(space, Discrete):
        return int(index)
    coordinates = np.unravel_index(int(index), tuple(int(value) for value in space.nvec))
    return np.asarray(coordinates, dtype=space.dtype)


class ConcentrationActionMemory:
    """Frozen key-value table with action-conditioned content lookup."""

    def __init__(self, observation_space, num_actions: int, mask_only: bool = False):
        self.num_actions = num_actions
        self.facedown = int(np.max(observation_space.nvec) - 1)
        self.mask_only = mask_only
        self.known_values = np.full((num_actions,), -1, dtype=np.int16)
        self.removed = np.zeros((num_actions,), dtype=bool)
        self.first_action: int | None = None

    @property
    def feature_dim(self) -> int:
        return 1 if self.mask_only else 6

    def reset(self) -> None:
        self.known_values.fill(-1)
        self.removed.fill(False)
        self.first_action = None

    def observe(
        self,
        observation,
        previous_action: int = -1,
        previous_reward: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        cards = np.asarray(observation, dtype=np.int16).reshape(-1)
        if previous_action >= 0:
            visible = np.flatnonzero(cards != self.facedown)
            self.known_values[visible] = cards[visible]
            if self.first_action is None:
                self.first_action = int(previous_action)
            else:
                if previous_reward > 0.0:
                    self.removed[self.first_action] = True
                    self.removed[int(previous_action)] = True
                self.first_action = None

        valid = ~self.removed.copy()
        if self.first_action is not None:
            valid[self.first_action] = False
        if self.mask_only:
            return np.ones((self.num_actions, 1), dtype=np.float32), valid

        second_pick = self.first_action is not None
        target_value = (
            self.known_values[self.first_action] if second_pick else np.int16(-2)
        )
        known = (self.known_values >= 0) & valid
        unseen = (self.known_values < 0) & valid
        match = known & (self.known_values == target_value)
        # Fixed one-hot key/query equality is the content-addressing operation;
        # the shared readout only learns how much to value its result.
        features = np.stack(
            [
                np.ones((self.num_actions,), dtype=np.float32),
                np.full((self.num_actions,), float(second_pick), dtype=np.float32),
                match.astype(np.float32),
                unseen.astype(np.float32),
                known.astype(np.float32),
                valid.astype(np.float32),
            ],
            axis=1,
        )
        return features, valid


class BattleshipActionMemory:
    """Frozen spatial map with per-action neighborhood retrieval."""

    def __init__(
        self,
        num_actions: int,
        mask_only: bool = False,
        minimal_features: bool = False,
    ):
        side = int(round(np.sqrt(num_actions)))
        if side * side != num_actions:
            raise ValueError("Battleship action count must form a square board")
        self.num_actions = num_actions
        self.side = side
        self.mask_only = mask_only
        self.minimal_features = minimal_features
        self.tried = np.zeros((num_actions,), dtype=bool)
        self.hits = np.zeros((num_actions,), dtype=bool)
        self.coordinates = np.asarray(
            [(row, col) for row in range(side) for col in range(side)],
            dtype=np.int16,
        )

    @property
    def feature_dim(self) -> int:
        if self.mask_only:
            return 1
        return 4 if self.minimal_features else 9

    def reset(self) -> None:
        self.tried.fill(False)
        self.hits.fill(False)

    def _line_extensions(self) -> np.ndarray:
        extensions = np.zeros((self.side, self.side), dtype=np.float32)
        hit_board = self.hits.reshape(self.side, self.side)
        for row in range(self.side):
            columns = np.flatnonzero(hit_board[row])
            for left, right in zip(columns[:-1], columns[1:]):
                if right - left == 1:
                    if left > 0:
                        extensions[row, left - 1] = 1.0
                    if right + 1 < self.side:
                        extensions[row, right + 1] = 1.0
        for col in range(self.side):
            rows = np.flatnonzero(hit_board[:, col])
            for top, bottom in zip(rows[:-1], rows[1:]):
                if bottom - top == 1:
                    if top > 0:
                        extensions[top - 1, col] = 1.0
                    if bottom + 1 < self.side:
                        extensions[bottom + 1, col] = 1.0
        return extensions.reshape(-1)

    def observe(
        self,
        observation,
        previous_action: int = -1,
        previous_reward: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        del previous_reward
        if previous_action >= 0:
            self.tried[int(previous_action)] = True
            if int(observation) == 1:
                self.hits[int(previous_action)] = True
        valid = ~self.tried
        if self.mask_only:
            return np.ones((self.num_actions, 1), dtype=np.float32), valid

        rows = self.coordinates[:, 0]
        cols = self.coordinates[:, 1]
        hit_rows = rows[self.hits]
        hit_cols = cols[self.hits]
        miss = self.tried & (~self.hits)
        miss_rows = rows[miss]
        miss_cols = cols[miss]
        if hit_rows.size:
            distances = np.abs(rows[:, None] - hit_rows[None, :]) + np.abs(
                cols[:, None] - hit_cols[None, :]
            )
            adjacent_hits = np.sum(distances == 1, axis=1).astype(np.float32)
            distance_two_hits = np.sum(distances == 2, axis=1).astype(np.float32)
            same_row_hits = np.sum(rows[:, None] == hit_rows[None, :], axis=1).astype(
                np.float32
            )
            same_col_hits = np.sum(cols[:, None] == hit_cols[None, :], axis=1).astype(
                np.float32
            )
        else:
            adjacent_hits = np.zeros((self.num_actions,), dtype=np.float32)
            distance_two_hits = np.zeros_like(adjacent_hits)
            same_row_hits = np.zeros_like(adjacent_hits)
            same_col_hits = np.zeros_like(adjacent_hits)
        if miss_rows.size:
            miss_distances = np.abs(rows[:, None] - miss_rows[None, :]) + np.abs(
                cols[:, None] - miss_cols[None, :]
            )
            adjacent_misses = np.sum(miss_distances == 1, axis=1).astype(np.float32)
        else:
            adjacent_misses = np.zeros((self.num_actions,), dtype=np.float32)
        parity = ((rows + cols) % 2 == 0).astype(np.float32)
        line_extension = self._line_extensions()
        scale = float(max(self.side, 1))
        if self.minimal_features:
            feature_columns = [
                np.ones((self.num_actions,), dtype=np.float32),
                adjacent_hits / 4.0,
                distance_two_hits / 8.0,
                line_extension,
            ]
        else:
            feature_columns = [
                np.ones((self.num_actions,), dtype=np.float32),
                parity,
                adjacent_hits / 4.0,
                (adjacent_hits > 0).astype(np.float32),
                distance_two_hits / 8.0,
                same_row_hits / scale,
                same_col_hits / scale,
                line_extension,
                adjacent_misses / 4.0,
            ]
        features = np.stack(feature_columns, axis=1)
        return features, valid


def make_memory(env_id: str, env, mask_only: bool, minimal_features: bool):
    num_actions = action_count(env.action_space)
    if "Concentration" in env_id:
        return ConcentrationActionMemory(env.observation_space, num_actions, mask_only)
    if "Battleship" in env_id:
        return BattleshipActionMemory(num_actions, mask_only, minimal_features)
    raise ValueError(f"Unsupported environment for action memory: {env_id}")


def choose_action(
    q_values: np.ndarray,
    valid: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        candidates = np.arange(q_values.size)
    if rng.random() < epsilon:
        return int(rng.choice(candidates))
    valid_q = q_values[candidates]
    best = candidates[np.flatnonzero(valid_q == np.max(valid_q))]
    return int(rng.choice(best))


def train_seed(
    env_id: str,
    seed: int,
    steps: int,
    alpha: float,
    gamma: float,
    trace_lambda: float,
    epsilon_start: float,
    epsilon_end: float,
    kappa: float,
    mask_only: bool,
    minimal_features: bool,
):
    env = gym.make(env_id)
    env.action_space.seed(seed + 17)
    rng = np.random.default_rng(seed + 29)
    memory = make_memory(env_id, env, mask_only, minimal_features)
    weights = np.zeros((memory.feature_dim,), dtype=np.float32)
    traces = np.zeros_like(weights)
    observation, _ = env.reset(seed=seed)
    memory.reset()
    action_features, valid = memory.observe(observation)
    episode_return = 0.0
    episode_length = 0
    episodes = []

    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        q_values = action_features @ weights
        action_index = choose_action(q_values, valid, epsilon, rng)
        next_observation, reward, terminated, truncated, _ = env.step(
            decode_action(action_index, env.action_space)
        )
        done = bool(terminated or truncated)
        next_features, next_valid = memory.observe(
            next_observation,
            previous_action=action_index,
            previous_reward=float(reward),
        )
        next_q_values = next_features @ weights
        next_candidates = np.flatnonzero(next_valid)
        bootstrap = (
            0.0
            if done or next_candidates.size == 0
            else gamma * float(np.max(next_q_values[next_candidates]))
        )
        td_error = np.float32(
            np.clip(float(reward) + bootstrap - q_values[action_index], -10.0, 10.0)
        )
        traces *= np.float32(gamma * trace_lambda)
        traces += action_features[action_index]
        gradient = td_error * traces
        step_size = alpha / max(1.0, alpha * kappa * float(np.sum(np.abs(gradient))))
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
            memory.reset()
            action_features, valid = memory.observe(observation)
            traces.fill(0.0)
            episode_return = 0.0
            episode_length = 0
        else:
            action_features, valid = next_features, next_valid

        if (step + 1) % 50_000 == 0:
            recent = [episode["return"] for episode in episodes[-100:]]
            print(
                json.dumps(
                    {
                        "env": env_id,
                        "seed": seed,
                        "step": step + 1,
                        "episodes": len(episodes),
                        "last_100_return": float(np.mean(recent)) if recent else None,
                        "weights": weights.tolist(),
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
    weights: np.ndarray,
    episodes: int,
    mask_only: bool,
    minimal_features: bool,
):
    env = gym.make(env_id)
    rng = np.random.default_rng(seed + 100_029)
    memory = make_memory(env_id, env, mask_only, minimal_features)
    rows = []
    observation, _ = env.reset(seed=seed + 100_000)
    memory.reset()
    action_features, valid = memory.observe(observation)
    episode_return = 0.0
    episode_length = 0
    while len(rows) < episodes:
        q_values = action_features @ weights
        action_index = choose_action(q_values, valid, 0.0, rng)
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
            memory.reset()
            action_features, valid = memory.observe(observation)
            episode_return = 0.0
            episode_length = 0
        else:
            action_features, valid = memory.observe(
                observation,
                previous_action=action_index,
                previous_reward=float(reward),
            )
    env.close()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        choices=("popgym-ConcentrationEasy-v0", "popgym-BattleshipEasy-v0"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--trace-lambda", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--minimal-features", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
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
            kappa=args.kappa,
            mask_only=args.mask_only,
            minimal_features=args.minimal_features,
        )
        eval_rows = evaluate_seed(
            env_id=args.env_id,
            seed=seed,
            weights=weights,
            episodes=args.eval_episodes,
            mask_only=args.mask_only,
            minimal_features=args.minimal_features,
        )
        write_csv(args.output / f"monitor_seed_{seed}.csv", train_rows)
        write_csv(args.output / f"eval_seed_{seed}.csv", eval_rows)
        summary = {
            "seed": seed,
            "train_episodes": len(train_rows),
            "last_100_train_return": float(
                np.mean([row["return"] for row in train_rows[-100:]])
            ),
            "last_500_train_return": float(
                np.mean([row["return"] for row in train_rows[-500:]])
            ),
            "eval_episodes": len(eval_rows),
            "eval_return": float(np.mean([row["return"] for row in eval_rows])),
            "eval_length": float(np.mean([row["length"] for row in eval_rows])),
            "weights": weights.tolist(),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    aggregate = {
        "strict_streaming": True,
        "algorithm": "shared linear Q(lambda) over frozen action-conditioned memory",
        "environment": args.env_id,
        "steps": args.steps,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "trace_lambda": args.trace_lambda,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "kappa": args.kappa,
        "mask_only": args.mask_only,
        "minimal_features": args.minimal_features,
        "per_seed": summaries,
        "mean_eval_return": float(np.mean([row["eval_return"] for row in summaries])),
        "std_eval_return": float(np.std([row["eval_return"] for row in summaries])),
        "mean_eval_length": float(np.mean([row["eval_length"] for row in summaries])),
    }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main()
