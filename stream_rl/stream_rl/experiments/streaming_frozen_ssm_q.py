"""Strictly-online frozen-SSM Q-readout for classic passive T-maze.

One transition produces one semi-gradient Q(0) update.  There is no replay,
rollout batch, target network, or backpropagation through history.  The
pretrained SSM is fixed; only a linear action-value readout is learned.

The observation's existing ``position`` channel supplies a phase gate:
navigation features never see the cue, and the frozen-memory features become
visible to the readout only at the junction.  This is an architectural
factorization of the observation, not an oracle goal label or reward shaping.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from stream_rl.src.env.tmaze import TMazeClassicActive, TMazeClassicPassive


def load_ssm(path: Path):
    with np.load(path) as data:
        return tuple(
            jnp.asarray(data[name], dtype=jnp.float32)
            for name in ("decay", "B", "C", "D")
        )


def frozen_encoder_matrix(seed: int, input_dim: int = 2, embed_dim: int = 64):
    matrix = jax.random.normal(
        jax.random.key(seed), (input_dim, embed_dim), dtype=jnp.float32
    )
    return matrix / jnp.sqrt(float(input_dim))


def encode(observation, matrix):
    projected = jnp.tanh(observation @ matrix)
    # Preserve the environment's already-observed junction indicator exactly.
    return projected.at[0].set(observation[0])


def memory_step(hidden, embedding, constants):
    decay, input_matrix, readout_matrix, residual_matrix = constants
    hidden = decay * hidden + input_matrix @ embedding
    output = jnp.tanh(readout_matrix @ hidden + residual_matrix @ embedding)
    return hidden, output


def readout_features(observation, memory, use_memory: bool, env_kind: str):
    position = jnp.clip(observation[0], 0.0, 1.0)
    if env_kind == "active":
        # Zero before the oracle, positive after the cue has been written.
        # The norm hides cue sign from navigation while marking visit phase.
        memory_written = (
            jnp.sqrt(jnp.mean(jnp.square(memory)) + 1e-8)
            if use_memory
            else jnp.asarray(0.0, dtype=jnp.float32)
        )
        navigation = jnp.asarray(
            [1.0, position, memory_written], dtype=jnp.float32
        )
    else:
        navigation = jnp.asarray([1.0, position], dtype=jnp.float32)
    if not use_memory:
        return navigation
    return jnp.concatenate([navigation, position * memory])


def mask_observable_invalid_actions(q_values, observation, env_kind: str):
    """Mask non-branch actions at the visibly marked active-T-maze junction."""
    if env_kind != "active":
        return q_values
    at_junction = observation[0] >= 0.5
    branch_mask = jnp.asarray([False, True, False, True])
    invalid = at_junction & (~branch_mask)
    return jnp.where(invalid, -jnp.inf, q_values)


def train_seed(
    seed: int,
    total_steps: int,
    corridor_length: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    trace_lambda: float,
    constants,
    encoder_matrix,
    use_memory: bool,
    env_kind: str,
):
    env = (
        TMazeClassicActive(corridor_length=corridor_length)
        if env_kind == "active"
        else TMazeClassicPassive(corridor_length=corridor_length)
    )
    params = env.default_params
    navigation_dim = 3 if env_kind == "active" else 2
    feature_dim = navigation_dim + (64 if use_memory else 0)

    reset_key, loop_key = jax.random.split(jax.random.key(seed))
    observation, env_state = env.reset(reset_key, params)
    hidden = jnp.zeros((128,), dtype=jnp.float32)
    hidden, memory = memory_step(hidden, encode(observation, encoder_matrix), constants)
    features = readout_features(observation, memory, use_memory, env_kind)
    weights = jnp.zeros((feature_dim, env.num_actions), dtype=jnp.float32)
    traces = jnp.zeros_like(weights)

    def step(carry, step_index):
        (
            key,
            observation,
            env_state,
            hidden,
            memory,
            features,
            weights,
            traces,
            episode_return,
        ) = carry
        action_key, explore_key, env_key, reset_key, next_key = jax.random.split(key, 5)
        fraction = step_index.astype(jnp.float32) / max(float(total_steps - 1), 1.0)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        q_values = mask_observable_invalid_actions(
            features @ weights, observation, env_kind
        )
        # Random jitter makes initial/tied greedy actions unbiased by action ID.
        greedy_action = jnp.argmax(
            q_values + 1e-6 * jax.random.uniform(action_key, q_values.shape)
        )
        random_action = jax.random.randint(
            action_key, shape=(), minval=0, maxval=env.num_actions
        )
        if env_kind == "active":
            random_branch = 1 + 2 * jax.random.randint(
                action_key, shape=(), minval=0, maxval=2
            )
            random_action = jnp.where(
                observation[0] >= 0.5, random_branch, random_action
            )
        action = jnp.where(
            jax.random.bernoulli(explore_key, epsilon), random_action, greedy_action
        )

        next_observation, next_env_state, reward, done, info = env.step_env(
            env_key, env_state, action, params
        )
        next_hidden, next_memory = memory_step(
            hidden, encode(next_observation, encoder_matrix), constants
        )
        next_features = readout_features(
            next_observation, next_memory, use_memory, env_kind
        )
        next_q_values = mask_observable_invalid_actions(
            next_features @ weights, next_observation, env_kind
        )
        bootstrap = jnp.max(next_q_values)
        target = reward + gamma * (1.0 - done.astype(jnp.float32)) * bootstrap
        td_error = target - q_values[action]
        traces = gamma * trace_lambda * traces
        traces = traces.at[:, action].add(features)
        normalized_alpha = alpha / jnp.maximum(1.0, jnp.vdot(features, features))
        weights = weights + normalized_alpha * td_error * traces

        completed_return = episode_return + reward
        reset_observation, reset_state = env.reset_env(reset_key, params)
        reset_hidden = jnp.zeros_like(hidden)
        reset_hidden, reset_memory = memory_step(
            reset_hidden, encode(reset_observation, encoder_matrix), constants
        )
        reset_features = readout_features(
            reset_observation, reset_memory, use_memory, env_kind
        )

        observation = jnp.where(done, reset_observation, next_observation)
        env_state = jax.tree.map(
            lambda reset_value, next_value: jnp.where(
                done, reset_value, next_value
            ),
            reset_state,
            next_env_state,
        )
        hidden = jnp.where(done, reset_hidden, next_hidden)
        memory = jnp.where(done, reset_memory, next_memory)
        features = jnp.where(done, reset_features, next_features)
        episode_return = jnp.where(done, 0.0, completed_return)
        traces = jnp.where(done, jnp.zeros_like(traces), traces)
        logs = {
            "done": done,
            "success": info["success"],
            "episode_return": jnp.where(done, completed_return, jnp.nan),
            "td_error": td_error,
            "epsilon": epsilon,
            "action": action,
        }
        return (
            next_key,
            observation,
            env_state,
            hidden,
            memory,
            features,
            weights,
            traces,
            episode_return,
        ), logs

    initial = (
        loop_key,
        observation,
        env_state,
        hidden,
        memory,
        features,
        weights,
        traces,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    final, logs = jax.jit(
        lambda state: jax.lax.scan(step, state, jnp.arange(total_steps))
    )(initial)
    return final[6], jax.tree.map(np.asarray, logs)


def evaluate_seed(
    seed: int,
    weights,
    episodes: int,
    corridor_length: int,
    constants,
    encoder_matrix,
    use_memory: bool,
    env_kind: str,
):
    """Fresh-episode greedy evaluation with frozen weights and no updates."""
    env = (
        TMazeClassicActive(corridor_length=corridor_length)
        if env_kind == "active"
        else TMazeClassicPassive(corridor_length=corridor_length)
    )
    params = env.default_params
    total_steps = episodes * env.episode_length
    reset_key, loop_key = jax.random.split(jax.random.key(seed + 100_000))
    observation, env_state = env.reset(reset_key, params)
    hidden = jnp.zeros((128,), dtype=jnp.float32)
    hidden, memory = memory_step(
        hidden, encode(observation, encoder_matrix), constants
    )
    features = readout_features(observation, memory, use_memory, env_kind)

    def step(carry, _):
        key, observation, env_state, hidden, memory, features, episode_return = carry
        action_key, env_key, reset_key, next_key = jax.random.split(key, 4)
        q_values = mask_observable_invalid_actions(
            features @ weights, observation, env_kind
        )
        action = jnp.argmax(
            q_values + 1e-6 * jax.random.uniform(action_key, q_values.shape)
        )
        next_observation, next_env_state, reward, done, info = env.step_env(
            env_key, env_state, action, params
        )
        next_hidden, next_memory = memory_step(
            hidden, encode(next_observation, encoder_matrix), constants
        )
        next_features = readout_features(
            next_observation, next_memory, use_memory, env_kind
        )
        completed_return = episode_return + reward

        reset_observation, reset_state = env.reset_env(reset_key, params)
        reset_hidden, reset_memory = memory_step(
            jnp.zeros_like(hidden),
            encode(reset_observation, encoder_matrix),
            constants,
        )
        reset_features = readout_features(
            reset_observation, reset_memory, use_memory, env_kind
        )
        carry = (
            next_key,
            jnp.where(done, reset_observation, next_observation),
            jax.tree.map(
                lambda reset_value, next_value: jnp.where(
                    done, reset_value, next_value
                ),
                reset_state,
                next_env_state,
            ),
            jnp.where(done, reset_hidden, next_hidden),
            jnp.where(done, reset_memory, next_memory),
            jnp.where(done, reset_features, next_features),
            jnp.where(done, 0.0, completed_return),
        )
        return carry, {
            "done": done,
            "success": info["success"],
            "episode_return": jnp.where(done, completed_return, jnp.nan),
        }

    initial = (
        loop_key,
        observation,
        env_state,
        hidden,
        memory,
        features,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    _, logs = jax.jit(
        lambda state: jax.lax.scan(step, state, xs=None, length=total_steps)
    )(initial)
    return jax.tree.map(np.asarray, logs)


def episode_rows(seed: int, logs):
    done_indices = np.flatnonzero(logs["done"])
    return [
        {
            "seed": seed,
            "step": int(index + 1),
            "episode_return": float(logs["episode_return"][index]),
            "success": float(logs["success"][index]),
        }
        for index in done_indices
    ]


def summarize(rows):
    return {
        "episodes": len(rows),
        "last_100_return": float(np.mean([r["episode_return"] for r in rows[-100:]])),
        "last_100_success": float(np.mean([r["success"] for r in rows[-100:]])),
        "last_500_return": float(np.mean([r["episode_return"] for r in rows[-500:]])),
        "last_500_success": float(np.mean([r["success"] for r in rows[-500:]])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--env", choices=("passive", "active"), default="passive")
    parser.add_argument("--corridor-length", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--trace-lambda", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Frozen-readout evaluation lengths; defaults to the training length.",
    )
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("stream_rl/experiments/checkpoints/frozen_ssm_delayed_recall.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    constants = load_ssm(args.checkpoint)
    encoder_matrix = frozen_encoder_matrix(220512258)
    summaries = []
    eval_lengths = args.eval_lengths or [args.corridor_length]
    for seed in args.seeds:
        weights, logs = train_seed(
            seed=seed,
            total_steps=args.steps,
            corridor_length=args.corridor_length,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            trace_lambda=args.trace_lambda,
            constants=constants,
            encoder_matrix=encoder_matrix,
            use_memory=not args.no_memory,
            env_kind=args.env,
        )
        rows = episode_rows(seed, logs)
        eval_by_length = {}
        eval_rows_by_length = {}
        for eval_length in eval_lengths:
            eval_logs = evaluate_seed(
                seed=seed,
                weights=weights,
                episodes=args.eval_episodes,
                corridor_length=eval_length,
                constants=constants,
                encoder_matrix=encoder_matrix,
                use_memory=not args.no_memory,
                env_kind=args.env,
            )
            eval_rows = episode_rows(seed, eval_logs)
            eval_rows_by_length[eval_length] = eval_rows
            eval_by_length[str(eval_length)] = {
                "episodes": len(eval_rows),
                "return": float(
                    np.mean([row["episode_return"] for row in eval_rows])
                ),
                "success": float(
                    np.mean([row["success"] for row in eval_rows])
                ),
            }
        with (args.output / f"monitor_seed_{seed}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        for eval_length, eval_rows in eval_rows_by_length.items():
            with (args.output / f"eval_L{eval_length}_seed_{seed}.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=eval_rows[0].keys())
                writer.writeheader()
                writer.writerows(eval_rows)
        train_length_eval = eval_by_length.get(str(args.corridor_length))
        summary = {
            "seed": seed,
            "use_memory": not args.no_memory,
            **summarize(rows),
            "eval_episodes": args.eval_episodes,
            "eval_by_length": eval_by_length,
            "eval_return": (
                train_length_eval["return"] if train_length_eval else None
            ),
            "eval_success": (
                train_length_eval["success"] if train_length_eval else None
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))

    aggregate = {
        "strict_streaming": True,
        "algorithm": "linear Q(0) readout over frozen pretrained SSM",
        "environment": f"{args.env}_tmaze",
        "corridor_length": args.corridor_length,
        "steps": args.steps,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "trace_lambda": args.trace_lambda,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "eval_episodes": args.eval_episodes,
        "eval_lengths": eval_lengths,
        "use_memory": not args.no_memory,
        "per_seed": summaries,
        "mean_last_500_success": float(
            np.mean([summary["last_500_success"] for summary in summaries])
        ),
        "mean_last_500_return": float(
            np.mean([summary["last_500_return"] for summary in summaries])
        ),
        "mean_eval_success": (
            float(np.mean([summary["eval_success"] for summary in summaries]))
            if all(summary["eval_success"] is not None for summary in summaries)
            else None
        ),
        "mean_eval_return": (
            float(np.mean([summary["eval_return"] for summary in summaries]))
            if all(summary["eval_return"] is not None for summary in summaries)
            else None
        ),
        "mean_eval_by_length": {
            str(length): {
                "return": float(
                    np.mean(
                        [
                            summary["eval_by_length"][str(length)]["return"]
                            for summary in summaries
                        ]
                    )
                ),
                "success": float(
                    np.mean(
                        [
                            summary["eval_by_length"][str(length)]["success"]
                            for summary in summaries
                        ]
                    )
                ),
            }
            for length in eval_lengths
        },
    }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main()
