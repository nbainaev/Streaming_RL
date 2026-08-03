"""Post-hoc comparison of PPO and StreamAC recurrent representations.

Backbones remain frozen.  Every probe is fitted only after policy training,
with episode-disjoint train/test splits.  The script supports the local
Active/Passive T-maze implementations and POPGym RepeatPrevious.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2, vq

from memorax.utils import Timestep
from memorax.utils.axes import remove_time_axis

from stream_rl.experiments.analyze_lstm_latents import grouped_ridge_probe
from stream_rl.experiments.runners.base_runner import load_checkpoint, load_yaml
from stream_rl.src.agents.factory import build_agent
from stream_rl.src.env.factory import make_env


def runner_init_key(seed: int):
    return jax.random.split(jax.random.key(seed))[0]


def rtu_dynamics_summary(params):
    """Extract learned RTU decay/time-scale/phase distributions."""
    found = []

    def visit(node, path=()):
        if not isinstance(node, Mapping):
            return
        if "nu_log" in node and "theta_log" in node:
            nu_log = np.asarray(node["nu_log"], dtype=np.float64)
            theta_log = np.asarray(node["theta_log"], dtype=np.float64)
            radius = np.exp(-np.exp(nu_log))
            time_constant = np.exp(-nu_log)
            phase = np.exp(theta_log)
            found.append(
                {
                    "parameter_path": "/".join(path),
                    "units": int(len(radius)),
                    "radius_mean": float(radius.mean()),
                    "radius_min": float(radius.min()),
                    "radius_max": float(radius.max()),
                    "time_constant_q10": float(np.quantile(time_constant, 0.1)),
                    "time_constant_median": float(np.median(time_constant)),
                    "time_constant_q90": float(np.quantile(time_constant, 0.9)),
                    "phase_mean": float(phase.mean()),
                    "phase_std": float(phase.std()),
                }
            )
        for key, value in node.items():
            visit(value, (*path, str(key)))

    visit(params)
    return found


def recurrent_specs(agent_cfg: dict):
    return [
        (index, layer.get("cell", "gru"))
        for index, layer in enumerate(agent_cfg["actor_architecture"])
        if layer["type"] == "rnn"
    ]


def append_recurrent(records: dict, carry, specs):
    for layer_number, (block_index, cell) in enumerate(specs):
        value = carry[block_index]
        if cell == "lstm":
            c, h = value
            records.setdefault(f"layer_{layer_number}_c", []).append(np.asarray(c))
            records.setdefault(f"layer_{layer_number}_h", []).append(np.asarray(h))
        elif cell in {"rtu_rtrl", "rtu_bptt"}:
            # RTRL carry is (RTUCarry, sensitivity).  Sensitivities are
            # derivative bookkeeping rather than an agent representation.
            dynamics = value[0] if cell == "rtu_rtrl" else value
            real = np.asarray(dynamics.real)
            imaginary = np.asarray(dynamics.imaginary)
            records.setdefault(f"layer_{layer_number}_real", []).append(real)
            records.setdefault(f"layer_{layer_number}_imag", []).append(imaginary)
            records.setdefault(f"layer_{layer_number}_h", []).append(
                np.concatenate([real, imaginary], axis=-1)
            )
        else:
            records.setdefault(f"layer_{layer_number}_h", []).append(np.asarray(value))


def unwrap_state(env_state):
    return env_state.env_state if hasattr(env_state, "env_state") else env_state


def collect_tmaze(agent, env, params, actor_params, agent_cfg, episodes, seed):
    keys = jax.random.split(jax.random.key(seed), episodes)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    timestep = Timestep(
        obs=obs,
        action=jnp.zeros((episodes,), dtype=jnp.int32),
        reward=jnp.zeros((episodes,), dtype=jnp.float32),
        done=jnp.ones((episodes,), dtype=jnp.bool_),
    )
    carry = agent.actor_network.initialize_carry((episodes, None))
    specs = recurrent_specs(agent_cfg)
    records = {
        key: []
        for key in (
            "episode", "goal", "x", "y", "oracle", "time", "action",
            "previous_action", "observation",
        )
    }
    successes = []
    key = jax.random.key(seed + 1)

    for _ in range(env.episode_length):
        carry, (dist, _) = agent.actor_network.apply(
            actor_params, *timestep.to_sequence(), initial_carry=carry
        )
        action = remove_time_axis(jnp.argmax(dist.logits, axis=-1)).astype(jnp.int32)
        append_recurrent(records, carry, specs)
        state = unwrap_state(env_state)
        records["episode"].append(np.arange(episodes, dtype=np.int32))
        records["goal"].append(np.asarray((state.goal_y + 1) // 2))
        records["x"].append(np.asarray(state.x))
        records["y"].append(np.asarray(state.y + 1))
        records["oracle"].append(np.asarray(state.oracle_visited, dtype=np.int32))
        records["time"].append(np.asarray(state.time_step))
        records["action"].append(np.asarray(action))
        records["previous_action"].append(np.asarray(timestep.action))
        records["observation"].append(
            np.asarray(timestep.obs).reshape(episodes, -1)
        )

        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, episodes)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, env_state, action, params)
        if bool(np.asarray(next_done).all()):
            successes = np.asarray(info["success"], dtype=np.float32).tolist()
        timestep = Timestep(
            obs=next_obs,
            action=jnp.where(next_done, 0, action),
            reward=jnp.where(next_done, 0.0, next_reward),
            done=next_done,
        )

    data = {key: np.concatenate(value, axis=0) for key, value in records.items()}
    state_matrix = np.stack(
        [data["goal"], data["x"], data["y"], data["oracle"], data["time"]],
        axis=1,
    )
    _, data["task_state"] = np.unique(state_matrix, axis=0, return_inverse=True)
    data["position"] = data["x"] * 3 + data["y"]
    data["hidden_goal"] = data["goal"]
    labels = ["hidden_goal", "position", "time", "task_state"]
    return data, labels, {
        "success_rate": float(np.mean(successes)),
        "mean_return": float("nan"),
    }


def _native_popgym_states(vector_env):
    memory_rows, deck_rows = [], []
    for wrapped in vector_env.envs:
        base = wrapped.unwrapped
        memory, remaining_fraction = base.get_state()
        memory_rows.append(np.asarray(memory, dtype=np.int32))
        # RepeatPreviousEasy uses one 52-card deck, i.e. 13 cards per suit.
        # Convert the normalized remaining fractions back to exact counts.
        per_suit = base.deck.num_cards / 4
        deck_rows.append(
            np.rint(np.asarray(remaining_fraction) * per_suit).astype(np.int32)
        )
    return np.asarray(memory_rows), np.asarray(deck_rows)


def collect_popgym(agent, actor_params, agent_cfg, env_cfg, episodes, seed):
    import gymnasium
    import popgym  # noqa: F401

    vector_env = gymnasium.make_vec(env_cfg["env_id"], num_envs=episodes)
    obs, _ = vector_env.reset(seed=seed)
    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs[:, None]
    first_env = vector_env.envs[0].unwrapped
    episode_length = int(first_env.max_episode_length)
    memory_length = int(getattr(first_env, "k", 1))
    timestep = Timestep(
        obs=jnp.asarray(obs),
        action=jnp.zeros((episodes,), dtype=jnp.int32),
        reward=jnp.zeros((episodes,), dtype=jnp.float32),
        done=jnp.ones((episodes,), dtype=jnp.bool_),
    )
    carry = agent.actor_network.initialize_carry((episodes, None))
    specs = recurrent_specs(agent_cfg)
    records = {
        key: []
        for key in (
            "episode", "target", "time", "valid", "correct", "action",
            "previous_action", "observation", "memory_state", "deck_count_0",
            "deck_count_1", "deck_count_2", "deck_count_3",
        )
    }
    returns = np.zeros(episodes, dtype=np.float64)

    for time_index in range(episode_length):
        memory, deck_counts = _native_popgym_states(vector_env)
        target = memory[:, 0].astype(np.int32)
        memory_code = np.sum(
            memory * (4 ** np.arange(memory.shape[1], dtype=np.int64)), axis=1
        ).astype(np.int32)
        carry, (dist, _) = agent.actor_network.apply(
            actor_params, *timestep.to_sequence(), initial_carry=carry
        )
        action = np.asarray(
            remove_time_axis(jnp.argmax(dist.logits, axis=-1)), dtype=np.int32
        )
        append_recurrent(records, carry, specs)
        valid = np.full(episodes, time_index >= memory_length - 1, dtype=np.int32)
        records["episode"].append(np.arange(episodes, dtype=np.int32))
        records["target"].append(target)
        records["time"].append(np.full(episodes, time_index, dtype=np.int32))
        records["valid"].append(valid)
        records["correct"].append((action == target).astype(np.int32))
        records["action"].append(action)
        records["previous_action"].append(np.asarray(timestep.action))
        records["observation"].append(obs.reshape(episodes, -1))
        records["memory_state"].append(memory_code)
        for suit in range(4):
            records[f"deck_count_{suit}"].append(deck_counts[:, suit])

        next_obs, reward, terminated, truncated, _ = vector_env.step(action)
        next_done = np.asarray(terminated | truncated)
        returns += np.asarray(reward)
        next_obs = np.asarray(next_obs)
        if next_obs.ndim == 1:
            next_obs = next_obs[:, None]
        timestep = Timestep(
            obs=jnp.asarray(next_obs),
            action=jnp.asarray(np.where(next_done, 0, action)),
            reward=jnp.asarray(np.where(next_done, 0.0, reward), dtype=jnp.float32),
            done=jnp.asarray(next_done),
        )
        obs = next_obs

    vector_env.close()
    data = {key: np.concatenate(value, axis=0) for key, value in records.items()}
    data["hidden_target"] = data["target"]
    data["task_state"] = data["target"] * episode_length + data["time"]
    mask = data["valid"].astype(bool)
    labels = [
        "hidden_target", "time", "task_state", "memory_state",
        "deck_count_0", "deck_count_1", "deck_count_2", "deck_count_3",
    ]
    return data, labels, {
        "success_rate": float(data["correct"][mask].mean()),
        "mean_return": float(returns.mean()),
    }


def representations(data: dict):
    reps = {}
    layer_numbers = sorted(
        {int(key.split("_")[1]) for key in data if key.startswith("layer_")}
    )
    for layer in layer_numbers:
        h = data[f"layer_{layer}_h"]
        reps[f"layer_{layer}_h"] = h
        c_key = f"layer_{layer}_c"
        if c_key in data:
            c = data[c_key]
            reps[f"layer_{layer}_c"] = c
            reps[f"layer_{layer}_ch"] = np.concatenate([c, h], axis=1)
    return reps


def history_representation(data: dict, length=4):
    """Fixed recent observation/action history control.

    Comparing a frozen recurrent state with this control distinguishes useful
    state abstraction from information recoverable by merely retaining a
    short literal history.
    """
    obs = np.asarray(data["observation"], dtype=np.float64)
    if "memory_state" in data:
        categorical_parts = []
        for feature in range(obs.shape[1]):
            values = obs[:, feature].astype(np.int32)
            categorical_parts.append(np.eye(int(values.max()) + 1)[values])
        obs = np.concatenate(categorical_parts, axis=1)
    previous_action = np.asarray(data["previous_action"], dtype=np.int32)
    episodes = np.asarray(data["episode"])
    time = np.asarray(data["time"])
    num_episodes = len(np.unique(episodes))
    action_dim = int(max(previous_action.max(initial=0), data["action"].max(initial=0)) + 1)
    action_one_hot = np.eye(action_dim)[previous_action]
    input_vector = np.concatenate([obs, action_one_hot], axis=1)
    pieces = []
    for lag in range(length):
        shifted = np.zeros_like(input_vector)
        offset = lag * num_episodes
        if offset == 0:
            shifted = input_vector.copy()
        else:
            valid = (
                episodes[offset:] == episodes[:-offset]
            ) & (time[offset:] == time[:-offset] + lag)
            destination = np.arange(offset, len(input_vector))[valid]
            source = destination - offset
            shifted[destination] = input_vector[source]
        pieces.append(shifted)
    return np.concatenate(pieces, axis=1)


def transition_probe(x, data):
    """Decode next task state from frozen z_t and the selected action."""
    episodes = np.asarray(data["episode"])
    time = np.asarray(data["time"])
    num_episodes = len(np.unique(episodes))
    if len(x) <= num_episodes:
        return {"next_state_linear_probe_accuracy": float("nan")}
    current = np.arange(len(x) - num_episodes)
    following = current + num_episodes
    valid = (
        episodes[current] == episodes[following]
    ) & (time[following] == time[current] + 1)
    current, following = current[valid], following[valid]
    actions = np.asarray(data["action"], dtype=np.int32)
    action_dim = int(actions.max(initial=0) + 1)
    features = np.concatenate([x[current], np.eye(action_dim)[actions[current]]], axis=1)
    targets = np.asarray(data["task_state"])[following]
    accuracy = grouped_ridge_probe(features, targets, episodes[current])
    counts = np.bincount(targets)
    return {
        "next_state_linear_probe_accuracy": accuracy,
        "next_state_chance_accuracy": float(counts.max() / counts.sum()),
        "next_state_samples": int(len(targets)),
    }


def aliased_state_probe(x, data):
    """Decode environment state only where the current observation is aliased."""
    observation = np.round(np.asarray(data["observation"], dtype=np.float64), 6)
    _, observation_id = np.unique(observation, axis=0, return_inverse=True)
    state = np.asarray(data["task_state"])
    aliased_ids = {
        value
        for value in np.unique(observation_id)
        if len(np.unique(state[observation_id == value])) > 1
    }
    mask = np.asarray([value in aliased_ids for value in observation_id])
    if mask.sum() < 20 or len(np.unique(state[mask])) < 2:
        return {
            "aliased_state_linear_probe_accuracy": float("nan"),
            "aliased_state_chance_accuracy": float("nan"),
            "aliased_state_samples": int(mask.sum()),
        }
    accuracy = grouped_ridge_probe(
        x[mask], state[mask], np.asarray(data["episode"])[mask]
    )
    counts = np.bincount(state[mask])
    return {
        "aliased_state_linear_probe_accuracy": accuracy,
        "aliased_state_chance_accuracy": float(counts.max() / counts.sum()),
        "aliased_state_samples": int(mask.sum()),
    }


def cross_context_factorization_probe(x, data):
    """Test whether a state factor transfers to a held-out latent context.

    T-maze asks whether position geometry transfers across hidden goals;
    POPGym asks whether elapsed-time geometry transfers across card targets.
    """
    if "position" in data and "hidden_goal" in data:
        target_name, context_name = "position", "hidden_goal"
    elif "time" in data and "hidden_target" in data:
        target_name, context_name = "time", "hidden_target"
    else:
        return {"cross_context_accuracy": float("nan")}
    target = np.asarray(data[target_name])
    context = np.asarray(data[context_name])
    classes = np.unique(target)
    class_index = {value: index for index, value in enumerate(classes)}
    indexed_target = np.asarray([class_index[value] for value in target])
    predictions, truths = [], []
    for held_out in np.unique(context):
        train, test = context != held_out, context == held_out
        if train.sum() == 0 or test.sum() == 0:
            continue
        x_train, x_test = standardize(x[train], x[test])
        x_train = np.concatenate([x_train, np.ones((train.sum(), 1))], axis=1)
        x_test = np.concatenate([x_test, np.ones((test.sum(), 1))], axis=1)
        targets = np.eye(len(classes))[indexed_target[train]]
        gram = x_train.T @ x_train + 1e-2 * np.eye(x_train.shape[1])
        weights = np.linalg.solve(gram, x_train.T @ targets)
        predictions.append(np.argmax(x_test @ weights, axis=1))
        truths.append(indexed_target[test])
    if not predictions:
        return {"cross_context_accuracy": float("nan")}
    predicted = np.concatenate(predictions)
    truth = np.concatenate(truths)
    return {
        "cross_context_accuracy": float(np.mean(predicted == truth)),
        "cross_context_chance_accuracy": float(
            np.bincount(truth).max() / len(truth)
        ),
        "cross_context_target": target_name,
        "cross_context_variable": context_name,
    }


def graph_topology_alignment(x, data):
    """Compare latent centroid distances with empirical graph distances."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr

    labels = np.asarray(data["task_state"])
    unique = np.unique(labels)
    index = {label: i for i, label in enumerate(unique)}
    adjacency = np.full((len(unique), len(unique)), np.inf)
    np.fill_diagonal(adjacency, 0.0)
    episodes = np.asarray(data["episode"])
    time = np.asarray(data["time"])
    num_episodes = len(np.unique(episodes))
    current = np.arange(max(0, len(labels) - num_episodes))
    following = current + num_episodes
    valid = (
        episodes[current] == episodes[following]
    ) & (time[following] == time[current] + 1)
    for source, target in zip(labels[current[valid]], labels[following[valid]]):
        i, j = index[source], index[target]
        adjacency[i, j] = adjacency[j, i] = 1.0
    graph_distances = shortest_path(adjacency, directed=False)
    centroids = np.asarray([x[labels == label].mean(axis=0) for label in unique])
    latent_distances = squareform(pdist(centroids))
    triangle = np.triu_indices(len(unique), k=1)
    finite = np.isfinite(graph_distances[triangle])
    if finite.sum() < 3:
        return {
            "graph_distance_latent_spearman": float("nan"),
            "graph_distance_pairs": int(finite.sum()),
        }
    correlation = spearmanr(
        graph_distances[triangle][finite], latent_distances[triangle][finite]
    ).statistic
    return {
        "graph_distance_latent_spearman": float(correlation),
        "graph_distance_pairs": int(finite.sum()),
    }


def standardize(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def pca_reduce(train, test, dimensions=32):
    train, test = standardize(train, test)
    _, singular, vt = np.linalg.svd(train, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / max(variance.sum(), 1e-12)
    retained = min(dimensions, int(np.searchsorted(cumulative, 0.95) + 1), vt.shape[0])
    return train @ vt[:retained].T, test @ vt[:retained].T, retained


def homogeneity(labels, clusters):
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    _, y = np.unique(labels, return_inverse=True)
    _, z = np.unique(clusters, return_inverse=True)
    table = np.zeros((z.max() + 1, y.max() + 1), dtype=np.float64)
    np.add.at(table, (z, y), 1)
    p_y = table.sum(axis=0) / table.sum()
    entropy_y = -np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0]))
    conditional = 0.0
    for row in table:
        if row.sum() == 0:
            continue
        p_cluster = row.sum() / table.sum()
        probabilities = row / row.sum()
        conditional += p_cluster * -np.sum(
            probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])
        )
    return float(1.0 if entropy_y < 1e-12 else 1.0 - conditional / entropy_y)


def split_groups(groups, seed=0):
    unique = np.unique(groups)
    shuffled = np.random.default_rng(seed).permutation(unique)
    test_groups = set(shuffled[: max(1, len(shuffled) // 4)].tolist())
    test = np.asarray([group in test_groups for group in groups])
    return ~test, test


def heldout_cluster_metrics(x, labels, groups, seed=0):
    train, test = split_groups(groups, seed)
    x_train, x_test, retained = pca_reduce(x[train], x[test])
    y_train, y_test = labels[train], labels[test]
    classes = np.unique(labels)
    clusters = len(classes)
    if clusters > 128:
        return {
            "homogeneity": float("nan"),
            "state_prediction_accuracy": float("nan"),
            "chance_accuracy": float(np.max(np.bincount(labels)) / len(labels)),
            "clusters": int(clusters),
            "pca_dimensions": int(retained),
            "clustering_skipped": "more_than_128_ground_truth_states",
        }
    rng = np.random.default_rng(seed)
    fit_indices = rng.choice(
        len(x_train), size=min(12000, len(x_train)), replace=False
    )
    centroids, train_assignment = kmeans2(
        x_train[fit_indices], clusters, minit="++", iter=60, seed=seed
    )
    train_assignment, _ = vq(x_train, centroids)
    test_assignment, _ = vq(x_test, centroids)
    majority = {}
    global_majority = np.bincount(y_train).argmax()
    for cluster in range(clusters):
        values = y_train[train_assignment == cluster]
        majority[cluster] = (
            np.bincount(values).argmax() if len(values) else global_majority
        )
    predicted = np.asarray([majority[int(cluster)] for cluster in test_assignment])
    return {
        "homogeneity": homogeneity(y_test, test_assignment),
        "state_prediction_accuracy": float(np.mean(predicted == y_test)),
        "chance_accuracy": float(np.max(np.bincount(y_test)) / len(y_test)),
        "clusters": int(clusters),
        "pca_dimensions": int(retained),
    }


def tsne_plot(path: Path, x, labels, episodes, title, seed):
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for t-SNE plots") from exc

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(x), size=min(2500, len(x)), replace=False)
    x_selected, _ = standardize(x[selected], x[selected])
    embedding = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(selected) // 20)),
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(x_selected)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels[selected], s=8, alpha=0.65,
        cmap="tab20",
    )
    selected_episodes = episodes[selected]
    for episode in np.unique(selected_episodes)[:12]:
        mask = selected_episodes == episode
        if mask.sum() > 2:
            ax.plot(embedding[mask, 0], embedding[mask, 1], alpha=0.18, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="ground-truth state")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def centroid_geometry_comparisons(output_dir: Path, configs, seeds):
    from itertools import combinations
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    comparisons = []
    by_environment = {}
    for config in configs:
        by_environment.setdefault(config["environment"]["env_id"], []).append(config)
    for environment, entries in by_environment.items():
        for entry_a, entry_b in combinations(entries, 2):
            for seed in seeds:
                arrays = []
                for entry in (entry_a, entry_b):
                    path = (
                        output_dir / entry["run"] / f"seed_{seed}"
                        / "trained" / "latents.npz"
                    )
                    data = np.load(path)
                    h_keys = sorted(
                        key for key in data.files
                        if key.startswith("layer_") and key.endswith("_h")
                    )
                    x = data[h_keys[-1]]
                    labels = data["task_state"]
                    arrays.append((x, labels, h_keys[-1]))
                common = np.intersect1d(
                    np.unique(arrays[0][1]), np.unique(arrays[1][1])
                )
                centroids = [
                    np.asarray([x[labels == label].mean(axis=0) for label in common])
                    for x, labels, _ in arrays
                ]
                grams = []
                for centroid in centroids:
                    centered = centroid - centroid.mean(axis=0, keepdims=True)
                    gram = centered @ centered.T
                    center = (
                        np.eye(len(common))
                        - np.ones((len(common), len(common))) / len(common)
                    )
                    grams.append(center @ gram @ center)
                cka = float(
                    np.sum(grams[0] * grams[1])
                    / max(
                        np.sqrt(np.sum(grams[0] ** 2) * np.sum(grams[1] ** 2)),
                        1e-12,
                    )
                )
                rdm_correlation = float(
                    spearmanr(pdist(centroids[0]), pdist(centroids[1])).statistic
                )
                comparisons.append(
                    {
                        "environment": environment,
                        "seed": seed,
                        "run_a": entry_a["run"],
                        "agent_a": entry_a["agent"]["name"],
                        "run_b": entry_b["run"],
                        "agent_b": entry_b["agent"]["name"],
                        "states": int(len(common)),
                        "centroid_linear_cka": cka,
                        "centroid_rdm_spearman": rdm_correlation,
                    }
                )
    return comparisons


def analyze_run(run_dir: Path, seeds, step, episodes, output_dir: Path):
    agent_cfg = load_yaml(run_dir / "agent.yaml")
    env_cfg = load_yaml(run_dir / "env.yaml")
    env, params = make_env(env_cfg, num_envs=agent_cfg["num_envs"])
    agent = build_agent(agent_cfg["name"], agent_cfg, env, params)
    rows = []
    rtu_summaries = []
    for seed in seeds:
        initial = agent.init(runner_init_key(seed))
        trained = load_checkpoint(initial, run_dir, seed, step)
        trained_rtu = rtu_dynamics_summary(trained.actor_params)
        if trained_rtu:
            rtu_summaries.append(
                {"run": run_dir.name, "seed": seed, "layers": trained_rtu}
            )
        for condition, actor_params in (
            ("trained", trained.actor_params),
            ("untrained", initial.actor_params),
        ):
            if env_cfg["namespace"] == "popgym":
                data, labels, performance = collect_popgym(
                    agent, actor_params, agent_cfg, env_cfg, episodes, seed + 10000
                )
            else:
                data, labels, performance = collect_tmaze(
                    agent, env, params, actor_params, agent_cfg, episodes, seed + 10000
                )
            reps = representations(data)
            reps["history_4"] = history_representation(data, length=4)
            seed_dir = output_dir / run_dir.name / f"seed_{seed}" / condition
            seed_dir.mkdir(parents=True, exist_ok=True)
            recurrent_rep_names = sorted(
                name for name in reps if name.startswith("layer_") and name.endswith("_h")
            )
            last_rep_name = recurrent_rep_names[-1]
            if condition == "trained":
                tsne_plot(
                    seed_dir / "tsne.png",
                    reps[last_rep_name],
                    data["task_state"],
                    data["episode"],
                    f"{run_dir.name}, seed {seed}, {last_rep_name}",
                    seed,
                )
            for rep_name, x in reps.items():
                for label_name in labels:
                    mask = np.ones(len(x), dtype=bool)
                    if env_cfg["namespace"] == "popgym" and label_name == "hidden_target":
                        mask = data["valid"].astype(bool)
                    probe = grouped_ridge_probe(
                        x[mask], data[label_name][mask], data["episode"][mask]
                    )
                    cluster = heldout_cluster_metrics(
                        x[mask], data[label_name][mask], data["episode"][mask], seed
                    )
                    structural = {}
                    if label_name == "task_state":
                        structural = {
                            **transition_probe(x, data),
                            **aliased_state_probe(x, data),
                            **cross_context_factorization_probe(x, data),
                            **graph_topology_alignment(x, data),
                        }
                    rows.append(
                        {
                            "run": run_dir.name,
                            "agent": agent_cfg["name"],
                            "seed": seed,
                            "condition": condition,
                            "environment": env_cfg["env_id"],
                            "representation": rep_name,
                            "label": label_name,
                            "linear_probe_accuracy": probe,
                            **cluster,
                            **structural,
                            **performance,
                        }
                    )
            np.savez_compressed(seed_dir / "latents.npz", **data)
    return rows, agent_cfg, env_cfg, rtu_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--step", type=int, default=102400)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, configs, rtu_summaries = [], [], []
    for run_dir in args.run_dirs:
        run_rows, agent_cfg, env_cfg, run_rtu_summaries = analyze_run(
            run_dir.resolve(), seeds, args.step, args.episodes, args.output_dir.resolve()
        )
        rows.extend(run_rows)
        rtu_summaries.extend(run_rtu_summaries)
        configs.append({"run": run_dir.name, "agent": agent_cfg, "environment": env_cfg})

    # Correlate behavior with representation organization across independent
    # seeds/models, using the final recurrent representation and task state.
    selected = [
        row for row in rows
        if row["label"] == "task_state"
        and row["condition"] == "trained"
        and row["representation"] == max(
            candidate["representation"] for candidate in rows
            if candidate["run"] == row["run"] and candidate["seed"] == row["seed"]
        )
    ]
    if len(selected) >= 3:
        from scipy.stats import spearmanr

        correlation = {
            "homogeneity_vs_success_spearman": float(
                spearmanr(
                    [row["homogeneity"] for row in selected],
                    [row["success_rate"] for row in selected],
                ).statistic
            ),
            "linear_probe_vs_success_spearman": float(
                spearmanr(
                    [row["linear_probe_accuracy"] for row in selected],
                    [row["success_rate"] for row in selected],
                ).statistic
            ),
            "n": len(selected),
        }
    else:
        correlation = {"n": len(selected)}
    correlations_by_label = {}
    trained_last_h = [
        row for row in rows
        if row["condition"] == "trained"
        and row["representation"].startswith("layer_")
        and row["representation"].endswith("_h")
        and row["representation"] == max(
            candidate["representation"] for candidate in rows
            if candidate["run"] == row["run"]
            and candidate["seed"] == row["seed"]
            and candidate["condition"] == "trained"
            and candidate["representation"].startswith("layer_")
            and candidate["representation"].endswith("_h")
        )
    ]
    if len(trained_last_h) >= 3:
        from scipy.stats import spearmanr

        for label in sorted({row["label"] for row in trained_last_h}):
            label_rows = [row for row in trained_last_h if row["label"] == label]
            homogeneity_values = np.asarray(
                [row["homogeneity"] for row in label_rows], dtype=np.float64
            )
            probe_values = np.asarray(
                [row["linear_probe_accuracy"] for row in label_rows],
                dtype=np.float64,
            )
            success_values = np.asarray(
                [row["success_rate"] for row in label_rows], dtype=np.float64
            )
            finite_h = np.isfinite(homogeneity_values) & np.isfinite(success_values)
            finite_p = np.isfinite(probe_values) & np.isfinite(success_values)
            correlations_by_label[label] = {
                "homogeneity_vs_success_spearman": (
                    float(spearmanr(homogeneity_values[finite_h], success_values[finite_h]).statistic)
                    if finite_h.sum() >= 3 else float("nan")
                ),
                "linear_probe_vs_success_spearman": (
                    float(spearmanr(probe_values[finite_p], success_values[finite_p]).statistic)
                    if finite_p.sum() >= 3 else float("nan")
                ),
                "n": len(label_rows),
            }
    geometry = centroid_geometry_comparisons(args.output_dir.resolve(), configs, seeds)
    result = {
        "rows": rows,
        "correlations": correlation,
        "correlations_by_label": correlations_by_label,
        "pairwise_centroid_geometry": geometry,
        "ppo_stream_geometry": geometry,
        "rtu_dynamics": rtu_summaries,
        "configs": configs,
    }
    (args.output_dir / "representation_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=True)
    )
    print(json.dumps(correlation, indent=2))


if __name__ == "__main__":
    main()
