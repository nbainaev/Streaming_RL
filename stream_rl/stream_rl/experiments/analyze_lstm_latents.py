"""Relate recurrent LSTM states to ground-truth states of a finite POMDP.

The analysis deliberately separates episodes across train/test folds.  It
reports linear-probe accuracy, unsupervised clustering agreement, and the
geometry of class centroids, then writes compact data for visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from memorax.utils import Timestep
from memorax.utils.axes import remove_time_axis

from stream_rl.experiments.runners.base_runner import load_checkpoint, load_yaml
from stream_rl.src.agents.factory import build_agent
from stream_rl.src.env.factory import make_env


def _recurrent_states(carry):
    states = []
    for block_carry in carry:
        if (
            isinstance(block_carry, tuple)
            and len(block_carry) == 2
            and hasattr(block_carry[0], "shape")
            and hasattr(block_carry[1], "shape")
        ):
            states.append((block_carry[0], block_carry[1]))
    if not states:
        raise ValueError(f"No LSTM (c, h) carry found in structure: {jax.tree.structure(carry)}")
    return states


def _runner_init_key(seed):
    return jax.random.split(jax.random.key(seed))[0]


def collect_rollouts(agent, env, env_params, actor_params, episodes, seed):
    if not hasattr(env, "episode_length"):
        raise ValueError("Latent collector requires env.episode_length")
    reset_keys = jax.random.split(jax.random.key(seed), episodes)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
    action = jnp.zeros((episodes,), dtype=jnp.int32)
    reward = jnp.zeros((episodes,), dtype=jnp.float32)
    done = jnp.ones((episodes,), dtype=jnp.bool_)
    timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
    carry = agent.actor_network.initialize_carry((episodes, None))

    records = {"obs": [], "cue": [], "time": [], "action": [], "episode": []}
    layer_c, layer_h = [], []
    key = jax.random.key(seed + 1)

    for _ in range(env.episode_length):
        carry, (dist, _) = agent.actor_network.apply(
            actor_params,
            *timestep.to_sequence(),
            initial_carry=carry,
        )
        action = remove_time_axis(jnp.argmax(dist.logits, axis=-1)).astype(jnp.int32)
        recurrent = _recurrent_states(carry)
        layer_c.append([np.asarray(c) for c, _ in recurrent])
        layer_h.append([np.asarray(h) for _, h in recurrent])

        base_state = env_state.env_state if hasattr(env_state, "env_state") else env_state
        records["obs"].append(np.asarray(timestep.obs))
        records["cue"].append(np.asarray(base_state.cue))
        records["time"].append(np.asarray(base_state.time_step))
        records["action"].append(np.asarray(action))
        records["episode"].append(np.arange(episodes, dtype=np.int32))

        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, episodes)
        next_obs, env_state, next_reward, next_done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, env_state, action, env_params)
        timestep = Timestep(
            obs=next_obs,
            action=jnp.where(next_done, 0, action),
            reward=jnp.where(next_done, 0.0, next_reward),
            done=next_done,
        )

    flat = {key: np.concatenate(value, axis=0) for key, value in records.items()}
    num_layers = len(layer_c[0])
    for layer in range(num_layers):
        flat[f"layer_{layer}_c"] = np.concatenate([step[layer] for step in layer_c], axis=0)
        flat[f"layer_{layer}_h"] = np.concatenate([step[layer] for step in layer_h], axis=0)
    flat["joint"] = flat["cue"] * env.episode_length + flat["time"]
    flat["action_correct"] = (flat["action"] == flat["cue"]).astype(np.int32)
    return flat


def _standardize(train, test):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def grouped_ridge_probe(x, y, groups, folds=5, alpha=1e-2):
    classes = np.unique(y)
    class_to_index = {value: index for index, value in enumerate(classes)}
    y_index = np.asarray([class_to_index[value] for value in y])
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(unique_groups)
    group_fold = {group: index % folds for index, group in enumerate(shuffled)}
    predictions = np.empty_like(y_index)

    for fold in range(folds):
        test = np.asarray([group_fold[group] == fold for group in groups])
        train = ~test
        x_train, x_test = _standardize(x[train], x[test])
        x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
        x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
        targets = np.eye(len(classes))[y_index[train]]
        gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
        weights = np.linalg.solve(gram, x_train.T @ targets)
        predictions[test] = np.argmax(x_test @ weights, axis=1)
    return float(np.mean(predictions == y_index))


def _contingency(labels_a, labels_b):
    a_values, a = np.unique(labels_a, return_inverse=True)
    b_values, b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((len(a_values), len(b_values)), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def adjusted_rand_index(labels_a, labels_b):
    table = _contingency(labels_a, labels_b)
    n = table.sum()
    choose2 = lambda values: np.sum(values * (values - 1) / 2)
    sum_cells = choose2(table)
    sum_rows = choose2(table.sum(axis=1))
    sum_cols = choose2(table.sum(axis=0))
    total = n * (n - 1) / 2
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    return float((sum_cells - expected) / max(maximum - expected, 1e-12))


def normalized_mutual_information(labels_a, labels_b):
    table = _contingency(labels_a, labels_b).astype(np.float64)
    p = table / table.sum()
    pa = p.sum(axis=1, keepdims=True)
    pb = p.sum(axis=0, keepdims=True)
    nz = p > 0
    mutual = np.sum(p[nz] * np.log(p[nz] / (pa @ pb)[nz]))
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    return float(mutual / max(np.sqrt(ha * hb), 1e-12))


def clustering_metrics(x, labels, clusters):
    xz, _ = _standardize(x, x)
    best_assignment, best_inertia = None, np.inf
    for seed in range(5):
        centroids, assignment = kmeans2(xz, clusters, minit="++", iter=50, seed=seed)
        inertia = np.mean(np.sum((xz - centroids[assignment]) ** 2, axis=1))
        if inertia < best_inertia:
            best_assignment, best_inertia = assignment, inertia
    table = _contingency(best_assignment, labels)
    row, col = linear_sum_assignment(-table)
    aligned_accuracy = table[row, col].sum() / table.sum()
    return {
        "ari": adjusted_rand_index(best_assignment, labels),
        "nmi": normalized_mutual_information(best_assignment, labels),
        "aligned_accuracy": float(aligned_accuracy),
        "inertia": float(best_inertia),
    }


def pca_projection(x):
    xz, _ = _standardize(x, x)
    _, singular, vt = np.linalg.svd(xz, full_matrices=False)
    projection = xz @ vt[:2].T
    explained = singular**2 / np.sum(singular**2)
    return projection, explained[:2]


def centroid_geometry(x, cue, time):
    labels = sorted({(int(c), int(t)) for c, t in zip(cue, time)})
    centroids = np.asarray([x[(cue == c) & (time == t)].mean(axis=0) for c, t in labels])
    latent_distance, cue_distance, time_distance = [], [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            latent_distance.append(np.linalg.norm(centroids[i] - centroids[j]))
            cue_distance.append(float(labels[i][0] != labels[j][0]))
            time_distance.append(abs(labels[i][1] - labels[j][1]))
    return {
        "cue_distance_spearman": float(spearmanr(latent_distance, cue_distance).statistic),
        "time_distance_spearman": float(spearmanr(latent_distance, time_distance).statistic),
    }


def analyze(name, data, episode_length):
    last_layer = max(int(key.split("_")[1]) for key in data if key.startswith("layer_") and key.endswith("_h"))
    representations = {"observation": data["obs"]}
    for layer in range(last_layer + 1):
        c = data[f"layer_{layer}_c"]
        h = data[f"layer_{layer}_h"]
        representations[f"layer_{layer}_h"] = h
        representations[f"layer_{layer}_c"] = c
        representations[f"layer_{layer}_ch"] = np.concatenate([c, h], axis=1)
    blank = data["time"] > 0
    result = {
        "name": name,
        "episodes": int(np.unique(data["episode"]).size),
        "layers": last_layer + 1,
        "deterministic_action_accuracy_blank": float(data["action_correct"][blank].mean()),
        "deterministic_action_accuracy_by_time": {
            str(t): float(data["action_correct"][data["time"] == t].mean())
            for t in range(episode_length)
        },
        "representations": {},
    }
    for rep_name, x in representations.items():
        probes = {
            "cue_all": grouped_ridge_probe(x, data["cue"], data["episode"]),
            "cue_blank": grouped_ridge_probe(x[blank], data["cue"][blank], data["episode"][blank]),
            "time": grouped_ridge_probe(x, data["time"], data["episode"]),
            "joint_state": grouped_ridge_probe(x, data["joint"], data["episode"]),
        }
        cluster = {
            "cue": clustering_metrics(x, data["cue"], clusters=2),
            "time": clustering_metrics(x, data["time"], clusters=episode_length),
            "joint_state": clustering_metrics(x, data["joint"], clusters=2 * episode_length),
        }
        geometry = centroid_geometry(x, data["cue"], data["time"])
        result["representations"][rep_name] = {
            "probes": probes,
            "clustering": cluster,
            "geometry": geometry,
        }
    return result, representations[f"layer_{last_layer}_ch"]


def plot_centroid_trajectories(output, trained, untrained, cue, time):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, title, x in zip(axes, ["Untrained LSTM", "Trained LSTM"], [untrained, trained]):
        projection, explained = pca_projection(x)
        for cue_value, marker in [(0, "o"), (1, "s")]:
            points = np.asarray([
                projection[(cue == cue_value) & (time == t)].mean(axis=0)
                for t in sorted(np.unique(time))
            ])
            ax.plot(points[:, 0], points[:, 1], marker=marker, label=f"cue={cue_value}")
            for t, point in enumerate(points):
                ax.annotate(str(t), point, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"{title}\nPCA variance {explained.sum():.1%}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    agent_cfg = load_yaml(run_dir / "agent.yaml")
    env_cfg = load_yaml(run_dir / "env.yaml")
    env, env_params = make_env(env_cfg, num_envs=agent_cfg["num_envs"])
    agent = build_agent(agent_cfg["name"], agent_cfg, env, env_params)
    initial_state = agent.init(_runner_init_key(args.seed))
    trained_state = load_checkpoint(initial_state, run_dir, args.seed, args.step)

    trained = collect_rollouts(
        agent, env, env_params, trained_state.actor_params, args.episodes, args.seed + 1000
    )
    untrained = collect_rollouts(
        agent, env, env_params, initial_state.actor_params, args.episodes, args.seed + 1000
    )
    trained_metrics, trained_rep = analyze("trained", trained, env.episode_length)
    untrained_metrics, untrained_rep = analyze("untrained", untrained, env.episode_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": args.step,
        "environment": env_cfg,
        "trained": trained_metrics,
        "untrained": untrained_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(args.output_dir / "latents.npz", **{f"trained_{k}": v for k, v in trained.items()})
    plot_centroid_trajectories(
        args.output_dir / "centroid_pca.png",
        trained_rep,
        untrained_rep,
        trained["cue"],
        trained["time"],
    )

    rows = []
    projection, explained = pca_projection(trained_rep)
    for cue_value in sorted(np.unique(trained["cue"])):
        for time_value in sorted(np.unique(trained["time"])):
            mask = (trained["cue"] == cue_value) & (trained["time"] == time_value)
            point = projection[mask].mean(axis=0)
            rows.append({
                "cue": int(cue_value), "time": int(time_value),
                "pc1": float(point[0]), "pc2": float(point[1]),
            })
    with (args.output_dir / "centroid_pca.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cue", "time", "pc1", "pc2"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "pca_meta.json").write_text(json.dumps({"explained": explained.tolist()}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
