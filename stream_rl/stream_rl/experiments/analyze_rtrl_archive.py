"""Audit and visualize the GitHub ``stream_eprop`` log archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


RETURN = "info.returned_episode_returns"
SUCCESS = "info.success"
SEEDS = (0, 42, 123)


METHOD_LABELS = {
    "ac_eprop_gru_symmetric": "GRU e-prop (symmetric)",
    "ac_eprop_rnn": "RNN e-prop",
    "ac_gru_bptt1": "GRU one-step",
    "ac_gru_bptt5": "GRU nominal TBPTT(5)",
    "ac_lstm": "LSTM one-step",
    "ac_rtu_bptt1": "RTU one-step",
    "ac_rtu_bptt5": "RTU nominal TBPTT(5)",
    "ac_rtu_rtrl": "RTU exact RTRL",
    "ppo_gru": "PPO-GRU",
}


def _read_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def _method(run_dir: Path) -> str:
    variant = run_dir.parent.name
    for fragment, label in METHOD_LABELS.items():
        if fragment in variant:
            return label
    return variant


def _task(run_dir: Path, env: dict) -> str:
    env_id = str(env.get("env_id", ""))
    if "popgym" in env_id.lower():
        name = env_id.split("-")[-1].removesuffix("-v0")
        return name
    match = re.search(r"_L(\d+)(?:__|$)", run_dir.name)
    return f"ActiveTmazeL{match.group(1)}" if match else env_id


def _has_layernorm(agent: dict) -> bool:
    return any(
        layer.get("type") == "layernorm"
        for key in ("actor_architecture", "critic_architecture")
        for layer in agent.get(key, [])
    )


def _cohort(agent: dict) -> str:
    if agent.get("name") == "ppo":
        return "tuned" if _has_layernorm(agent) else "legacy"
    if agent.get("name") == "stream_eprop":
        return "tuned" if agent.get("adaptive", False) else "legacy"
    return "tuned" if agent.get("adaptive", False) and _has_layernorm(agent) else "legacy"


def _read_monitor(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    returns = [float(row[RETURN]) for row in rows if row.get(RETURN, "")]
    successes = [float(row[SUCCESS]) for row in rows if row.get(SUCCESS, "")]
    steps = [int(float(row.get("total_steps") or row.get("step") or 0)) for row in rows]
    tail = min(500, len(returns))
    final_return = float(np.mean(returns[-tail:])) if tail else math.nan
    final_success = float(np.mean(successes[-min(500, len(successes)):])) if successes else math.nan
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "returns": returns,
        "successes": successes,
        "steps": steps,
        "episodes": len(returns),
        "max_step": max(steps, default=0),
        "final_return": final_return,
        "final_success": final_success,
        "finite": bool(np.isfinite(returns).all()) if returns else False,
        "sha256": digest,
    }


def collect(logs: Path):
    records = []
    for agent_path in sorted(logs.rglob("agent.yaml")):
        run_dir = agent_path.parent
        env_path = run_dir / "env.yaml"
        runner_path = run_dir / "runner.yaml"
        if not env_path.exists() or not runner_path.exists():
            continue
        agent, env, runner = _read_yaml(agent_path), _read_yaml(env_path), _read_yaml(runner_path)
        for seed in SEEDS:
            monitor_path = run_dir / f"monitor_seed_{seed}.csv"
            if not monitor_path.exists():
                continue
            monitor = _read_monitor(monitor_path)
            records.append(
                {
                    "method": _method(run_dir),
                    "task": _task(run_dir, env),
                    "cohort": _cohort(agent),
                    "seed": seed,
                    "path": str(run_dir),
                    "runner_steps": int(runner.get("total_timesteps", 0)),
                    "num_envs": int(agent.get("num_envs", -1)),
                    "tbptt_steps": int(agent.get("tbptt_steps", 1)),
                    "agent": agent,
                    **monitor,
                }
            )
    return records


def deduplicate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["task"], record["cohort"], record["seed"])].append(record)
    selected, duplicate_report = [], []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: (row["max_step"], row["episodes"], row["path"]))
        chosen = values[-1]
        selected.append(chosen)
        if len(values) > 1:
            duplicate_report.append(
                {
                    "method": key[0],
                    "task": key[1],
                    "cohort": key[2],
                    "seed": key[3],
                    "candidates": len(values),
                    "unique_files": len({row["sha256"] for row in values}),
                    "chosen": chosen["path"],
                }
            )
    return selected, duplicate_report


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], record["task"], record["cohort"])].append(record)
    rows = []
    for (method, task, cohort), values in sorted(groups.items()):
        returns = np.asarray([row["final_return"] for row in values], dtype=float)
        success = np.asarray([row["final_success"] for row in values], dtype=float)
        rows.append(
            {
                "method": method,
                "task": task,
                "cohort": cohort,
                "n_seeds": len(values),
                "seeds": sorted(row["seed"] for row in values),
                "return_mean": float(np.nanmean(returns)),
                "return_sd": float(np.nanstd(returns, ddof=1)) if len(returns) > 1 else 0.0,
                "success_mean": float(np.nanmean(success)) if np.isfinite(success).any() else math.nan,
                "success_sd": float(np.nanstd(success, ddof=1)) if np.isfinite(success).sum() > 1 else 0.0,
                "max_step_min": min(row["max_step"] for row in values),
                "all_finite": all(row["finite"] for row in values),
                "num_envs": sorted({row["num_envs"] for row in values}),
                "tbptt_steps": sorted({row["tbptt_steps"] for row in values}),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rolling_curve(record, bins=120):
    steps = np.asarray(record["steps"], dtype=float)
    values = np.asarray(record["returns"], dtype=float)
    if len(values) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(0, max(steps.max(), 1), bins + 1)
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (steps >= lo) & (steps < hi)
        if mask.any():
            x.append((lo + hi) / 2)
            y.append(values[mask].mean())
    return np.asarray(x), np.asarray(y)


def plot_learning_curves(records, output: Path):
    tasks = ["RepeatPreviousEasy", "AutoencodeEasy", "ActiveTmazeL5"]
    methods = [
        "PPO-GRU",
        "RTU exact RTRL",
        "RTU one-step",
        "GRU one-step",
        "GRU e-prop (symmetric)",
        "LSTM one-step",
    ]
    colors = dict(zip(methods, plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, task in zip(axes, tasks):
        for method in methods:
            curves = []
            subset = [
                row for row in records
                if row["task"] == task and row["method"] == method and row["cohort"] == "tuned"
            ]
            if not subset:
                continue
            max_step = min(row["max_step"] for row in subset)
            grid = np.linspace(0, max_step, 100)
            for row in subset:
                x, y = _rolling_curve(row)
                if len(x) > 1:
                    curves.append(np.interp(grid, x, y))
            if not curves:
                continue
            values = np.asarray(curves)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
            axis.plot(grid, mean, label=method, color=colors[method], linewidth=1.7)
            axis.fill_between(grid, mean - sd, mean + sd, color=colors[method], alpha=0.15)
        axis.set_title(task)
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Episode return, mean ± seed SD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(output / "archive_learning_curves.png", dpi=180)
    plt.close(fig)


def plot_final_comparison(summary, output: Path):
    tasks = ["RepeatPreviousEasy", "AutoencodeEasy", "ActiveTmazeL5"]
    methods = [
        "PPO-GRU",
        "RTU exact RTRL",
        "RTU one-step",
        "RTU nominal TBPTT(5)",
        "GRU one-step",
        "GRU nominal TBPTT(5)",
        "GRU e-prop (symmetric)",
        "LSTM one-step",
    ]
    lookup = {(row["task"], row["method"]): row for row in summary if row["cohort"] == "tuned"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, task in zip(axes, tasks):
        labels, means, errors = [], [], []
        for method in methods:
            row = lookup.get((task, method))
            if row is None:
                continue
            labels.append(method)
            metric = "success" if task.startswith("ActiveTmaze") else "return"
            means.append(row[f"{metric}_mean"])
            errors.append(row[f"{metric}_sd"])
        y = np.arange(len(labels))
        axis.barh(y, means, xerr=errors, color="#457b9d", alpha=0.9)
        axis.set_yticks(y, labels, fontsize=7)
        axis.axvline(0.0 if not task.startswith("ActiveTmaze") else 0.5, color="#b23a48", linestyle="--", linewidth=0.9)
        axis.set_title(task)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    axes[0].set_xlabel("Final return")
    axes[1].set_xlabel("Final return")
    axes[2].set_xlabel("Final success rate")
    fig.tight_layout()
    fig.savefig(output / "archive_final_comparison.png", dpi=180)
    plt.close(fig)


def plot_nominal_horizon(records, output: Path):
    pairs = []
    mapping = {
        "GRU one-step": "GRU nominal TBPTT(5)",
        "RTU one-step": "RTU nominal TBPTT(5)",
    }
    lookup = {(r["method"], r["task"], r["cohort"], r["seed"]): r for r in records}
    for one, five in mapping.items():
        for key, row in lookup.items():
            method, task, cohort, seed = key
            if method != one:
                continue
            peer = lookup.get((five, task, cohort, seed))
            if peer:
                pairs.append((one.split()[0], row["final_return"], peer["final_return"]))
    fig, axis = plt.subplots(figsize=(5.3, 5.0))
    for family, marker in (("GRU", "o"), ("RTU", "s")):
        subset = [(x, y) for name, x, y in pairs if name == family]
        if subset:
            axis.scatter([x for x, _ in subset], [y for _, y in subset], label=family, alpha=0.65, marker=marker)
    values = [value for _, x, y in pairs for value in (x, y)]
    if values:
        lo, hi = min(values), max(values)
        axis.plot([lo, hi], [lo, hi], color="#333333", linewidth=0.9)
    axis.set_xlabel("Nominal TBPTT(1) final return")
    axis.set_ylabel("Nominal TBPTT(5) final return")
    axis.set_title("Archive labels: unroll is not gradient horizon")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "archive_nominal_tbptt_scatter.png", dpi=180)
    plt.close(fig)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = collect(args.logs)
    selected, duplicate_report = deduplicate(raw)
    summary = aggregate(selected)
    _write_csv(args.output / "archive_summary.csv", summary)
    _write_csv(
        args.output / "archive_selected_runs.csv",
        selected,
        [
            "method", "task", "cohort", "seed", "final_return", "final_success",
            "episodes", "max_step", "finite", "num_envs", "tbptt_steps", "path",
        ],
    )
    plot_learning_curves(selected, args.output)
    plot_final_comparison(summary, args.output)
    pairs = plot_nominal_horizon(selected, args.output)

    payload = {
        "archive": str(args.logs),
        "raw_monitor_records": len(raw),
        "selected_records": len(selected),
        "unique_monitor_hashes": len({row["sha256"] for row in raw}),
        "non_finite_selected": sum(not row["finite"] for row in selected),
        "num_envs_values": sorted({row["num_envs"] for row in selected}),
        "duplicates": duplicate_report,
        "nominal_tbptt_pairs": len(pairs),
        "nominal_tbptt_mean_absolute_difference": float(
            np.mean([abs(x - y) for _, x, y in pairs])
        ) if pairs else math.nan,
        "summary": summary,
        "methodological_warning": (
            "tbptt_steps is passed to flax scan unroll, while StreamAC detaches the previous carry "
            "before every one-step loss. Archive TBPTT(5) labels are nominal, not a five-step gradient horizon."
        ),
    }
    (args.output / "archive_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True)
    )
    print(args.output / "archive_audit.json")


if __name__ == "__main__":
    main()
