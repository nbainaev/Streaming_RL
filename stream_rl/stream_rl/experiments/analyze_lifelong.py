"""Summarize CT-graph monitor CSVs by reward phase and training seed."""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from collections import OrderedDict
from pathlib import Path


def mean(values):
    return statistics.mean(values) if values else float("nan")


def summarize_seed(path: Path, early_episodes: int, late_episodes: int, skip_phases: int):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty monitor file: {path}")
    required = {"info.returned_episode_returns", "info.reward_phase"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing CT-graph columns: {sorted(missing)}")

    returns = [float(row["info.returned_episode_returns"]) for row in rows]
    successes = [float(row.get("info.success", 0.0)) for row in rows]
    phases = OrderedDict()
    for row in rows:
        phase = int(float(row["info.reward_phase"]))
        phases.setdefault(phase, []).append(float(row["info.returned_episode_returns"]))

    complete_phases = list(phases)[skip_phases:-1]
    early, late = [], []
    for phase in complete_phases:
        phase_returns = phases[phase]
        if len(phase_returns) < early_episodes + late_episodes:
            continue
        early.extend(phase_returns[:early_episodes])
        late.extend(phase_returns[-late_episodes:])

    tail_size = min(500, len(rows))
    early_mean = mean(early)
    late_mean = mean(late)
    return {
        "episodes": len(rows),
        "last_500_return": mean(returns[-tail_size:]),
        "last_500_success": mean(successes[-tail_size:]),
        "phase_early_return": early_mean,
        "phase_late_return": late_mean,
        "phase_adaptation_gain": late_mean - early_mean,
        "complete_phases": len(complete_phases),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns", nargs="+",
        help="Monitor paths or glob patterns, e.g. logs/pilot_validation/*/*/monitor_seed_*.csv",
    )
    parser.add_argument("--early-episodes", type=int, default=10)
    parser.add_argument("--late-episodes", type=int, default=10)
    parser.add_argument("--skip-phases", type=int, default=2)
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no monitor CSV files matched")

    grouped = OrderedDict()
    for path in paths:
        model = path.parent.parent.name
        grouped.setdefault(model, []).append(
            summarize_seed(path, args.early_episodes, args.late_episodes, args.skip_phases)
        )

    columns = [
        "last_500_return", "last_500_success", "phase_early_return",
        "phase_late_return", "phase_adaptation_gain",
    ]
    print("| model | seeds | " + " | ".join(columns) + " |")
    print("|---|---:|" + "---:|" * len(columns))
    for model, seed_rows in grouped.items():
        values = [mean([row[column] for row in seed_rows]) for column in columns]
        print(
            f"| {model} | {len(seed_rows)} | "
            + " | ".join(f"{value:.4f}" for value in values)
            + " |"
        )


if __name__ == "__main__":
    main()
