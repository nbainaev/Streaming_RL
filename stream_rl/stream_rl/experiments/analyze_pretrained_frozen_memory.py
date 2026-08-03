"""Summarize frozen-memory T-maze experiments from monitor CSV files."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


RUN_ORDER = (
    "ppo_no_memory",
    "ppo_pretrained_ssm",
    "ppo_pretrained_ssm_aux",
    "ppo_gru",
    "stream_no_memory",
    "stream_pretrained_ssm",
    "stream_pretrained_ssm_aux",
    "stream_gru",
)


def load_monitor(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "step": float(raw["total_steps"]),
                    "success": float(raw.get("info.success", raw["info.returned_episode_returns"])),
                }
            )
    return rows


def first_threshold_step(rows: list[dict[str, float]], window: int = 100, threshold: float = 0.9):
    running = 0.0
    for index, row in enumerate(rows):
        running += row["success"]
        if index >= window:
            running -= rows[index - window]["success"]
        if index + 1 >= window and running / window >= threshold:
            return int(row["step"])
    return None


def mean_std(values: list[float]) -> str:
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}"


def summarize(root: Path) -> None:
    print("| run | seeds | last 100 success | last 500 success | steps to rolling-100 >= 0.90 |")
    print("|---|---:|---:|---:|---:|")
    for run_name in RUN_ORDER:
        run_dir = root / run_name
        paths = sorted(run_dir.glob("monitor_seed_*.csv"))
        if not paths:
            continue
        last_100 = []
        last_500 = []
        threshold_steps = []
        for path in paths:
            rows = load_monitor(path)
            last_100.append(statistics.mean(row["success"] for row in rows[-100:]))
            last_500.append(statistics.mean(row["success"] for row in rows[-500:]))
            step = first_threshold_step(rows)
            if step is not None:
                threshold_steps.append(float(step))
        threshold = mean_std(threshold_steps) if len(threshold_steps) == len(paths) else f"{len(threshold_steps)}/{len(paths)} seeds"
        print(
            f"| {run_name} | {len(paths)} | {mean_std(last_100)} | "
            f"{mean_std(last_500)} | {threshold} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "logs/pretrained_frozen_memory_forced_tmaze/"
            "pretrained_frozen_memory_forced_tmaze"
        ),
    )
    args = parser.parse_args()
    summarize(args.root)


if __name__ == "__main__":
    main()
