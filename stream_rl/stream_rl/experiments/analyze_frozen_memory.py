"""Summarize the passive T-maze frozen-memory ablation from monitor CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = {
    "ppo_frozen_ssm_zero_preserving": "PPO + frozen SSM (readout only)",
    "ppo_gru_control": "PPO + learned GRU",
    "stream_ac_frozen_ssm": "StreamAC + frozen SSM (readout only)",
    "stream_ac_gru_control": "StreamAC + learned GRU",
}


def mean(values):
    return float(statistics.mean(values))


def summarize(run_root: Path, max_steps: int, tail_episodes: int):
    records = []
    for run_id, label in METHODS.items():
        run_dir = run_root / run_id
        for path in sorted(run_dir.glob("monitor_seed_*.csv")):
            seed = int(path.stem.rsplit("_", 1)[-1])
            with path.open(newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if int(row["total_steps"]) <= max_steps
                ]
            tail = rows[-tail_episodes:]
            returns = [float(row["info.returned_episode_returns"]) for row in tail]
            successes = [float(row["info.success"]) for row in tail]
            records.append(
                {
                    "run_id": run_id,
                    "method": label,
                    "seed": seed,
                    "episodes": len(rows),
                    "tail_episodes": len(tail),
                    "mean_return": mean(returns),
                    "success_rate": mean(successes),
                }
            )

    aggregates = []
    for run_id, label in METHODS.items():
        subset = [row for row in records if row["run_id"] == run_id]
        success = [row["success_rate"] for row in subset]
        returns = [row["mean_return"] for row in subset]
        aggregates.append(
            {
                "run_id": run_id,
                "method": label,
                "seeds": len(subset),
                "mean_return": mean(returns),
                "success_rate": mean(success),
                "success_std": float(statistics.stdev(success)) if len(success) > 1 else 0.0,
            }
        )
    return records, aggregates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("logs/frozen_memory_passive_tmaze/frozen_memory_passive_tmaze"),
    )
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--tail-episodes", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logs/frozen_memory_passive_tmaze/summary")
    )
    args = parser.parse_args()

    records, aggregates = summarize(args.run_root, args.max_steps, args.tail_episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregates[0].keys())
        writer.writeheader()
        writer.writerows(aggregates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "max_steps": args.max_steps,
                "tail_episodes": args.tail_episodes,
                "per_seed": records,
                "aggregate": aggregates,
                "interpretation": (
                    "A success rate near 0.5 is the open-loop fixed-branch baseline; "
                    "evidence of cue memory requires a reproducible rate materially above 0.5."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    for row in aggregates:
        print(
            f"{row['method']}: return={row['mean_return']:.4f}, "
            f"success={row['success_rate']:.4f} +/- {row['success_std']:.4f}"
        )


if __name__ == "__main__":
    main()
