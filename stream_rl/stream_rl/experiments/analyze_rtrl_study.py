"""Aggregate corrected RTRL-approximation runs and archive baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "stream_rtu1_h64_linear_tbptt5_true_tuned": "RTU true TBPTT(5)",
    "stream_rtu1_h64_linear_tbptt5_true_matched": "RTU true TBPTT(5), matched lambda",
    "stream_lstm1_linear_tuned": "LSTM one-step",
    "stream_delta1_h64_k16_linear_tuned": "Delta-rule one-step",
    "eprop_gru_sym_linear_tuned": "GRU approximate e-prop",
    "eprop_lstm_sym_linear_tuned": "LSTM approximate e-prop",
}

ARCHIVE_KEEP = {
    "PPO-GRU": "PPO-GRU",
    "RTU exact RTRL": "RTU exact RTRL",
    "RTU one-step": "RTU one-step",
    "GRU one-step": "GRU one-step",
}


def _variant(path: Path) -> str:
    directory = path.parent.name
    return directory.split("popgym__", 1)[-1]


def _elapsed_seconds(path: Path) -> float:
    stat = path.stat()
    created = getattr(stat, "st_birthtime", stat.st_ctime)
    return max(float(stat.st_mtime - created), 0.0)


def collect_new(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    curves: list[pd.DataFrame] = []
    for root in roots:
        for path in sorted(root.glob("**/monitor_seed_*.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            variant = _variant(path)
            label = LABELS.get(variant, variant)
            seed = int(path.stem.rsplit("_", 1)[-1])
            return_col = "info.returned_episode_returns"
            finite = np.isfinite(frame[return_col].to_numpy()).all()
            tail = frame[return_col].tail(500)
            elapsed = _elapsed_seconds(path)
            max_step = int(frame["total_steps"].max())
            summaries.append(
                {
                    "method": label,
                    "variant": variant,
                    "seed": seed,
                    "episodes": int(len(frame)),
                    "max_step": max_step,
                    "final_return": float(tail.mean()),
                    "finite": bool(finite),
                    "observed_seconds": elapsed,
                    "observed_steps_per_second": (
                        max_step / elapsed if elapsed > 0 else np.nan
                    ),
                    "path": str(path),
                }
            )
            curve = frame[["total_steps", return_col]].copy()
            curve["return_smooth"] = curve[return_col].rolling(
                250, min_periods=50
            ).mean()
            curve["method"] = label
            curve["seed"] = seed
            curve["run_path"] = str(path)
            curves.append(
                curve[["total_steps", "return_smooth", "method", "seed", "run_path"]]
            )
    summary = pd.DataFrame(summaries)
    curve_frame = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if not summary.empty:
        summary = (
            summary.sort_values(["method", "seed", "max_step", "episodes"])
            .drop_duplicates(["method", "seed"], keep="last")
            .reset_index(drop=True)
        )
        if not curve_frame.empty:
            curve_frame = curve_frame[curve_frame["run_path"].isin(summary["path"])]
    return summary, curve_frame


def archive_rows(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[
        (data["task"] == "RepeatPreviousEasy")
        & (data["cohort"] == "tuned")
        & data["method"].isin(ARCHIVE_KEEP)
    ].copy()
    return data


def plot_learning_curves(curves: pd.DataFrame, output: Path) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.get_cmap("tab10")
    for idx, (method, group) in enumerate(curves.groupby("method", sort=False)):
        grid = np.linspace(0, group["total_steps"].max(), 180)
        lines = []
        for _, seed_data in group.groupby("seed"):
            clean = seed_data.dropna(subset=["return_smooth"])
            if len(clean) < 2:
                continue
            lines.append(
                np.interp(
                    grid,
                    clean["total_steps"].to_numpy(),
                    clean["return_smooth"].to_numpy(),
                )
            )
        if not lines:
            continue
        values = np.asarray(lines)
        mean = values.mean(axis=0)
        ax.plot(grid, mean, lw=2.2, label=method, color=colors(idx))
        if len(values) > 1:
            ax.fill_between(
                grid,
                mean - values.std(axis=0, ddof=1),
                mean + values.std(axis=0, ddof=1),
                color=colors(idx),
                alpha=0.18,
            )
    ax.axhline(-0.5, color="#777777", ls="--", lw=1, label="random-policy region")
    ax.set(xlabel="Environment steps", ylabel="Return (rolling 250 episodes)")
    ax.set_title("Corrected approximations on POPGym RepeatPreviousEasy")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_final(new_summary: pd.DataFrame, archive: pd.DataFrame, output: Path) -> None:
    rows = []
    if not archive.empty:
        for row in archive.itertuples():
            rows.append(
                {
                    "method": row.method,
                    "mean": row.return_mean,
                    "sd": row.return_sd,
                    "n": row.n_seeds,
                    "source": "archive",
                }
            )
    if not new_summary.empty:
        for method, group in new_summary.groupby("method", sort=False):
            rows.append(
                {
                    "method": method,
                    "mean": group["final_return"].mean(),
                    "sd": group["final_return"].std(ddof=1),
                    "n": group["seed"].nunique(),
                    "source": "corrected",
                }
            )
    plot_data = pd.DataFrame(rows).sort_values("mean", ascending=True)
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(plot_data))
    colors = ["#6688aa" if source == "archive" else "#dd8844" for source in plot_data["source"]]
    ax.barh(y, plot_data["mean"], xerr=plot_data["sd"].fillna(0), color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(y, [f"{m} (n={n})" for m, n in zip(plot_data["method"], plot_data["n"])])
    ax.axvline(-0.5, color="#777777", ls="--", lw=1)
    ax.set(xlabel="Final mean return (last 500 episodes)")
    ax.set_title("Comparable-budget comparison (~170k steps)")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--archive-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    new_summary, curves = collect_new(args.roots)
    archive = archive_rows(args.archive_summary)
    new_summary.to_csv(args.output / "corrected_runs.csv", index=False)
    if not new_summary.empty:
        aggregate = (
            new_summary.groupby("method", as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                return_mean=("final_return", "mean"),
                return_sd=("final_return", "std"),
                max_step_min=("max_step", "min"),
                throughput_median=("observed_steps_per_second", "median"),
                all_finite=("finite", "all"),
            )
        )
    else:
        aggregate = pd.DataFrame()
    aggregate.to_csv(args.output / "corrected_summary.csv", index=False)
    archive.to_csv(args.output / "archive_equal_budget.csv", index=False)
    plot_learning_curves(curves, args.output / "corrected_learning_curves.png")
    plot_final(new_summary, archive, args.output / "equal_budget_final.png")
    audit = {
        "new_runs": int(len(new_summary)),
        "new_methods": sorted(new_summary["method"].unique().tolist()) if not new_summary.empty else [],
        "new_seeds": sorted(map(int, new_summary["seed"].unique())) if not new_summary.empty else [],
        "all_finite": bool(new_summary["finite"].all()) if not new_summary.empty else None,
        "complete_nominal_170k": int((new_summary["max_step"] >= 167_000).sum()) if not new_summary.empty else 0,
    }
    (args.output / "corrected_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
