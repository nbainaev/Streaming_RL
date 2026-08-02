import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func


WINDOW = 50

RUNS = {
    "PPO": Path("logs/ppo/Pendulum-v1_1"),
    "StreamAC": Path("logs/stream_ac/Pendulum-v1_1"),
}


def load_curve(log_dir: Path):
    df = load_results(str(log_dir))
    timesteps, returns = ts2xy(df, "timesteps")

    if len(returns) < WINDOW:
        raise ValueError(
            f"{log_dir}: only {len(returns)} completed episodes; "
            f"need at least {WINDOW} for this smoothing window."
        )

    smooth_steps, smooth_returns = window_func(
        timesteps, returns, WINDOW, np.mean
    )
    return smooth_steps, smooth_returns


plt.figure(figsize=(9, 5))

for label, log_dir in RUNS.items():
    if not log_dir.exists():
        raise FileNotFoundError(f"Directory not found: {log_dir}")

    x, y = load_curve(log_dir)
    plt.plot(x, y, linewidth=2, label=label)

plt.axhline(-200, color="gray", linestyle="--", linewidth=1, label="Pendulum solved threshold")
plt.xlabel("Environment steps")
plt.ylabel(f"Mean episodic return ({WINDOW}-episode moving average)")
plt.title("Pendulum-v1: PPO vs StreamAC")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("logs/pendulum_ppo_vs_stream_ac.png", dpi=180)
plt.close()

print("Saved: logs/pendulum_ppo_vs_stream_ac.png")