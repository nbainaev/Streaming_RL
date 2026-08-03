import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func


LOG_DIR = Path("logs/ppo/CartPole-v1_1")
OUT_FILE = Path("logs/ppo_cartpole_train.png")
WINDOW = 50

# Проверяем фактический путь: RL Zoo создаёт CartPole-v1_1, _2, ... .
if not LOG_DIR.exists():
    candidates = sorted(Path("logs/ppo").glob("CartPole-v1_*"))
    if not candidates:
        raise FileNotFoundError(
            "Не найдены папки logs/ppo/CartPole-v1_*. "
            "Выполни: find logs -name '*.monitor.csv'"
        )
    LOG_DIR = candidates[-1]

df = load_results(str(LOG_DIR))
timesteps, returns = ts2xy(df, "timesteps")

plt.figure(figsize=(9, 5))
plt.scatter(timesteps, returns, s=8, alpha=0.25, label="Episode return")

if len(returns) >= WINDOW:
    smooth_x, smooth_y = window_func(timesteps, returns, WINDOW, np.mean)
    plt.plot(smooth_x, smooth_y, color="crimson", linewidth=2,
             label=f"Moving average ({WINDOW} episodes)")

plt.xlabel("Environment steps")
plt.ylabel("Episode return")
plt.title("PPO — CartPole-v1")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=180)
plt.close()

print(f"Read logs from: {LOG_DIR}")
print(f"Saved figure:   {OUT_FILE}")