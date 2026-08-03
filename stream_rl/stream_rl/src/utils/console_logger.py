"""Simple line-based console logger. Prints scalar training metrics plus
mean/last episodic return computed from completed episodes in the chunk
(info["returned_episode"] marks completion, matching
RecordEpisodeStatistics's contract) — without this, it's impossible to see
whether the agent is actually learning from actor/critic losses alone."""
import numpy as np
import jax.numpy as jnp


class ConsoleLogger:
    def __init__(self, precision: int = 4, **kwargs):
        self.precision = precision

    def _episodic_summary(self, info: dict) -> str | None:
        returned_episode = info.get("returned_episode")
        returns = info.get("returned_episode_returns")
        if returned_episode is None or returns is None:
            return None

        done_mask = np.asarray(returned_episode).reshape(-1).astype(bool)
        returns = np.asarray(returns).reshape(-1)

        finished_returns = returns[done_mask]
        if finished_returns.size == 0:
            return None

        return (
            f"episodic_return(mean={finished_returns.mean():.{self.precision}f}, "
            f"last={finished_returns[-1]:.{self.precision}f}, "
            f"n={finished_returns.size})"
        )

    def log(self, data: dict, step: int, **kwargs) -> None:
        parts = [f"step={step}"]

        info = data.get("info")
        if info:
            summary = self._episodic_summary(info)
            if summary is not None:
                parts.append(summary)

        for key, value in sorted(data.items()):
            if key in ("info", "intermediates"):
                continue
            try:
                mean = float(jnp.mean(value))
            except (TypeError, ValueError):
                continue
            parts.append(f"{key}={mean:.{self.precision}f}")

        print(" | ".join(parts))

    def finish(self) -> None:
        pass