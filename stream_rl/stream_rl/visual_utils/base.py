from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t

# Bookkeeping columns present in every monitor_seed_*.csv / metrics_seed_*.csv
# that are never plottable metrics themselves.
_NON_METRIC_COLUMNS = {'step', 'seed', 'episode_idx', 'total_steps'}


@dataclass
class PlotStyle:
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: str = '-'
    linewidth: float = 2.0
    alpha_ci: float = 0.2
    marker: Optional[str] = None
    zorder: int = 2


@dataclass
class AxisStyle:
    xlabel: str = 'Environment steps'
    ylabel: str = 'Episode return'
    title: str = ''
    xscale: str = 'linear'
    yscale: str = 'linear'
    grid_alpha: float = 0.25
    legend_loc: str = 'best'
    figsize: Tuple[float, float] = (9, 5)


def smooth_by_steps(x: np.ndarray, y: np.ndarray, window_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Trailing moving average over a window measured in x-units (env steps)."""
    if len(x) == 0:
        return x, y
    left_idx = np.searchsorted(x, x - window_steps, side='left')
    y_smooth = np.empty_like(y, dtype=float)
    cumsum = np.concatenate([[0.0], np.cumsum(y)])
    for i in range(len(x)):
        lo = left_idx[i]
        y_smooth[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return x, y_smooth


def smooth_gaussian(x: np.ndarray, y: np.ndarray, sigma_steps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-kernel smoothing with sigma expressed in x-units (env steps).

    Points aren't evenly spaced in x (episodes complete at irregular step
    counts), so sigma_steps is converted to an index-space sigma using the
    median spacing — a reasonable approximation as long as spacing doesn't
    vary wildly within one run, which holds here (episode lengths are
    usually fairly stable within a single training run).
    """
    if sigma_steps <= 0 or len(x) < 2:
        return x, y
    dx = float(np.median(np.diff(x)))
    if dx <= 0:
        return x, y
    sigma_idx = max(sigma_steps / dx, 1e-6)
    y_smooth = gaussian_filter1d(y.astype(float), sigma=sigma_idx, mode='nearest')
    return x, y_smooth


def apply_smoothing(x: np.ndarray, y: np.ndarray, mode: str, window_steps: float, sigma_steps: float) -> Tuple[np.ndarray, np.ndarray]:
    if mode == 'window' and window_steps > 0:
        return smooth_by_steps(x, y, window_steps)
    if mode == 'gaussian' and sigma_steps > 0:
        return smooth_gaussian(x, y, sigma_steps)
    return x, y


class LogLoader:
    """Loads one metric column, from one seed-log source, for one run.

    `source` selects which file family to read:
      - 'monitor': monitor_seed_*.csv — one row per completed episode
        (returns, lengths, success when the env emits it).
      - 'metrics': metrics_seed_*.csv — one row per log flush during
        training (actor/entropy, critic/value, critic/td_error, ...).
    """

    _RECOVERED_GLOB = {'monitor': 'monitor_seed_*_recovered.csv', 'metrics': 'metrics_seed_*_recovered.csv'}
    _RAW_GLOB = {'monitor': 'monitor_seed_*.csv', 'metrics': 'metrics_seed_*.csv'}
    # Back-compat aliases for the pre-rework episodic-return column names.
    _Y_COL_ALIASES = (
        'info.returned_episode_returns',
        'info.episodic_return',
        'returned_episode_returns',
    )

    def __init__(self, run_root: Path, source: str = 'monitor', y_col: Optional[str] = None,
                 smoothing: str = 'window', window_steps: float = 5000, sigma_steps: float = 2000,
                 x_col: str = 'step'):
        if source not in ('monitor', 'metrics'):
            raise ValueError(f"source must be 'monitor' or 'metrics', got {source!r}")
        self.run_root = Path(run_root)
        self.source = source
        self.x_col = x_col
        self.y_col = y_col
        self.smoothing = smoothing
        self.window_steps = window_steps
        self.sigma_steps = sigma_steps

    def _seed_files(self) -> List[Path]:
        recovered = sorted(self.run_root.glob(self._RECOVERED_GLOB[self.source]))
        if recovered:
            return recovered
        files = sorted(self.run_root.glob(self._RAW_GLOB[self.source]))
        if not files:
            raise FileNotFoundError(
                f"No {self.source}_seed_*.csv files found under {self.run_root}."
            )
        return files

    @classmethod
    def list_columns(cls, run_root: Path) -> List[Tuple[str, str]]:
        """Returns [(source, column), ...] for every plottable column found
        under run_root, across both monitor_seed_*.csv and metrics_seed_*.csv
        (using the first seed file of each kind as a schema sample)."""
        run_root = Path(run_root)
        out: List[Tuple[str, str]] = []
        for source in ('monitor', 'metrics'):
            files = sorted(run_root.glob(cls._RAW_GLOB[source])) or sorted(run_root.glob(cls._RECOVERED_GLOB[source]))
            if not files:
                continue
            try:
                columns = pd.read_csv(files[0], nrows=1).columns
            except Exception:
                continue
            for col in columns:
                if col in _NON_METRIC_COLUMNS:
                    continue
                out.append((source, col))
        return out

    def _resolve_y_col(self, columns) -> str:
        if self.y_col is not None:
            if self.y_col in columns:
                return self.y_col
            raise ValueError(f'Column {self.y_col!r} not found. Available: {list(columns)}')
        for alias in self._Y_COL_ALIASES:
            if alias in columns:
                return alias
        candidates = [c for c in columns if c not in _NON_METRIC_COLUMNS]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f'Could not auto-resolve a y column (none requested). Available: {list(columns)}'
        )

    def load_seed_curves(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        curves = []
        for path in self._seed_files():
            df = pd.read_csv(path)
            if self.x_col not in df.columns:
                raise ValueError(f'{path} is missing required x column: {self.x_col}')
            y_col = self._resolve_y_col(df.columns)
            sub = df[[self.x_col, y_col]].dropna()
            if len(sub) == 0:
                continue
            x = sub[self.x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)
            order = np.argsort(x)
            x, y = x[order], y[order]
            x, y = apply_smoothing(x, y, self.smoothing, self.window_steps, self.sigma_steps)
            curves.append((x, y))
        if not curves:
            raise ValueError(f'No valid curves found under {self.run_root}')
        return curves

    def load_seed_summary(self, reduction: str = 'final', last_k: int = 20) -> List[float]:
        """One scalar per seed — for box plots / heatmaps. Uses the
        UNSMOOTHED series for 'final'/'max' (smoothing would bias a
        single-point summary); 'mean_last_k' averages the last k raw rows,
        which already acts as its own local smoothing."""
        values = []
        for path in self._seed_files():
            df = pd.read_csv(path)
            y_col = self._resolve_y_col(df.columns)
            sub = df[[self.x_col, y_col]].dropna().sort_values(self.x_col)
            if len(sub) == 0:
                continue
            y = sub[y_col].to_numpy(dtype=float)
            if reduction == 'final':
                values.append(float(y[-1]))
            elif reduction == 'max':
                values.append(float(np.max(y)))
            elif reduction == 'mean_last_k':
                k = min(last_k, len(y))
                values.append(float(np.mean(y[-k:])))
            elif reduction == 'mean':
                values.append(float(np.mean(y)))
            else:
                raise ValueError(f'Unknown reduction: {reduction}')
        return values


class CurveAggregator:
    def __init__(self, curves: List[Tuple[np.ndarray, np.ndarray]], confidence: float = 0.95, n_points: int = 200):
        if not curves:
            raise ValueError('CurveAggregator requires at least one curve.')
        self.curves = curves
        self.confidence = confidence
        self.n_points = n_points

    def aggregate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_min = max(c[0][0] for c in self.curves)
        x_max = min(c[0][-1] for c in self.curves)
        if x_min >= x_max:
            raise ValueError('Seed curves do not overlap in x-range; cannot build a common grid.')
        grid = np.linspace(x_min, x_max, self.n_points)
        interpolated = np.stack([np.interp(grid, x, y) for x, y in self.curves])
        mean = interpolated.mean(axis=0)
        n_seeds = interpolated.shape[0]
        if n_seeds > 1:
            std = interpolated.std(axis=0, ddof=1)
            margin = student_t.ppf((1 + self.confidence) / 2, n_seeds - 1) * std / np.sqrt(n_seeds)
        else:
            margin = np.zeros_like(mean)
        return grid, mean, mean - margin, mean + margin
