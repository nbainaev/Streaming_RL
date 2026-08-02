from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml
from scipy.stats import t as student_t

from stream_rl.visual_utils.base import LogLoader


@dataclass
class SweepPoint:
    run_root: str
    seed: str
    x_value: Any
    metric_value: float
    split_value: Optional[Any] = None


@dataclass
class SweepSummary:
    x_value: Any
    mean: float
    lo: float
    hi: float
    n: int
    split_value: Optional[Any] = None


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_by_path(tree: dict[str, Any], path: str) -> Any:
    cur: Any = tree
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Path '{path}' not found at '{part}'")
    return _coerce_scalar(cur)


class RunConfigLoader:
    """Reads a run's config (env/agent/runner) for grid-parameter lookups.

    `structure.json` is what `ExperimentRunner.write_structure()` actually
    writes into every run directory (alongside monitor_seed_*.csv), so it's
    tried first; the other candidates are for configs produced by other
    tooling that might land in the same directory layout.
    """
    CANDIDATES = (
        'structure.json',
        'structure.yaml',
        'structure.yml',
        'manifest.json',
        'config.yaml',
        'config.yml',
        'config.json',
    )

    def __init__(self, run_root: Path):
        self.run_root = Path(run_root)

    def load(self) -> dict[str, Any]:
        for name in self.CANDIDATES:
            path = self.run_root / name
            if path.exists():
                return self._load_file(path)
        for path in sorted(self.run_root.glob('*.y*ml')) + sorted(self.run_root.glob('*.json')):
            try:
                data = self._load_file(path)
            except Exception:
                continue
            if isinstance(data, dict) and any(k in data for k in ('env', 'agent', 'runner', 'env_config', 'agent_config', 'runner_config')):
                return data
        raise FileNotFoundError(f'No structure/config file found under {self.run_root}')

    def _load_file(self, path: Path) -> dict[str, Any]:
        if path.suffix == '.json':
            return json.loads(path.read_text())
        return yaml.safe_load(path.read_text())


def normalize_structure_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Maps whichever config-key spelling a given structure/manifest file
    uses (env/env_config/env_cfg, ...) onto the canonical env/agent/runner
    keys used by dotted-path lookups (e.g. 'env.kwargs.corridor_length')."""
    if any(k in raw for k in ('env', 'agent', 'runner')):
        return raw
    out = {}
    for canonical, aliases in (
        ('env', ('env_config', 'env_cfg')),
        ('agent', ('agent_config', 'agent_cfg')),
        ('runner', ('runner_config', 'runner_cfg')),
    ):
        for alias in aliases:
            if alias in raw:
                out[canonical] = raw[alias]
                break
    return out or raw


class SeedMetricLoader:
    """Per-seed scalar metric loader for our own monitor/metrics CSV
    schema (see visual_utils.base.LogLoader) — replaces an earlier version
    built on stable_baselines3.common.monitor.load_results, which expects
    SB3's own Monitor wrapper format and can't read this project's JAX
    runner output at all."""

    def __init__(self, run_root: Path, source: str = 'monitor', y_col: Optional[str] = None, window_steps: float = 0):
        self.run_root = Path(run_root)
        self.source = source
        self.y_col = y_col
        self.window_steps = window_steps

    def iter_seed_metrics(self, metric: str, last_k: int = 100) -> Iterable[tuple[str, float]]:
        loader = LogLoader(
            self.run_root, source=self.source, y_col=self.y_col,
            smoothing='window' if self.window_steps > 0 else 'none', window_steps=self.window_steps,
        )
        reduction = {
            'final_episode_return': 'final',
            'final_smoothed_return': 'final',
            'final_mean_last_k_episodes': 'mean_last_k',
            'best_episode_return': 'max',
        }.get(metric)
        if reduction is None:
            raise ValueError(f'Unknown metric: {metric}')
        values = loader.load_seed_summary(reduction=reduction, last_k=last_k)
        seed_files = loader._seed_files()
        for path, value in zip(seed_files, values):
            yield path.stem, value


class SweepAggregator:
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def summarize(self, points: list[SweepPoint]) -> list[SweepSummary]:
        grouped: dict[tuple[Any, Any], list[float]] = {}
        for p in points:
            key = (p.x_value, p.split_value)
            grouped.setdefault(key, []).append(float(p.metric_value))

        out: list[SweepSummary] = []
        for (x_value, split_value), values in grouped.items():
            arr = np.asarray(values, dtype=float)
            mean = float(arr.mean())
            n = int(arr.size)
            if n > 1:
                std = arr.std(ddof=1)
                margin = float(student_t.ppf((1 + self.confidence) / 2, n - 1) * std / np.sqrt(n))
            else:
                margin = 0.0
            out.append(SweepSummary(
                x_value=_coerce_scalar(x_value),
                split_value=_coerce_scalar(split_value),
                mean=mean,
                lo=mean - margin,
                hi=mean + margin,
                n=n,
            ))

        def sort_key(s: SweepSummary):
            xv = s.x_value
            return (0, float(xv), '' if s.split_value is None else str(s.split_value)) if isinstance(xv, (int, float)) else (1, str(xv), '' if s.split_value is None else str(s.split_value))

        return sorted(out, key=sort_key)


def load_sweep_points(
    run_roots: list[str],
    group_by: str,
    metric: str,
    last_k: int = 100,
    split_by: Optional[str] = None,
    window_steps: float = 0,
    source: str = 'monitor',
    y_col: Optional[str] = None,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for run_root in run_roots:
        rr = Path(run_root)
        cfg = normalize_structure_config(RunConfigLoader(rr).load())
        x_value = get_by_path(cfg, group_by)
        split_value = get_by_path(cfg, split_by) if split_by else None
        metric_loader = SeedMetricLoader(rr, source=source, y_col=y_col, window_steps=window_steps)
        for seed, metric_value in metric_loader.iter_seed_metrics(metric=metric, last_k=last_k):
            points.append(SweepPoint(
                run_root=str(rr),
                seed=seed,
                x_value=x_value,
                metric_value=metric_value,
                split_value=split_value,
            ))
    return points
