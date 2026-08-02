import argparse
import copy
import hashlib
import itertools
import json
import secrets
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def resolve_path(base_file: str | Path, target: str | Path) -> Path:
    base_file = Path(base_file).resolve()
    target = Path(target)
    if target.is_absolute():
        return target
    return (base_file.parent / target).resolve()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def get_by_dotted_path(data: dict, path: str):
    cur = data
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f'Path does not exist: {path}')
        cur = cur[part]
    return cur


def set_by_dotted_path(data: dict, path: str, value):
    cur = data
    parts = path.split('.')
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f'Path does not exist: {path}')
        cur = cur[part]
    leaf = parts[-1]
    if not isinstance(cur, dict) or leaf not in cur:
        raise KeyError(f'Path does not exist: {path}')
    cur[leaf] = value


def validate_prefixed_path(path: str) -> None:
    if not (path.startswith('runner.') or path.startswith('agent.') or path.startswith('env.')):
        raise ValueError(
            f"Override/grid path '{path}' must start with one of: 'runner.', 'agent.', 'env.'"
        )


def strip_prefix(path: str) -> tuple[str, str]:
    root, rest = path.split('.', 1)
    return root, rest


def is_compatible_type(old_value, new_value) -> bool:
    if old_value is None:
        return True
    if isinstance(old_value, bool):
        return isinstance(new_value, bool)
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return isinstance(new_value, int) and not isinstance(new_value, bool)
    if isinstance(old_value, float):
        return isinstance(new_value, (int, float)) and not isinstance(new_value, bool)
    if isinstance(old_value, str):
        return isinstance(new_value, str)
    if isinstance(old_value, list):
        return isinstance(new_value, list)
    if isinstance(old_value, dict):
        return isinstance(new_value, dict)
    return isinstance(new_value, type(old_value))


def apply_overrides(merged: dict, overrides: dict | None) -> None:
    if not overrides:
        return
    for full_path, value in overrides.items():
        validate_prefixed_path(full_path)
        root, inner_path = strip_prefix(full_path)
        old_value = get_by_dotted_path(merged[root], inner_path)
        if not is_compatible_type(old_value, value):
            raise TypeError(
                f'Type mismatch for {full_path}: expected {type(old_value).__name__}, '
                f'got {type(value).__name__} (old={old_value!r}, new={value!r})'
            )
        set_by_dotted_path(merged[root], inner_path, value)


def normalize_grid(grid: dict | None) -> dict[str, list]:
    if not grid:
        return {}
    normalized = {}
    for full_path, values in grid.items():
        validate_prefixed_path(full_path)
        if not isinstance(values, list):
            raise ValueError(f"Grid entry '{full_path}' must be a list, got {type(values).__name__}")
        if len(values) == 0:
            raise ValueError(f"Grid entry '{full_path}' must be non-empty")
        normalized[full_path] = values
    return normalized


def expand_grid(grid: dict[str, list]) -> list[dict[str, object]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def sanitize_token(value) -> str:
    s = str(value)
    keep = []
    for ch in s:
        keep.append(ch if ch.isalnum() or ch in ('-', '_', '.') else '-')
    return ''.join(keep).strip('-')[:80] or 'value'


def shorten_key(path: str) -> str:
    for prefix in ('runner.', 'agent.', 'env.'):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    return path.replace('.', '_')


def stable_hash(payload: dict, n: int = 10) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:n]


def build_variant_name(experiment_name: str, grid_values: dict[str, object]) -> str:
    if not grid_values:
        return experiment_name
    parts = [experiment_name]
    for key in sorted(grid_values):
        parts.append(f"{shorten_key(key)}={sanitize_token(grid_values[key])}")
    raw = '__'.join(parts)
    if len(raw) <= 180:
        return raw
    return f"{experiment_name}__{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def load_merged_config(base_runner_config_path: Path) -> dict:
    runner_cfg = load_yaml(base_runner_config_path)
    ensure('agent_config' in runner_cfg, f"Runner config {base_runner_config_path} must contain 'agent_config'")
    ensure('env_config' in runner_cfg, f"Runner config {base_runner_config_path} must contain 'env_config'")
    agent_config_path = resolve_path(base_runner_config_path, runner_cfg['agent_config'])
    env_config_path = resolve_path(base_runner_config_path, runner_cfg['env_config'])
    agent_cfg = load_yaml(agent_config_path)
    env_cfg = load_yaml(env_config_path)
    return {
        'runner': runner_cfg,
        'agent': agent_cfg,
        'env': env_cfg,
        '__paths__': {
            'base_runner_config': str(base_runner_config_path.resolve()),
            'agent_config': str(agent_config_path.resolve()),
            'env_config': str(env_config_path.resolve()),
        },
    }


def refresh_merged_configs(merged: dict, base_path: Path) -> None:
    """
    Если в merged['runner'] изменились пути к agent_config или env_config,
    перезагружаем соответствующие словари из новых файлов и обновляем __paths__.
    """
    runner_cfg = merged['runner']
    old_agent_path = merged['__paths__']['agent_config']
    new_agent_path_raw = runner_cfg.get('agent_config')
    if new_agent_path_raw:
        new_agent_abs = resolve_path(base_path, new_agent_path_raw)
        if str(new_agent_abs.resolve()) != old_agent_path:
            merged['agent'] = load_yaml(new_agent_abs)
            merged['__paths__']['agent_config'] = str(new_agent_abs.resolve())

    old_env_path = merged['__paths__']['env_config']
    new_env_path_raw = runner_cfg.get('env_config')
    if new_env_path_raw:
        new_env_abs = resolve_path(base_path, new_env_path_raw)
        if str(new_env_abs.resolve()) != old_env_path:
            merged['env'] = load_yaml(new_env_abs)
            merged['__paths__']['env_config'] = str(new_env_abs.resolve())


def build_run_identity(experiment: dict, runner_cfg: dict, grid_values: dict[str, object],
                       overrides: dict | None) -> tuple[str, str]:
    """Создаёт идентификатор варианта (без учёта seed) – общий для всех сидов."""
    series_name = runner_cfg.get('experiment_name', experiment['name'])
    variant_name = build_variant_name(experiment['name'], grid_values)
    payload = {
        'experiment_name': experiment['name'],
        'series_name': series_name,
        'grid_values': grid_values,
        'overrides': overrides or {},
    }
    deterministic_tag = stable_hash(payload, n=10)
    random_tag = secrets.token_hex(4)
    run_id = f"{sanitize_token(variant_name)}__{deterministic_tag}__{random_tag}"
    return variant_name, run_id


def materialize_variant(work_dir: Path, experiment: dict, merged_variant: dict,
                        variant_name: str, run_id: str,
                        grid_values: dict[str, object], overrides: dict | None) -> Path:
    """
    Создаёт папку для варианта и записывает в неё общие конфиги (без учёта seed).
    Если папка уже существует (например, при повторном вызове для того же run_id),
    ничего не перезаписывает, а просто возвращает путь к папке.
    """
    runner_cfg = copy.deepcopy(merged_variant['runner'])
    agent_cfg = copy.deepcopy(merged_variant['agent'])
    env_cfg = copy.deepcopy(merged_variant['env'])

    # Удаляем список сидов – теперь они передаются через --seed
    runner_cfg.pop('seeds', None)

    series_name = runner_cfg.get('experiment_name', experiment['name'])
    variant_dir = work_dir / sanitize_token(series_name) / run_id

    # Если папка уже существует, считаем, что конфиги уже записаны
    if variant_dir.exists():
        return variant_dir

    variant_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg['experiment_name'] = series_name
    runner_cfg['variant_name'] = variant_name
    runner_cfg['run_id'] = run_id

    agent_out = variant_dir / 'agent.yaml'
    env_out = variant_dir / 'env.yaml'
    runner_out = variant_dir / 'runner.yaml'

    save_yaml(agent_out, agent_cfg)
    save_yaml(env_out, env_cfg)

    runner_cfg['agent_config'] = str(agent_out.resolve())
    runner_cfg['env_config'] = str(env_out.resolve())
    scenario_path = runner_cfg.get('scenario_config')
    if scenario_path is not None:
        base_runner_config_path = Path(merged_variant['__paths__']['base_runner_config'])
        scenario_abs = resolve_path(base_runner_config_path, scenario_path)
        runner_cfg['scenario_config'] = str(scenario_abs.resolve())
    save_yaml(runner_out, runner_cfg)

    # Сохраняем манифест с общей информацией (без seed)
    manifest = {
        'series_name': series_name,
        'experiment_name': experiment['name'],
        'variant_name': variant_name,
        'run_id': run_id,
        'command': experiment['command'],
        'grid_values': grid_values,
        'overrides': overrides or {},
        'base_paths': merged_variant['__paths__'],
        'materialized_paths': {
            'runner_config': str(runner_out.resolve()),
            'agent_config': str(agent_out.resolve()),
            'env_config': str(env_out.resolve()),
        },
    }
    with open(variant_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return variant_dir


def build_jobs(meta_cfg: dict, meta_cfg_path: Path) -> tuple[list[dict], Path]:
    work_dir = Path(meta_cfg.get('work_dir', 'logs')).resolve()
    experiments = meta_cfg.get('experiments', [])
    ensure(isinstance(experiments, list) and experiments, 'No experiments specified')

    jobs = []
    for experiment in experiments:
        ensure('name' in experiment, 'Experiment must contain name')
        ensure('command' in experiment, f"Experiment {experiment.get('name')} missing command")
        ensure('base_runner_config' in experiment, f"Experiment {experiment.get('name')} missing base_runner_config")

        base_runner_config_path = resolve_path(meta_cfg_path, experiment['base_runner_config'])
        merged = load_merged_config(base_runner_config_path)

        if 'series_name' in experiment:
            merged['runner']['experiment_name'] = experiment['series_name']
        if 'seeds' in experiment:
            merged['runner']['seeds'] = list(experiment['seeds'])

        overrides = experiment.get('overrides', {}) or {}
        grid = normalize_grid(experiment.get('grid', {}) or {})

        overlap = set(overrides) & set(grid)
        ensure(not overlap, f"Experiment {experiment['name']} has keys present in both overrides and grid: {sorted(overlap)}")

        # Применяем оверрайды к базовому merged
        apply_overrides(merged, overrides)

        # Перезагружаем конфиги, если изменились пути
        refresh_merged_configs(merged, base_runner_config_path)

        # Повторно применяем оверрайды для agent/env
        agent_env_overrides = {k: v for k, v in overrides.items() if k.startswith('agent.') or k.startswith('env.')}
        apply_overrides(merged, agent_env_overrides)

        # Получаем список сидов
        seeds = merged['runner'].get('seeds', [])
        if not seeds:
            raise ValueError(f"Experiment {experiment['name']} has no seeds defined.")

        # Для каждой комбинации grid создаём один вариант (папку) и затем jobs для каждого seed
        for combo in expand_grid(grid):
            variant = copy.deepcopy(merged)
            apply_overrides(variant, combo)

            # Создаём идентификатор варианта (без seed)
            variant_name, run_id = build_run_identity(
                experiment=experiment,
                runner_cfg=variant['runner'],
                grid_values=combo,
                overrides=overrides,
            )

            # Материализуем папку варианта (один раз для всех сидов)
            variant_dir = materialize_variant(
                work_dir=work_dir,
                experiment=experiment,
                merged_variant=variant,
                variant_name=variant_name,
                run_id=run_id,
                grid_values=combo,
                overrides=overrides,
            )

            # Для каждого seed создаём отдельный job
            runner_config_path = variant_dir / 'runner.yaml'
            for seed in seeds:
                cmd = shlex.split(experiment['command']) + ['--runner-config', str(runner_config_path), '--seed', str(seed)]
                jobs.append({
                    'series_name': variant['runner'].get('experiment_name', experiment['name']),
                    'experiment_name': experiment['name'],
                    'variant_name': variant_name,
                    'run_id': run_id,           # одинаковый для всех сидов варианта
                    'seed': seed,
                    'variant_dir': variant_dir,
                    'runner_config': runner_config_path,
                    'command': cmd,
                    'command_str': ' '.join(shlex.quote(x) for x in cmd),
                })
    return jobs, work_dir


def run_one_job(job: dict, dry_run: bool = False) -> dict:
    started_at = time.time()
    if dry_run:
        return {'job': job, 'returncode': 0, 'duration_sec': 0.0, 'status': 'dry_run'}

    log_file = Path(job['variant_dir']) / f"stdout_seed_{job['seed']}.log"
    err_file = Path(job['variant_dir']) / f"stderr_seed_{job['seed']}.log"
    with open(log_file, 'w') as out, open(err_file, 'w') as err:
        proc = subprocess.run(job['command'], stdout=out, stderr=err, text=True)
    return {
        'job': job,
        'returncode': proc.returncode,
        'duration_sec': time.time() - started_at,
        'status': 'ok' if proc.returncode == 0 else 'failed',
    }


def run_jobs_sequential(jobs: list[dict], dry_run: bool = False) -> list[dict]:
    results = []
    for job in jobs:
        print(f"[RUN] {job['run_id']} (seed={job['seed']}) :: {job['command_str']}", flush=True)
        result = run_one_job(job, dry_run=dry_run)
        print(f"[{result['status'].upper()}] {job['run_id']} seed={job['seed']} (returncode={result['returncode']}, {result['duration_sec']:.2f}s)", flush=True)
        results.append(result)
    return results


def run_jobs_parallel(jobs: list[dict], max_parallel: int, dry_run: bool = False) -> list[dict]:
    results = []
    pending = list(jobs)
    running = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        while pending or running:
            while pending and len(running) < max_parallel:
                job = pending.pop(0)
                print(f"[RUN] {job['run_id']} (seed={job['seed']}) :: {job['command_str']}", flush=True)
                fut = pool.submit(run_one_job, job, dry_run)
                running[fut] = job
            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                result = fut.result()
                job = running.pop(fut)
                print(f"[{result['status'].upper()}] {job['run_id']} seed={job['seed']} (returncode={result['returncode']}, {result['duration_sec']:.2f}s)", flush=True)
                results.append(result)
    return results


def write_summary(jobs: list[dict], results: list[dict], out_path: Path) -> None:
    summary = {
        'num_jobs': len(jobs),
        'num_failed': sum(r['returncode'] != 0 for r in results),
        'results': [
            {
                'series_name': r['job']['series_name'],
                'experiment_name': r['job']['experiment_name'],
                'variant_name': r['job']['variant_name'],
                'run_id': r['job']['run_id'],
                'seed': r['job']['seed'],
                'returncode': r['returncode'],
                'duration_sec': r['duration_sec'],
                'status': r['status'],
                'runner_config': str(r['job']['runner_config']),
                'command': r['job']['command'],
            }
            for r in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-config', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    meta_cfg_path = Path(args.meta_config).resolve()
    meta_cfg = load_yaml(meta_cfg_path)
    schedule = meta_cfg.get('schedule', 'sequential')
    if schedule not in {'sequential', 'parallel'}:
        raise ValueError(f"Unknown schedule='{schedule}'. Expected sequential or parallel.")

    jobs, work_dir = build_jobs(meta_cfg, meta_cfg_path)
    print(f'Built {len(jobs)} jobs.', flush=True)

    if schedule == 'sequential':
        results = run_jobs_sequential(jobs, dry_run=args.dry_run)
    else:
        max_parallel = int(meta_cfg.get('max_parallel', 1))
        if max_parallel < 1:
            raise ValueError('max_parallel must be >= 1')
        results = run_jobs_parallel(jobs, max_parallel=max_parallel, dry_run=args.dry_run)

    write_summary(jobs, results, work_dir / 'summary.json')

    num_failed = sum(r['returncode'] != 0 for r in results)
    if num_failed > 0:
        print(f'{num_failed} job(s) failed.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()