import argparse
import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def resolve_config_path(base_runner_cfg_path: str | Path, raw_path: str | Path) -> Path:
    path = resolve_path(base_runner_cfg_path, raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config path not found: raw={raw_path!r}, resolved={str(path)!r}, "
            f"base_runner_cfg={str(Path(base_runner_cfg_path).resolve())!r}"
        )
    return path


def _merge_shallow(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    merged.update(copy.deepcopy(override))
    return merged


def deep_update_existing(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if key not in dst:
            raise KeyError(f'Unknown override key: {key}')
        if isinstance(dst[key], dict) and isinstance(value, dict):
            deep_update_existing(dst[key], value)
        else:
            dst[key] = value
    return dst


def checkpoint_dir(run_dir: Path, seed: int) -> Path:
    return Path(run_dir) / 'checkpoints' / f'seed_{seed}'


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.split('_')[1])


def latest_checkpoint_step(run_dir: Path, seed: int) -> int | None:
    ckpt_dir = checkpoint_dir(run_dir, seed)
    if not ckpt_dir.exists():
        return None
    steps = [_checkpoint_step(p) for p in ckpt_dir.glob('step_*.msgpack')]
    return max(steps) if steps else None


def save_checkpoint(state, run_dir: Path, seed: int, step: int, keep_last: int = 2) -> None:
    """Serializes the agent's full training state (a flax.struct pytree of
    arrays -- params, optimizer moments, traces, env state, RNG-derived
    counters) so a killed/interrupted run can resume instead of retraining
    from scratch. Keeps only the `keep_last` most recent checkpoints per
    (run_dir, seed) to bound disk use across ~400 runs."""
    import flax.serialization

    ckpt_dir = checkpoint_dir(run_dir, seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'step_{step}.msgpack'
    path.write_bytes(flax.serialization.to_bytes(state))

    existing = sorted(ckpt_dir.glob('step_*.msgpack'), key=_checkpoint_step)
    for stale in existing[:-keep_last]:
        stale.unlink()


def load_checkpoint(state_template, run_dir: Path, seed: int, step: int):
    """`state_template` must be a freshly-`agent.init(...)`-ed state with the
    same pytree structure as what was serialized (flax.serialization needs a
    target to restore into)."""
    import flax.serialization

    path = checkpoint_dir(run_dir, seed) / f'step_{step}.msgpack'
    return flax.serialization.from_bytes(state_template, path.read_bytes())


def configure_jax_platform(agent_cfg: dict) -> None:
    device = agent_cfg.get('device', 'auto').lower()
    if device == 'auto':
        return
    if device not in {'cpu', 'gpu'}:
        raise ValueError(f"Unknown device={device!r}. Expected one of: auto, cpu, gpu.")
    # JAX_PLATFORMS wants the backend name JAX itself registers ('cuda'), not
    # the generic 'gpu' this config's device key uses -- setting
    # JAX_PLATFORMS=gpu makes JAX try (and fail on) an unregistered 'gpu'
    # backend instead of falling back to 'cuda'. Keep 'cpu' registered too:
    # PopGym envs are plain Python/numpy gym envs stepped through
    # jax.pure_callback (see src/env/popgym_wrapper.py), which needs a local
    # CPU device to place its inputs on even while the rest of the graph
    # runs on GPU -- restricting to 'cuda' alone makes that callback crash.
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if device == 'gpu' else device


@dataclass
class ScenarioEvent:
    at_step: int
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    executed: bool = False


class BaseScenario:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def on_run_start(self, runner):
        return None

    def on_seed_start(self, runner):
        return None

    def on_chunk_end(self, runner):
        return None

    def on_seed_end(self, runner):
        return None

    def on_run_end(self, runner):
        return None


class NoOpScenario(BaseScenario):
    pass


class EventScenario(BaseScenario):
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        events = cfg.get('events', []) if cfg else []
        self.events = [
            ScenarioEvent(
                at_step=int(event['at_step']),
                action=str(event['action']),
                params=dict(event.get('params', {})),
            )
            for event in events
        ]
        self.events.sort(key=lambda e: e.at_step)

    def on_chunk_end(self, runner):
        for event in self.events:
            if event.executed:
                continue
            if runner.step >= event.at_step:
                runner.apply_scenario_action(event.action, event.params)
                runner.log_event(event.action, event.at_step, event.params)
                event.executed = True


SCENARIO_REGISTRY = {
    'none': NoOpScenario,
    'noop': NoOpScenario,
    'event': EventScenario,
}


class ExperimentRunner:
    def __init__(self, env_cfg: dict, agent_cfg: dict, runner_cfg: dict, run_dir: Path, scenario: BaseScenario):
        self.env_cfg = copy.deepcopy(env_cfg)
        self.agent_cfg = copy.deepcopy(agent_cfg)
        self.runner_cfg = copy.deepcopy(runner_cfg)
        self.run_dir = Path(run_dir)
        self.scenario = scenario

        self.seed = None
        self.step = 0
        self.episode_idx = 0
        self.frozen = False
        self.env = None
        self.env_params = None
        self.agent = None
        self.state = None
        self.key = None
        self.base_key = None
        self.monitor_logger = None
        self.metrics_logger = None
        self.console_logger = None
        self.first_state_repr = None
        self.event_log_path = self.run_dir / 'scenario_events.jsonl'

    def build_env(self):
        from stream_rl.src.env.factory import make_env
        env, env_params = make_env(self.env_cfg, num_envs=self.agent_cfg['num_envs'])
        return env, env_params

    def build_agent(self, env, env_params):
        from stream_rl.src.agents.factory import build_agent
        return build_agent(self.agent_cfg['name'], self.agent_cfg, env, env_params)

    def setup_seed(self, seed: int):
        import jax
        from stream_rl.src.env.monitor_logger import EpisodeMonitorLogger
        from stream_rl.src.utils.console_logger import ConsoleLogger

        self.seed = int(seed)
        self.step = 0
        self.episode_idx = 0
        self.frozen = False
        self.monitor_logger = EpisodeMonitorLogger(run_dir=self.run_dir, seed=self.seed, prefix='monitor')
        self.metrics_logger = EpisodeMonitorLogger(run_dir=self.run_dir, seed=self.seed, prefix='metrics')
        self.console_logger = ConsoleLogger()
        self.env, self.env_params = self.build_env()
        self.agent = self.build_agent(self.env, self.env_params)
        root_key = jax.random.key(self.seed)
        init_key, base_key = jax.random.split(root_key)
        self.state = self.agent.init(init_key)
        # base_key is fixed for the whole seed; every training chunk derives
        # its key from base_key + the *absolute* step count via fold_in, so
        # the RNG stream (and therefore the learning trajectory) no longer
        # depends on how training is chopped into chunks (i.e. on log_every).
        self.base_key = base_key
        self.key = base_key

        # Resume from the latest checkpoint for this (run_dir, seed) if one
        # exists -- e.g. a tmux session that was killed and relaunched with
        # the exact same `base_runner.py --runner-config ... --seed ...`
        # command. The freshly-init'ed state above is only used as the
        # pytree template flax.serialization restores into.
        resume_step = latest_checkpoint_step(self.run_dir, self.seed)
        if resume_step is not None:
            self.state = load_checkpoint(self.state, self.run_dir, self.seed, resume_step)
            self.step = int(resume_step)

        if self.first_state_repr is None:
            self.first_state_repr = str(self.state)

    def min_chunk(self) -> int:
        """Smallest step count `agent.train`/`evaluate` can be called with —
        also used as the actual training-scan granularity (see run_seed),
        independent of log_every."""
        if self.agent_cfg['name'] == 'ppo':
            return self.agent_cfg['num_envs'] * self.agent_cfg['num_steps']
        if self.agent_cfg['name'] == 'stream_tbptt':
            return self.agent_cfg['num_envs'] * self.agent_cfg['tbptt_steps']
        return int(self.agent_cfg['num_envs'])

    def log_event(self, action: str, at_step: int, params: dict):
        record = {
            'seed': self.seed,
            'step': self.step,
            'trigger_step': at_step,
            'action': action,
            'params': params,
        }
        with open(self.event_log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def apply_scenario_action(self, action: str, params: dict):
        if action == 'change_setup':
            self.change_setup(
                env_overrides=params.get('env_overrides'),
                agent_overrides=params.get('agent_overrides'),
                rebuild_agent=bool(params.get('rebuild_agent', False)),
                reset_state=bool(params.get('reset_state', False)),
            )
            return
        if action == 'freeze_agent':
            self.freeze_agent()
            return
        if action == 'unfreeze_agent':
            self.unfreeze_agent()
            return
        raise ValueError(f'Unknown scenario action: {action}')

    def _space_signature(self, env, env_params) -> tuple[Any, Any]:
        obs_shape = tuple(env.observation_space(env_params).shape)
        action_n = getattr(env.action_space(env_params), 'n', None)
        return obs_shape, action_n

    def change_setup(self, env_overrides: dict | None = None, agent_overrides: dict | None = None, rebuild_agent: bool = False, reset_state: bool = False):
        new_env_cfg = copy.deepcopy(self.env_cfg)
        new_agent_cfg = copy.deepcopy(self.agent_cfg)

        env_overrides = self._resolve_env_overrides(env_overrides)
        if env_overrides:
            deep_update_existing(new_env_cfg, env_overrides)
        if agent_overrides:
            deep_update_existing(new_agent_cfg, agent_overrides)

        new_env, new_env_params = self.build_env_from_cfg(new_env_cfg, new_agent_cfg)
        old_sig = self._space_signature(self.env, self.env_params)
        new_sig = self._space_signature(new_env, new_env_params)

        if old_sig != new_sig and not rebuild_agent:
            raise ValueError(
                'change_setup would change observation/action signature; '
                'set rebuild_agent=true explicitly if that is intended. '
                f'old={old_sig}, new={new_sig}'
            )

        self.env_cfg = new_env_cfg
        self.agent_cfg = new_agent_cfg
        self.env = new_env
        self.env_params = new_env_params

        if rebuild_agent:
            self.agent = self.build_agent(self.env, self.env_params)
            if reset_state:
                import jax
                init_key, next_key = jax.random.split(self.key)
                self.state = self.agent.init(init_key)
                self.key = next_key
            else:
                raise ValueError(
                    'rebuild_agent=true with reset_state=false is unsafe because state structure '
                    'may no longer match the rebuilt agent.'
                )
        elif env_overrides:
            # Algorithms retain the environment object internally.  Merely
            # replacing runner.env leaves training bound to the old setup,
            # making same-signature curricula silently ineffective.  Rebuild
            # the stateless algorithm/network definitions against the new env
            # while preserving the complete online learning state.
            self.agent = self.build_agent(self.env, self.env_params)

    def build_env_from_cfg(self, env_cfg: dict, agent_cfg: dict):
        from stream_rl.src.env.factory import make_env
        return make_env(env_cfg, num_envs=agent_cfg['num_envs'])

    def freeze_agent(self):
        self.frozen = True

    def unfreeze_agent(self):
        self.frozen = False

    def _resolve_env_overrides(self, env_overrides: dict | None) -> dict | None:
        if not env_overrides:
            return None

        resolved = copy.deepcopy(env_overrides)
        kwargs = resolved.get('kwargs', {})
        if 'corridor_length_delta' in kwargs:
            delta = int(kwargs.pop('corridor_length_delta'))
            current = int(self.env_cfg['kwargs']['corridor_length'])
            kwargs['corridor_length'] = max(1, current + delta)
        resolved['kwargs'] = kwargs
        return resolved

    def train_chunk(self, chunk: int):
        """Runs exactly `chunk` steps and returns the raw logdict. Does NOT
        do any periodic (log_every-scale) logging itself — see run_seed,
        which accumulates these across chunks and flushes on its own
        schedule, so log_every can't change the RNG stream or the training
        trajectory (only how often results are written out)."""
        import jax
        import lox
        from stream_rl.src.utils.lox_compat import patch_lox_scan_metadata

        patch_lox_scan_metadata()

        step_start = self.step
        train_mode = not self.frozen
        runner_fn = self.agent.train if train_mode else self.agent.evaluate

        chunk_fn = lox.spool(runner_fn)
        # fold_in(base_key, step_start) depends only on the absolute step
        # count reached so far, not on how many chunks got us there — unlike
        # sequential jax.random.split, this makes the key (and therefore the
        # whole learning trajectory) independent of the chunk size.
        chunk_key = jax.random.fold_in(self.base_key, step_start)
        self.state, logs = chunk_fn(chunk_key, self.state, chunk)

        info = logs.get('info', {})
        self.step += chunk
        num_envs = int(self.agent_cfg['num_envs'])
        self._log_completed_episodes(info, step_start=step_start, num_envs=num_envs)

        return logs, train_mode

    def _flush_metrics(self, logs, train_mode: bool) -> None:
        import numpy as np

        info = logs.get('info', {})
        scalar_logs = logs.filter(lambda k, v: k not in ('info', 'intermediates'))
        last_scalars = dict(scalar_logs.reduce('last'))

        if train_mode and last_scalars:
            self.metrics_logger.log(
                {k: float(np.asarray(v).reshape(-1)[0]) for k, v in last_scalars.items()},
                step=self.step,
            )

        extra = {'info': info, **last_scalars}
        if self.frozen:
            extra['info'] = {**extra.get('info', {}), 'frozen': True}

        self.console_logger.log(extra, step=self.step)

    def _log_completed_episodes(self, info: dict, step_start: int, num_envs: int) -> None:
        import numpy as np

        done = np.asarray(info['returned_episode']).reshape(-1, num_envs).astype(bool)
        returns = np.asarray(info['returned_episode_returns']).reshape(-1, num_envs)
        disc_returns = np.asarray(info['returned_discounted_episode_returns']).reshape(-1, num_envs)
        lengths = np.asarray(info['returned_episode_lengths']).reshape(-1, num_envs)
        # 'success' is env-specific (PassiveTMaze/ActiveTMaze emit it; PopGym
        # doesn't) — only logged when present, so the CSV schema stays
        # consistent for envs that never emit it.
        success = np.asarray(info['success']).reshape(-1, num_envs) if 'success' in info else None
        optional_episode_fields = {
            key: np.asarray(info[key]).reshape(-1, num_envs)
            for key in ('path_score', 'reward_phase', 'reward_switched')
            if key in info
        }

        t_idx, env_idx = np.nonzero(done)
        if len(t_idx) == 0:
            return

        order = np.argsort(t_idx)
        for t, e in zip(t_idx[order], env_idx[order]):
            episode_step = step_start + (int(t) + 1) * num_envs
            self.episode_idx += 1
            episode_info = {
                'returned_episode_returns': float(returns[t, e]),
                'returned_discounted_episode_returns': float(disc_returns[t, e]),
                'returned_episode_lengths': float(lengths[t, e]),
            }
            if success is not None:
                episode_info['success'] = float(success[t, e])
            for key, values in optional_episode_fields.items():
                episode_info[key] = float(values[t, e])
            self.monitor_logger.log(
                {
                    'episode_idx': int(self.episode_idx),
                    'total_steps': int(episode_step),
                    'info': episode_info,
                },
                step=episode_step,
            )

    def _reduce_episode_info(self, info: dict) -> dict:
        import jax.numpy as jnp

        done_mask = jnp.asarray(info['returned_episode']).reshape(-1).astype(bool)
        n_completed = int(done_mask.sum())

        def masked_mean(key):
            values = jnp.asarray(info[key]).reshape(-1)
            return float(jnp.where(done_mask, values, 0.0).sum() / max(n_completed, 1)) if n_completed > 0 else float('nan')

        return {
            'returned_episode_returns': masked_mean('returned_episode_returns'),
            'returned_discounted_episode_returns': masked_mean('returned_discounted_episode_returns'),
            'returned_episode_lengths': masked_mean('returned_episode_lengths'),
            'n_completed_episodes': n_completed,
        }

    def run_seed(self, seed: int):
        total_updates = int(self.runner_cfg['total_timesteps'])
        log_every = int(self.runner_cfg.get('log_every', 5000))
        min_chunk = self.min_chunk()
        # step_chunk is the actual jax.lax.scan granularity: independent of
        # log_every on purpose (see train_chunk/run_seed below), but NOT as
        # fine as min_chunk by default — one dispatch per env-step is
        # extremely slow. Override via runner_cfg['step_chunk'] if you need
        # exact scenario-event timing finer than 100 steps.
        step_chunk = int(self.runner_cfg.get('step_chunk', 100))
        # Round UP to the nearest multiple of min_chunk (e.g. PPO's own
        # num_envs*num_steps rollout batch, which can't be split — 384 stays
        # 384 no matter what step_chunk is requested). log_every does NOT
        # need to be an exact multiple of step_chunk: the flush check below
        # is a ">=" threshold, so a mismatch just means the odd remainder
        # steps get folded into the next flush — no effect on correctness.
        step_chunk = -(-step_chunk // min_chunk) * min_chunk
        if log_every < step_chunk:
            log_every = step_chunk

        usable_total = (total_updates // step_chunk) * step_chunk

        # setup_seed may resume self.step from a checkpoint (see
        # load_checkpoint above) -- remaining must be computed *after* it,
        # not before, or a resumed run would retrain the already-done steps.
        self.setup_seed(seed)
        self.scenario.on_seed_start(self)
        remaining = usable_total - self.step

        checkpoint_every = int(self.runner_cfg.get('checkpoint_every', -1))
        if checkpoint_every > 0:
            # Round up to a step_chunk multiple -- checkpoints can only be
            # taken between chunks, same constraint as log_every.
            checkpoint_every = -(-checkpoint_every // step_chunk) * step_chunk

        # Training always advances in fixed step_chunk increments — this is
        # the actual jax.lax.scan granularity, independent of log_every.
        # Scenario events are checked after every such increment (so their
        # at_step trigger is exact, not rounded to the nearest log_every).
        # Every completed episode is written to monitor_seed_*.csv as it
        # happens (inside train_chunk), regardless of log_every. The
        # periodic scalar-metrics flush (console + metrics_seed_*.csv) only
        # needs the *latest* chunk's values (it's a "last" reduction, not a
        # window average), so log_every only changes how often that flush
        # happens — never the RNG stream, the training trajectory, or which
        # episodes get logged.
        latest_logs, latest_train_mode = None, True
        steps_since_flush = 0
        steps_since_checkpoint = 0
        while remaining > 0:
            logs, train_mode = self.train_chunk(step_chunk)
            remaining -= step_chunk
            latest_logs, latest_train_mode = logs, train_mode
            steps_since_flush += step_chunk
            steps_since_checkpoint += step_chunk

            self.scenario.on_chunk_end(self)

            if steps_since_flush >= log_every:
                self._flush_metrics(latest_logs, latest_train_mode)
                steps_since_flush = 0

            if checkpoint_every > 0 and steps_since_checkpoint >= checkpoint_every:
                save_checkpoint(self.state, self.run_dir, self.seed, self.step)
                steps_since_checkpoint = 0

        self.scenario.on_seed_end(self)
        if checkpoint_every > 0:
            save_checkpoint(self.state, self.run_dir, self.seed, self.step)
        self.monitor_logger.finish()
        self.metrics_logger.finish()

    def write_structure(self):
        payload = {
            'env_config': self.env_cfg,
            'agent_config': self.agent_cfg,
            'runner_config': self.runner_cfg,
            'state_repr': self.first_state_repr,
            'scenario': self.scenario.cfg,
        }
        with open(self.run_dir / 'structure.json', 'w') as f:
            json.dump(payload, f, indent=2)

    def run_chunk(self, chunk: int, train: bool):
        import jax
        import lox
        from stream_rl.src.utils.lox_compat import patch_lox_scan_metadata

        patch_lox_scan_metadata()

        runner_fn = self.agent.train if train else self.agent.evaluate
        chunk_fn = lox.spool(runner_fn)
        self.key, chunk_key = jax.random.split(self.key)
        return chunk_fn(chunk_key, self.state, chunk)

    def run(self, seed: int):
        """Запускает один сид с заданным seed."""
        self.scenario.on_run_start(self)
        self.run_seed(seed)
        self.write_structure()
        self.scenario.on_run_end(self)


def load_scenario_cfg(runner_cfg: dict, runner_cfg_path: Path) -> dict:
    scenario_cfg_inline = copy.deepcopy(runner_cfg.get('scenario', {}))
    scenario_path = runner_cfg.get('scenario_config')

    if not scenario_path:
        return scenario_cfg_inline

    resolved = resolve_config_path(runner_cfg_path, scenario_path)
    file_cfg = load_yaml(resolved)
    if scenario_cfg_inline:
        return _merge_shallow(file_cfg, scenario_cfg_inline)
    return file_cfg


def build_scenario(cfg: dict | None) -> BaseScenario:
    cfg = cfg or {'type': 'none'}
    scenario_type = str(cfg.get('type', 'none')).lower()
    if scenario_type not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario type '{scenario_type}'. Available: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[scenario_type](cfg)


def build_run_dir(runner_cfg: dict) -> Path:
    log_root = Path(runner_cfg['log_root']).resolve()
    series_name = runner_cfg.get('experiment_name', 'experiment')
    run_id = runner_cfg.get('run_id', series_name)
    run_dir = log_root / series_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_materialized_configs(run_dir: Path, original_runner_cfg_path: Path, runner_cfg: dict):
    materialized_runner_cfg = copy.deepcopy(runner_cfg)

    agent_path = resolve_config_path(original_runner_cfg_path, runner_cfg['agent_config'])
    env_path = resolve_config_path(original_runner_cfg_path, runner_cfg['env_config'])
    materialized_runner_cfg['agent_config'] = str(agent_path.resolve())
    materialized_runner_cfg['env_config'] = str(env_path.resolve())

    scenario_path = runner_cfg.get('scenario_config')
    resolved_scenario_path = None
    if scenario_path:
        resolved_scenario_path = resolve_config_path(original_runner_cfg_path, scenario_path)
        materialized_runner_cfg['scenario_config'] = str(resolved_scenario_path.resolve())

    save_yaml(run_dir / 'runner.yaml', materialized_runner_cfg)
    (run_dir / 'agent.yaml').write_text(agent_path.read_text())
    (run_dir / 'env.yaml').write_text(env_path.read_text())
    if resolved_scenario_path is not None:
        (run_dir / 'scenario.yaml').write_text(resolved_scenario_path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner-config', required=True)
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed to run (must be provided, overrides any seeds in config)')
    args = parser.parse_args()

    original_runner_cfg_path = Path(args.runner_config).resolve()
    runner_cfg = load_yaml(original_runner_cfg_path)

    if 'agent_config' not in runner_cfg:
        raise KeyError("runner config must contain 'agent_config'")
    if 'env_config' not in runner_cfg:
        raise KeyError("runner config must contain 'env_config'")

    # Удаляем поле seeds, чтобы не было путаницы
    runner_cfg.pop('seeds', None)

    agent_cfg = load_yaml(resolve_config_path(original_runner_cfg_path, runner_cfg['agent_config']))
    env_cfg = load_yaml(resolve_config_path(original_runner_cfg_path, runner_cfg['env_config']))
    scenario_cfg = load_scenario_cfg(runner_cfg, original_runner_cfg_path)

    configure_jax_platform(agent_cfg)

    run_dir = build_run_dir(runner_cfg)
    copy_materialized_configs(run_dir, original_runner_cfg_path, runner_cfg)

    scenario = build_scenario(scenario_cfg)
    runner = ExperimentRunner(
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        runner_cfg=runner_cfg,
        run_dir=run_dir,
        scenario=scenario,
    )
    runner.run(seed=args.seed)


if __name__ == '__main__':
    main()
