"""Entry point: python main.py --config configs/runner/<name>.yaml --seed <n>

Thin proxy for stream_rl.experiments.runners.base_runner's own CLI (the
actual single-run entry point); kept as a stable top-level script name.
"""
import argparse

from stream_rl.experiments.runners.base_runner import (
    build_run_dir,
    build_scenario,
    copy_materialized_configs,
    configure_jax_platform,
    load_scenario_cfg,
    load_yaml,
    resolve_config_path,
    ExperimentRunner,
)
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a runner config (see configs/runner/).")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    runner_cfg_path = Path(args.config).resolve()
    runner_cfg = load_yaml(runner_cfg_path)
    runner_cfg.pop("seeds", None)

    agent_cfg = load_yaml(resolve_config_path(runner_cfg_path, runner_cfg["agent_config"]))
    env_cfg = load_yaml(resolve_config_path(runner_cfg_path, runner_cfg["env_config"]))
    scenario_cfg = load_scenario_cfg(runner_cfg, runner_cfg_path)

    configure_jax_platform(agent_cfg)

    run_dir = build_run_dir(runner_cfg)
    copy_materialized_configs(run_dir, runner_cfg_path, runner_cfg)

    scenario = build_scenario(scenario_cfg)
    runner = ExperimentRunner(
        env_cfg=env_cfg, agent_cfg=agent_cfg, runner_cfg=runner_cfg, run_dir=run_dir, scenario=scenario,
    )
    runner.run(seed=args.seed)


if __name__ == "__main__":
    main()
