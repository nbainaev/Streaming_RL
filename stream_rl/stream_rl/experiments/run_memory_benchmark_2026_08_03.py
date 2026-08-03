"""Resumable 432-run streaming-memory benchmark requested on 2026-08-03.

The scheduler materializes deterministic configs, launches one process per
(environment, model, memory condition, seed), and writes a completion marker
only after base_runner exits successfully.  Re-running the same command skips
completed seeds and lets base_runner resume incomplete ones from checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PACKAGE_ROOT / "stream_rl" / "experiments" / "configs"
CAMPAIGN_ROOT = CONFIG_ROOT / "campaigns" / "memory_benchmark_2026_08_03"
RESULT_ROOT = PACKAGE_ROOT.parent / "logs" / "memory_benchmark_2026_08_03"
SERIES = "memory_benchmark_2026_08_03"
SEEDS = (0, 1, 2)
LENGTHS = (2, 5, 10, 20, 30, 40, 50)

MODEL_CONFIGS = {
    "ppo_gru": ("ppo_gru.yaml", "ppo_gru_frozen_ssm.yaml"),
    "ac_gru_bptt1": (
        "stream_ac_gru_bptt1.yaml",
        "stream_ac_gru_bptt1_frozen_ssm.yaml",
    ),
    "ac_rtu_rtrl_bptt1": (
        "stream_ac_rtu_rtrl.yaml",
        "stream_ac_rtu_rtrl_frozen_ssm_matched.yaml",
    ),
    "ac_eprop_gru": ("eprop_stream_ac.yaml", "eprop_stream_ac_frozen_ssm.yaml"),
}

POPGYM_ENVS = {
    "autoencode_easy": "popgym-AutoencodeEasy-v0",
    "concentration_easy": "popgym-ConcentrationEasy-v0",
    "repeatprevious_easy": "popgym-RepeatPreviousEasy-v0",
    "battleship_easy": "popgym-BattleshipEasy-v0",
}


@dataclass(frozen=True)
class Job:
    phase: str
    task: str
    model: str
    memory: str
    seed: int
    runner_path: Path
    run_dir: Path
    total_steps: int

    @property
    def key(self) -> str:
        return f"{self.phase}/{self.task}/{self.model}/{self.memory}/seed_{self.seed}"

    @property
    def completion_path(self) -> Path:
        return self.run_dir / f"complete_seed_{self.seed}.json"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def materialize_variant(
    *, phase: str, task: str, model: str, memory: str, total_steps: int,
    env_cfg: dict, agent_source: str, smoke: bool,
) -> tuple[Path, Path]:
    suffix = "smoke" if smoke else "full"
    run_id = f"{suffix}__{phase}__{task}__{model}__{memory}"
    variant_cfg_dir = CAMPAIGN_ROOT / suffix / phase / task / model / memory
    run_dir = RESULT_ROOT / SERIES / run_id

    agent_cfg = load_yaml(CONFIG_ROOT / "agents" / agent_source)
    # A 125-step PPO rollout makes all requested budgets exactly divisible:
    # 100k, 300k, and 1M. Both memory conditions receive the same override.
    if agent_cfg["name"] == "ppo":
        agent_cfg["num_steps"] = 125

    agent_path = variant_cfg_dir / "agent.yaml"
    env_path = variant_cfg_dir / "env.yaml"
    runner_path = variant_cfg_dir / "runner.yaml"
    save_yaml(agent_path, agent_cfg)
    save_yaml(env_path, env_cfg)
    save_yaml(
        runner_path,
        {
            "experiment_name": SERIES,
            "run_id": run_id,
            "agent_config": str(agent_path.resolve()),
            "env_config": str(env_path.resolve()),
            "log_root": str(RESULT_ROOT.resolve()),
            "total_timesteps": total_steps,
            "step_chunk": 5_000 if phase in {"passive", "active"} else 1_000,
            "log_every": 5000 if total_steps <= 300_000 else 10_000,
            "checkpoint_every": (
                25_000 if total_steps <= 100_000
                else 50_000 if total_steps <= 300_000
                else 100_000
            ),
            "eval_every": -1,
            "scenario": {"type": "none"},
        },
    )
    return runner_path, run_dir


def build_jobs(phase: str, smoke: bool = False) -> list[Job]:
    phases = ("passive", "active", "popgym") if phase == "all" else (phase,)
    jobs: list[Job] = []
    seeds = (0,) if smoke else SEEDS

    for current_phase in phases:
        if current_phase in {"passive", "active"}:
            lengths = (2,) if smoke else LENGTHS
            total_steps = 1_000 if smoke else (100_000 if current_phase == "passive" else 300_000)
            for length in lengths:
                task = f"L{length}"
                env_cfg = {
                    "namespace": "tmaze",
                    "env_id": f"tmaze_{current_phase}",
                    "kwargs": {"corridor_length": length, "goal_reward": 1.0},
                }
                for model, sources in MODEL_CONFIGS.items():
                    for memory, source in zip(("base", "frozen_ssm"), sources):
                        runner_path, run_dir = materialize_variant(
                            phase=current_phase,
                            task=task,
                            model=model,
                            memory=memory,
                            total_steps=total_steps,
                            env_cfg=env_cfg,
                            agent_source=source,
                            smoke=smoke,
                        )
                        jobs.extend(
                            Job(current_phase, task, model, memory, seed, runner_path, run_dir, total_steps)
                            for seed in seeds
                        )
        else:
            total_steps = 1_000 if smoke else 1_000_000
            envs = {"autoencode_easy": POPGYM_ENVS["autoencode_easy"]} if smoke else POPGYM_ENVS
            for task, env_id in envs.items():
                env_cfg = {"namespace": "popgym", "env_id": env_id, "kwargs": {}}
                for model, sources in MODEL_CONFIGS.items():
                    for memory, source in zip(("base", "frozen_ssm"), sources):
                        runner_path, run_dir = materialize_variant(
                            phase=current_phase,
                            task=task,
                            model=model,
                            memory=memory,
                            total_steps=total_steps,
                            env_cfg=env_cfg,
                            agent_source=source,
                            smoke=smoke,
                        )
                        jobs.extend(
                            Job(current_phase, task, model, memory, seed, runner_path, run_dir, total_steps)
                            for seed in seeds
                        )
    return jobs


def is_complete(job: Job) -> bool:
    if job.completion_path.exists():
        return True
    final_checkpoint = job.run_dir / "checkpoints" / f"seed_{job.seed}" / f"step_{job.total_steps}.msgpack"
    if final_checkpoint.exists():
        job.completion_path.write_text(json.dumps({
            "job": job.key,
            "total_steps": job.total_steps,
            "recovered_from_final_checkpoint": True,
        }, indent=2))
        return True
    return False


def launch(job: Job) -> tuple[subprocess.Popen, object, float]:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / f"scheduler_seed_{job.seed}.log"
    log_handle = log_path.open("a")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    env["MPLCONFIGDIR"] = "/private/tmp/streaming-rl-mpl-cache"
    env["JAX_COMPILATION_CACHE_DIR"] = str(
        (RESULT_ROOT / "jax_compilation_cache").resolve()
    )
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
    command = [
        sys.executable,
        "-m",
        "stream_rl.experiments.runners.base_runner",
        "--runner-config",
        str(job.runner_path),
        "--seed",
        str(job.seed),
    ]
    process = subprocess.Popen(
        command,
        cwd=PACKAGE_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle, time.time()


def run_jobs(jobs: list[Job], max_parallel: int, limit: int | None) -> int:
    pending = [job for job in jobs if not is_complete(job)]
    def resume_step(job: Job) -> int:
        checkpoint_root = job.run_dir / "checkpoints" / f"seed_{job.seed}"
        steps = []
        for path in checkpoint_root.glob("step_*.msgpack"):
            try:
                steps.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(steps, default=0)

    # Finish partially trained jobs first. New jobs are grouped by seed so
    # seed 0 populates the persistent compilation cache before seeds 1 and 2.
    pending.sort(
        key=lambda job: (
            0 if resume_step(job) > 0 else 1,
            -resume_step(job),
            job.seed,
            {"passive": 0, "active": 1, "popgym": 2}[job.phase],
            job.task,
            job.model,
            job.memory,
        )
    )
    if limit is not None:
        pending = pending[:limit]
    print(json.dumps({
        "total_jobs": len(jobs),
        "already_complete": len(jobs) - len([j for j in jobs if not is_complete(j)]),
        "scheduled_now": len(pending),
        "max_parallel": max_parallel,
    }))

    running: dict[int, tuple[Job, subprocess.Popen, object, float]] = {}
    failures = 0
    cursor = 0
    try:
        while cursor < len(pending) or running:
            while cursor < len(pending) and len(running) < max_parallel:
                job = pending[cursor]
                cursor += 1
                process, handle, started = launch(job)
                running[process.pid] = (job, process, handle, started)
                print(f"START {job.key} pid={process.pid}", flush=True)

            time.sleep(1.0)
            for pid, (job, process, handle, started) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                elapsed = time.time() - started
                if code == 0:
                    job.completion_path.write_text(json.dumps({
                        "job": job.key,
                        "total_steps": job.total_steps,
                        "walltime_seconds": elapsed,
                        "completed_at_unix": time.time(),
                    }, indent=2))
                    print(f"DONE {job.key} wall={elapsed:.1f}s", flush=True)
                else:
                    failures += 1
                    print(f"FAIL {job.key} exit={code} wall={elapsed:.1f}s", flush=True)
                del running[pid]
    except KeyboardInterrupt:
        for job, process, handle, _ in running.values():
            process.terminate()
            handle.close()
            print(f"STOP {job.key}", flush=True)
        return 130
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("passive", "active", "popgym", "all"), default="passive")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    jobs = build_jobs(args.phase, smoke=args.smoke)
    manifest_name = (
        f"smoke_manifest_{args.phase}.json"
        if args.smoke
        else f"full_manifest_{args.phase}.json"
    )
    manifest_path = CAMPAIGN_ROOT / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([
        {
            "key": job.key,
            "runner": str(job.runner_path),
            "run_dir": str(job.run_dir),
            "total_steps": job.total_steps,
        }
        for job in jobs
    ], indent=2))
    if args.materialize_only:
        print(f"materialized {len(jobs)} jobs in {manifest_path}")
        return 0
    if args.phase == "all":
        # Preserve the experimental protocol strictly: no active job starts
        # before all passive jobs complete, and no POPGym job starts before
        # all active jobs complete.
        for phase in ("passive", "active", "popgym"):
            print(f"PHASE {phase}", flush=True)
            code = run_jobs(
                build_jobs(phase, smoke=args.smoke),
                args.max_parallel,
                args.limit,
            )
            if code != 0:
                return code
        return 0
    return run_jobs(jobs, args.max_parallel, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
