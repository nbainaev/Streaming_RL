from memorax.environments import make as memorax_make
from memorax.environments.wrappers import RecordEpisodeStatistics

from stream_rl.src.env.popgym_wrapper import make as popgym_make
from stream_rl.src.env.tmaze import (
    TMazeBase,
    TMazeClassicActive,
    TMazeClassicPassive,
    TMazePassiveForced,
)
from stream_rl.src.env.ctgraph import CTGraph
from stream_rl.src.env.continuing_statistics import RecordContinuingEpisodeStatistics
from stream_rl.src.env.delayed_cue import DelayedCue


def _make_tmaze_env(cfg: dict):
    env_id = cfg["env_id"]
    kwargs = dict(cfg.get("kwargs", {}))
    # goal_reward/penalty are overrides on top of default_params (which
    # auto-derives penalty = -1/(episode_length - 1), matching the Passive
    # T-Maze reward definition) rather than constructor args, since they
    # live on TMazeParams, not on the env instance.
    goal_reward = kwargs.pop("goal_reward", None)
    penalty = kwargs.pop("penalty", None)
    oracle_reward = kwargs.pop("oracle_reward", None)

    if env_id == "tmaze_passive":
        env = TMazeClassicPassive(**kwargs)
    elif env_id == "tmaze_passive_forced":
        env = TMazePassiveForced(**kwargs)
    elif env_id == "tmaze_active":
        env = TMazeClassicActive(**kwargs)
    elif env_id == "tmaze_base":
        env = TMazeBase(**kwargs)
    else:
        raise ValueError(
            f"Unknown tmaze env_id={env_id!r}. "
            "Expected one of: 'tmaze_base', 'tmaze_passive', "
            "'tmaze_passive_forced', 'tmaze_active'."
        )

    env_params = env.default_params
    if goal_reward is not None:
        env_params = env_params.replace(goal_reward=float(goal_reward))
    if penalty is not None:
        env_params = env_params.replace(penalty=float(penalty))
    if oracle_reward is not None:
        env_params = env_params.replace(oracle_reward=float(oracle_reward))
    return env, env_params


def make_env(cfg: dict, num_envs: int):
    namespace = cfg["namespace"]

    if namespace == "tmaze":
        env, env_params = _make_tmaze_env(cfg)
        env = RecordEpisodeStatistics(env)
        return env, env_params

    if namespace == "ctgraph":
        kwargs = dict(cfg.get("kwargs", {}))
        high_reward = float(kwargs.pop("high_reward", 1.0))
        fail_reward = float(kwargs.pop("fail_reward", -1.0))
        env = CTGraph(**kwargs)
        env_params = env.default_params.replace(
            high_reward=high_reward, fail_reward=fail_reward
        )
        env = (
            RecordContinuingEpisodeStatistics(env)
            if env.continuing_task
            else RecordEpisodeStatistics(env)
        )
        return env, env_params

    if namespace == "delayed_cue":
        kwargs = dict(cfg.get("kwargs", {}))
        correct_reward = float(kwargs.pop("correct_reward", 1.0))
        env = DelayedCue(**kwargs)
        env_params = env.default_params.replace(correct_reward=correct_reward)
        env = RecordEpisodeStatistics(env)
        return env, env_params

    kwargs = dict(cfg.get("kwargs", {}))

    if namespace == "popgym":
        # Use our own wrapper instead of memorax's: it additionally supports
        # Tuple observation spaces (popgym-Autoencode*) and MultiDiscrete
        # action spaces (popgym-Battleship*), which memorax's PopGymWrapper
        # rejects. See popgym_wrapper.py for details.
        kwargs["batch_shape"] = (num_envs,)
        env, env_params = popgym_make(cfg["env_id"], **kwargs)
        env = RecordEpisodeStatistics(env)
        return env, env_params

    env_id = f"{namespace}::{cfg['env_id']}"
    env, env_params = memorax_make(env_id, **kwargs)
    env = RecordEpisodeStatistics(env)
    return env, env_params
