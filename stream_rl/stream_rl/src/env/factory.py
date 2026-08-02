from memorax.environments import make as memorax_make
from memorax.environments.wrappers import RecordEpisodeStatistics

from stream_rl.src.env.popgym_wrapper import make as popgym_make
from stream_rl.src.env.tmaze import (
    TMazeBase,
    TMazeClassicActive,
    TMazeClassicPassive,
)


def _make_tmaze_env(cfg: dict):
    env_id = cfg["env_id"]
    kwargs = dict(cfg.get("kwargs", {}))
    # goal_reward/penalty are overrides on top of default_params (which
    # auto-derives penalty = -1/(episode_length - 1), matching the Passive
    # T-Maze reward definition) rather than constructor args, since they
    # live on TMazeParams, not on the env instance.
    goal_reward = kwargs.pop("goal_reward", None)
    penalty = kwargs.pop("penalty", None)

    if env_id == "tmaze_passive":
        env = TMazeClassicPassive(**kwargs)
    elif env_id == "tmaze_active":
        env = TMazeClassicActive(**kwargs)
    elif env_id == "tmaze_base":
        env = TMazeBase(**kwargs)
    else:
        raise ValueError(
            f"Unknown tmaze env_id={env_id!r}. "
            "Expected one of: 'tmaze_base', 'tmaze_passive', 'tmaze_active'."
        )

    env_params = env.default_params
    if goal_reward is not None:
        env_params = env_params.replace(goal_reward=float(goal_reward))
    if penalty is not None:
        env_params = env_params.replace(penalty=float(penalty))
    return env, env_params


def make_env(cfg: dict, num_envs: int):
    namespace = cfg["namespace"]

    if namespace == "tmaze":
        env, env_params = _make_tmaze_env(cfg)
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