"""Episode-style metrics for a continuing lifelong environment.

The wrapped environment returns ``done=False`` to the agent so recurrent
state and value bootstrapping cross attempt boundaries.  CT-graph exposes
the logical root-to-leaf boundary in ``info['episode_boundary']``; this
wrapper uses that signal only for reporting per-attempt statistics.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment


@struct.dataclass
class ContinuingStatisticsState:
    env_state: environment.EnvState
    episode_returns: float
    discounted_episode_returns: float
    episode_discount: float
    episode_lengths: int
    returned_episode_returns: float
    returned_discounted_episode_returns: float
    returned_episode_lengths: int


class RecordContinuingEpisodeStatistics:
    """Record logical attempts without forwarding their boundary as done."""

    def __init__(self, env, gamma: float = 0.99):
        self._env = env
        self._gamma = gamma

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        state = ContinuingStatisticsState(
            env_state, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 0
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        boundary = jnp.asarray(info["episode_boundary"], dtype=jnp.bool_)
        new_return = state.episode_returns + reward
        new_discounted_return = (
            state.discounted_episode_returns + state.episode_discount * reward
        )
        new_discount = state.episode_discount * self._gamma
        new_length = state.episode_lengths + 1
        state = ContinuingStatisticsState(
            env_state=env_state,
            episode_returns=new_return * (1 - boundary),
            discounted_episode_returns=new_discounted_return * (1 - boundary),
            episode_discount=new_discount * (1 - boundary) + boundary,
            episode_lengths=new_length * (1 - boundary),
            returned_episode_returns=(
                state.returned_episode_returns * (1 - boundary)
                + new_return * boundary
            ),
            returned_discounted_episode_returns=(
                state.returned_discounted_episode_returns * (1 - boundary)
                + new_discounted_return * boundary
            ),
            returned_episode_lengths=(
                state.returned_episode_lengths * (1 - boundary)
                + new_length * boundary
            ),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_discounted_episode_returns"] = (
            state.returned_discounted_episode_returns
        )
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = boundary
        return obs, state, reward, done, info
