"""A controlled finite-state POMDP for recurrent-representation analysis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class DelayedCueParams:
    correct_reward: float = 1.0


@struct.dataclass
class DelayedCueState:
    time_step: jax.Array
    cue: jax.Array


class DelayedCue(environment.Environment[DelayedCueState, DelayedCueParams]):
    """Remember a binary cue across blank observations, then report it."""

    def __init__(
        self,
        delay: int = 5,
        dense_rewards: bool = False,
        noise_dim: int = 0,
        noise_std: float = 0.0,
    ):
        if delay < 1:
            raise ValueError("delay must be >= 1")
        self.delay = int(delay)
        self.dense_rewards = bool(dense_rewards)
        if noise_dim < 0 or noise_std < 0:
            raise ValueError("noise_dim and noise_std must be non-negative")
        self.noise_dim = int(noise_dim)
        self.noise_std = float(noise_std)
        self.decision_time = self.delay + 1
        self.episode_length = self.delay + 2

    @property
    def default_params(self):
        return DelayedCueParams()

    @property
    def num_actions(self):
        return 2

    def action_space(self, params=None):
        return spaces.Discrete(2)

    def observation_space(self, params=None):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(2 + self.noise_dim,),
            dtype=jnp.float32,
        )

    def state_space(self, params=None):
        return spaces.Dict(
            {
                "time_step": spaces.Discrete(self.episode_length),
                "cue": spaces.Discrete(2),
            }
        )

    def _obs(self, state, key=None):
        signed_cue = jnp.where(state.cue == 0, -1.0, 1.0)
        cue_obs = jnp.where(state.time_step == 0, signed_cue, 0.0)
        decision_obs = (state.time_step == self.decision_time).astype(jnp.float32)
        core = jnp.asarray([cue_obs, decision_obs], dtype=jnp.float32)
        if self.noise_dim == 0:
            return core
        noise = (
            jnp.zeros((self.noise_dim,), dtype=jnp.float32)
            if key is None
            else self.noise_std * jax.random.normal(key, (self.noise_dim,))
        )
        return jnp.concatenate([core, noise.astype(jnp.float32)])

    def reset_env(self, key, params):
        del params
        cue_key, obs_key = jax.random.split(key)
        cue = jax.random.bernoulli(cue_key).astype(jnp.int32)
        state = DelayedCueState(
            time_step=jnp.asarray(0, dtype=jnp.int32),
            cue=cue,
        )
        return self._obs(state, obs_key), state

    def get_obs(self, state, params=None, key=None):
        del params
        return self._obs(state, key)

    def step_env(self, key, state, action, params):
        at_decision = state.time_step == self.decision_time
        correct = jnp.asarray(action, dtype=jnp.int32) == state.cue
        rewardable = jnp.where(self.dense_rewards, state.time_step > 0, at_decision)
        reward = jnp.where(rewardable & correct, params.correct_reward, 0.0).astype(jnp.float32)
        done = at_decision
        next_state = DelayedCueState(
            time_step=jnp.minimum(state.time_step + 1, self.decision_time),
            cue=state.cue,
        )
        info = {
            "success": at_decision & correct,
            "cue": state.cue,
            "pomdp_state": state.cue * self.episode_length + state.time_step,
            "time_step": state.time_step,
        }
        return self._obs(next_state, key), next_state, reward, done, info

    def is_terminal(self, state, params):
        del params
        return state.time_step >= self.decision_time
