"""PopGym wrapper extending memorax's PopGymWrapper with two observation/
action space shapes it doesn't support, both needed to run the full popgym
benchmark suite:

- Tuple(Discrete, ...) observation spaces (e.g. popgym-Autoencode*): flattened
  into a single int32 vector, one entry per component -- the same
  representation memorax already uses for MultiDiscrete observations.
- MultiDiscrete action spaces (e.g. popgym-Battleship*): exposed to the agent
  as a single flat Discrete(prod(nvec)) action (so existing discrete-action
  policy heads work unmodified) and unravelled back into the underlying
  env's multi-dim action before stepping it.

Everything else (Box/Discrete/MultiDiscrete observations, Discrete actions)
is handled identically to memorax.environments.popgym.PopGymWrapper.
"""

import warnings

import gymnasium.spaces as gym_spaces
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from memorax.environments.popgym import PopGymState
from memorax.utils.typing import Array, Key


class ExtendedPopGymWrapper:
    def __init__(self, environment, batch_shape: tuple[int, ...] = (1,)):
        self.environment = environment
        self.batch_shape = tuple(batch_shape)

        if len(self.batch_shape) > 1:
            warnings.warn(
                f"ExtendedPopGymWrapper batch_shape={self.batch_shape} treats leading "
                "axes as seeds, but all envs share a single underlying vec env "
                "and its RNG state, so sub-batches are not independently seeded. "
                "Seed each sub-env explicitly at make-time if you need "
                "independent seeds.",
                stacklevel=2,
            )

        observation_space = environment.single_observation_space
        self._obs_is_tuple = isinstance(observation_space, gym_spaces.Tuple)
        match observation_space:
            case gym_spaces.Tuple():
                if not all(
                    isinstance(s, gym_spaces.Discrete) for s in observation_space.spaces
                ):
                    raise NotImplementedError(
                        "Unsupported popgym observation space: Tuple of non-Discrete "
                        f"components ({observation_space.spaces!r})"
                    )
                n_components = len(observation_space.spaces)
                self.observation_shape = (n_components,)
                self.observation_dtype = jnp.int32
                self.observation_low = np.zeros(n_components, dtype=np.int32)
                self.observation_high = np.asarray(
                    [int(s.n) - 1 for s in observation_space.spaces], dtype=np.int32
                )
            case gym_spaces.Box():
                self.observation_shape = observation_space.shape
                self.observation_dtype = jnp.dtype(observation_space.dtype)
                self.observation_low = np.asarray(observation_space.low)
                self.observation_high = np.asarray(observation_space.high)
            case gym_spaces.Discrete():
                # Presented as a length-1 vector rather than a scalar: the
                # policy/value feature extractors (and the e-prop cell's
                # einsum over a feature axis) assume observations carry at
                # least one feature dimension, which a rank-0 array doesn't
                # provide (hit by popgym-RepeatPrevious*/Battleship*, whose
                # observation_space is Discrete(2)).
                self.observation_shape = (1,)
                self.observation_dtype = jnp.int32
                self.observation_low = np.zeros((1,), dtype=np.int32)
                self.observation_high = np.asarray(
                    [int(observation_space.n) - 1], dtype=np.int32
                )
            case gym_spaces.MultiDiscrete():
                self.observation_shape = observation_space.shape
                self.observation_dtype = jnp.int32
                self.observation_low = np.zeros(observation_space.shape, dtype=np.int32)
                self.observation_high = np.asarray(observation_space.nvec, dtype=np.int32) - 1
            case _:
                raise NotImplementedError(
                    f"Unsupported popgym observation space: {type(observation_space).__name__}"
                )

        action_space = environment.single_action_space
        if isinstance(action_space, gym_spaces.MultiDiscrete):
            self._action_nvec = np.asarray(action_space.nvec, dtype=np.int64)
            self.num_actions = int(np.prod(self._action_nvec))
        else:
            self._action_nvec = None
            self.num_actions = action_space.n

    def _stack_obs(self, observation):
        if self._obs_is_tuple:
            # gymnasium.make_vec batches a Tuple(Discrete, ...) observation
            # space into a tuple of per-component arrays, each shape
            # (num_envs,) -- stack them into (num_envs, n_components).
            return np.stack(observation, axis=-1)
        return observation

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, PopGymState]:
        def _reset(key):
            observation, _ = self.environment.reset()
            observation = self._stack_obs(observation)
            observation = np.reshape(
                observation, self.batch_shape + self.observation_shape
            )
            return jnp.array(observation, dtype=self.observation_dtype)

        observation = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.observation_shape, self.observation_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PopGymState(step=0)
        return observation, state

    def step(
        self,
        key: Key,
        state: PopGymState,
        action: Array,
        params=None,
    ) -> tuple[Array, PopGymState, Array, Array, dict]:
        def _step(action):
            action = np.reshape(action, (-1,))
            action = np.asarray(action, dtype=np.int64)
            if self._action_nvec is not None:
                # Undo the Discrete(prod(nvec)) flattening: map each flat
                # index back to its per-component MultiDiscrete indices.
                action = np.stack(
                    np.unravel_index(action, self._action_nvec), axis=-1
                )
            observation, rewards, terminations, truncations, infos = (
                self.environment.step(action)
            )
            observation = self._stack_obs(observation)
            observation = np.reshape(
                observation, self.batch_shape + self.observation_shape
            )
            rewards = np.reshape(rewards, self.batch_shape)
            dones = np.reshape(terminations | truncations, self.batch_shape)
            return (
                jnp.array(observation, dtype=self.observation_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.bool_),
            )

        observation, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(self.observation_shape, self.observation_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            ),
            action,
            vmap_method="broadcast_all",
        )

        new_state = PopGymState(step=state.step + 1)
        return observation, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=self.observation_low,
            high=self.observation_high,
            shape=self.observation_shape,
            dtype=self.observation_dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)


def make(env_id, batch_shape: tuple[int, ...] = (1,), **kwargs) -> tuple:
    import gymnasium
    import popgym  # noqa: F401  # registers popgym envs with gymnasium

    num_envs = int(np.prod(batch_shape))
    environment = gymnasium.make_vec(env_id, num_envs=num_envs, **kwargs)
    return ExtendedPopGymWrapper(environment, batch_shape=batch_shape), None
