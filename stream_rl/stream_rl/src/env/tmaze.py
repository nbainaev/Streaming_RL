"""JAX/gymnax T-Maze (Passive + Active), ported from twni2016/Memory-RL's
``envs/tmaze.py`` (``TMazeBase`` / ``TMazeClassicPassive`` / ``TMazeClassicActive``).

Reproduces its reward exactly:

    r_t = (1[x_{t+1} >= t - oracle_length] - 1) / (T - 1),  t <= T - 1
    r_T = 1[o_{T+1} = G]

which for the Passive case (``oracle_length=0``) is the Passive T-Maze
M_passive (O=S, L=T-1) reward: the per-step penalty defaults to
``-1/(T-1)`` and the terminal reward is the goal-hit indicator.

This is a self-contained copy of ``experiments/task2/envs/tmaze_base.py`` for
the streaming pipeline (kept independent so `stream_rl` and the `experiments`
package don't need to depend on each other's dependency set) — keep the two
in sync if the reward/observation logic changes.

Actions: 0=right, 1=up, 2=left, 3=down.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class TMazeParams:
    goal_reward: float = 1.0
    penalty: float = 0.0


@struct.dataclass
class TMazeState:
    x: jax.Array
    y: jax.Array
    goal_y: jax.Array
    oracle_visited: jax.Array
    time_step: jax.Array


class TMazeBase(environment.Environment[TMazeState, TMazeParams]):
    """T-Maze with a configurable oracle offset.

    ``oracle_length=0`` gives the Passive T-Maze: the goal is shown for free
    in the very first observation (the agent starts standing on the oracle
    cell). ``oracle_length=1`` gives the Active T-Maze (Bakker, 2001): the
    agent starts one cell past the oracle and must actively move left onto
    it to see the goal before the corridor pace-check penalty kicks in.
    """

    def __init__(
        self,
        corridor_length: int,
        oracle_length: int = 0,
        expose_goal: bool = False,
    ):
        if corridor_length < 1:
            raise ValueError("corridor_length must be at least 1")
        if oracle_length < 0:
            raise ValueError("oracle_length must be >= 0")
        self.corridor_length = int(corridor_length)
        self.oracle_length = int(oracle_length)
        self.expose_goal = bool(expose_goal)
        self.junction_x = self.oracle_length + self.corridor_length
        self.episode_length = self.corridor_length + 2 * self.oracle_length + 1

    @property
    def default_params(self) -> TMazeParams:
        return TMazeParams(penalty=-1.0 / (self.episode_length - 1))

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params=None):
        obs_dim = 3 if self.expose_goal else 2
        return spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=jnp.float32)

    def state_space(self, params=None):
        return spaces.Dict(
            {
                "x": spaces.Discrete(self.junction_x + 1),
                "y": spaces.Box(-1, 1, shape=(), dtype=jnp.int32),
                "goal_y": spaces.Box(-1, 1, shape=(), dtype=jnp.int32),
                "time_step": spaces.Discrete(self.episode_length + 1),
            }
        )

    def reset_env(self, key, params):
        goal_y = jax.random.choice(key, jnp.asarray([-1, 1], dtype=jnp.int32))
        state = TMazeState(
            x=jnp.asarray(self.oracle_length, dtype=jnp.int32),
            y=jnp.asarray(0, dtype=jnp.int32),
            goal_y=goal_y,
            oracle_visited=jnp.asarray(False),
            time_step=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def get_obs(self, state, params=None, key=None):
        at_oracle = (state.x == 0) & (~state.oracle_visited)
        at_junction_or_goal = state.x >= self.junction_x
        position = jnp.where(at_junction_or_goal, 1.0, 0.0)
        memory_signal = jnp.where(
            at_oracle,
            state.goal_y.astype(jnp.float32),
            jnp.where(at_junction_or_goal, state.y.astype(jnp.float32), 0.0),
        )
        observation = jnp.asarray([position, memory_signal], dtype=jnp.float32)
        if self.expose_goal:
            observation = jnp.concatenate(
                [observation, jnp.asarray([state.goal_y], dtype=jnp.float32)]
            )
        return observation

    def _is_valid_position(self, x: jax.Array, y: jax.Array) -> jax.Array:
        on_corridor = (y == 0) & (x >= 0) & (x <= self.junction_x)
        on_goal = (x == self.junction_x) & ((y == -1) | (y == 1))
        return on_corridor | on_goal

    def step_env(self, key, state, action, params):
        del key
        action = jnp.asarray(action, dtype=jnp.int32)
        dx = jnp.array([1, 0, -1, 0], dtype=jnp.int32)[action]
        dy = jnp.array([0, 1, 0, -1], dtype=jnp.int32)[action]

        cand_x = state.x + dx
        cand_y = state.y + dy
        valid = self._is_valid_position(cand_x, cand_y)

        next_x = jnp.where(valid, cand_x, state.x).astype(jnp.int32)
        next_y = jnp.where(valid, cand_y, state.y).astype(jnp.int32)
        next_time_step = state.time_step + 1
        # Deferred flag update (matches twni2016's mutate-on-use semantics):
        # exposure at x==0 is decided against the *incoming* oracle_visited
        # value in get_obs, so the flag only flips one step after the visit.
        next_oracle_visited = state.oracle_visited | (state.x == 0)

        next_state = TMazeState(
            x=next_x,
            y=next_y,
            goal_y=state.goal_y,
            oracle_visited=next_oracle_visited,
            time_step=next_time_step,
        )

        done = next_time_step >= self.episode_length
        success = next_y == state.goal_y
        lagging = next_x < (next_time_step - self.oracle_length)
        reward = jnp.where(
            done,
            jnp.where(success, params.goal_reward, 0.0),
            jnp.where(lagging, params.penalty, 0.0),
        ).astype(jnp.float32)
        info = {
            "success": done & success,
            "goal_y": state.goal_y,
            "x": next_x,
            "y": next_y,
        }
        return self.get_obs(next_state, params), next_state, reward, done, info

    def is_terminal(self, state, params):
        del params
        return state.time_step >= self.episode_length


class TMazeClassicPassive(TMazeBase):
    def __init__(self, corridor_length: int = 10, expose_goal: bool = False):
        super().__init__(
            corridor_length=corridor_length,
            oracle_length=0,
            expose_goal=expose_goal,
        )


class TMazeClassicActive(TMazeBase):
    def __init__(self, corridor_length: int = 10, expose_goal: bool = False):
        super().__init__(
            corridor_length=corridor_length,
            oracle_length=1,
            expose_goal=expose_goal,
        )
