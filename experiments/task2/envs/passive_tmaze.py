"""Native JAX port of Memory-RL's Passive T-Maze."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class PassiveTMazeParams:
    goal_reward: float = 1.0
    penalty: float = -0.1


@struct.dataclass
class PassiveTMazeState:
    x: jax.Array
    y: jax.Array
    goal_y: jax.Array
    time_step: jax.Array


class PassiveTMaze(
    environment.Environment[PassiveTMazeState, PassiveTMazeParams]
):
    """Passive T-Maze with a one-step oracle and a fixed episode horizon.

    Actions follow the reference implementation:
      0 = right, 1 = up, 2 = left, 3 = down.
    """

    def __init__(self, corridor_length: int, expose_goal: bool = False):
        if corridor_length < 1:
            raise ValueError("corridor_length must be at least 1")
        self.corridor_length = int(corridor_length)
        self.expose_goal = bool(expose_goal)

    @property
    def default_params(self) -> PassiveTMazeParams:
        return PassiveTMazeParams(penalty=-1.0 / self.corridor_length)

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params=None):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=((3,) if self.expose_goal else (2,)),
            dtype=jnp.float32,
        )

    def state_space(self, params=None):
        return spaces.Dict(
            {
                "x": spaces.Discrete(self.corridor_length + 1),
                "y": spaces.Box(-1, 1, shape=(), dtype=jnp.int32),
                "goal_y": spaces.Box(-1, 1, shape=(), dtype=jnp.int32),
                "time_step": spaces.Discrete(self.corridor_length + 2),
            }
        )

    def reset_env(self, key, params):
        goal_y = jax.random.choice(
            key, jnp.asarray([-1, 1], dtype=jnp.int32)
        )
        state = PassiveTMazeState(
            x=jnp.asarray(0, dtype=jnp.int32),
            y=jnp.asarray(0, dtype=jnp.int32),
            goal_y=goal_y,
            time_step=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state, params), state

    def get_obs(self, state, params=None, key=None):
        at_oracle = state.time_step == 0
        at_junction_or_goal = state.x >= self.corridor_length
        position = jnp.where(at_junction_or_goal, 1.0, 0.0)
        memory_signal = jnp.where(
            at_oracle,
            state.goal_y.astype(jnp.float32),
            jnp.where(
                at_junction_or_goal,
                state.y.astype(jnp.float32),
                0.0,
            ),
        )
        observation = jnp.asarray(
            [position, memory_signal], dtype=jnp.float32
        )
        if self.expose_goal:
            observation = jnp.concatenate(
                [
                    observation,
                    jnp.asarray(
                        [state.goal_y], dtype=jnp.float32
                    ),
                ]
            )
        return observation

    def step_env(self, key, state, action, params):
        del key
        action = jnp.asarray(action, dtype=jnp.int32)

        move_right = (action == 0) & (state.y == 0)
        move_left = (action == 2) & (state.y == 0)
        move_up = (action == 1) & (state.x == self.corridor_length)
        move_down = (action == 3) & (state.x == self.corridor_length)

        next_x = jnp.where(
            move_right,
            jnp.minimum(state.x + 1, self.corridor_length),
            jnp.where(move_left, jnp.maximum(state.x - 1, 0), state.x),
        )
        next_y = jnp.where(
            move_up,
            1,
            jnp.where(move_down, -1, state.y),
        ).astype(jnp.int32)
        next_time_step = state.time_step + 1
        next_state = PassiveTMazeState(
            x=next_x.astype(jnp.int32),
            y=next_y,
            goal_y=state.goal_y,
            time_step=next_time_step.astype(jnp.int32),
        )

        done = next_time_step >= self.corridor_length + 1
        success = next_y == state.goal_y
        lagging = next_x < next_time_step
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
        return state.time_step >= self.corridor_length + 1
