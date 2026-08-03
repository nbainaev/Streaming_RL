"""Pure-JAX CT-graph environment for non-stationary lifelong RL tests.

The transition structure follows soltoggio/CT-graph: root -> wait ->
decision, with action 0 used to leave wait states and actions 1..B used at
decision states.  The rewarding leaf changes on a fixed environment-step
schedule and is deliberately not part of the observation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


ROOT, WAIT, DECISION, END = 0, 1, 2, 3


@struct.dataclass
class CTGraphParams:
    high_reward: float = 1.0
    fail_reward: float = -1.0


@struct.dataclass
class CTGraphState:
    stage: jax.Array
    depth_index: jax.Array
    recorded_path: jax.Array
    global_step: jax.Array
    episode_id: jax.Array
    reward_phase: jax.Array


class CTGraph(environment.Environment[CTGraphState, CTGraphParams]):
    """Small vector-observation CT-graph with a changing hidden reward leaf."""

    def __init__(
        self,
        depth: int = 2,
        branching: int = 2,
        reward_switch_steps: int = 128,
        reward_distribution: str = "linear",
        reward_seed: int = 0,
        continuing_task: bool = False,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if branching < 2:
            raise ValueError("branching must be >= 2")
        if reward_switch_steps < 1:
            raise ValueError("reward_switch_steps must be >= 1")
        if reward_distribution not in {"linear", "needle_in_haystack"}:
            raise ValueError("reward_distribution must be 'linear' or 'needle_in_haystack'")
        self.depth = int(depth)
        self.branching = int(branching)
        self.reward_switch_steps = int(reward_switch_steps)
        self.reward_distribution = reward_distribution
        self.reward_seed = int(reward_seed)
        self.continuing_task = bool(continuing_task)

    @property
    def default_params(self) -> CTGraphParams:
        return CTGraphParams()

    @property
    def num_actions(self) -> int:
        return self.branching + 1

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params=None):
        # State type plus progress through the tree.  The rewarding path is
        # hidden, making changes detectable only through received reward.
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4 + self.depth + 1,),
            dtype=jnp.float32,
        )

    def state_space(self, params=None):
        return spaces.Dict(
            {
                "stage": spaces.Discrete(4),
                "depth_index": spaces.Discrete(self.depth + 1),
                "recorded_path": spaces.Box(
                    low=-1, high=self.branching - 1,
                    shape=(self.depth,), dtype=jnp.int32,
                ),
                "global_step": spaces.Discrete(2**31 - 1),
                "episode_id": spaces.Discrete(2**31 - 1),
                "reward_phase": spaces.Discrete(2**31 - 1),
            }
        )

    def _observation(self, state: CTGraphState) -> jax.Array:
        return jnp.concatenate(
            [
                jax.nn.one_hot(state.stage, 4, dtype=jnp.float32),
                jax.nn.one_hot(state.depth_index, self.depth + 1, dtype=jnp.float32),
            ]
        )

    def _reward_path(self, phase: jax.Array) -> jax.Array:
        """Enumerate leaves cyclically so every scheduled switch really moves."""
        num_paths = self.branching ** self.depth
        code = jnp.mod(phase + self.reward_seed, num_paths)
        powers = self.branching ** jnp.arange(self.depth - 1, -1, -1)
        return (code // powers % self.branching).astype(jnp.int32)

    def reset_env(self, key, params):
        del key, params
        state = CTGraphState(
            stage=jnp.asarray(ROOT, dtype=jnp.int32),
            depth_index=jnp.asarray(0, dtype=jnp.int32),
            recorded_path=jnp.full((self.depth,), -1, dtype=jnp.int32),
            global_step=jnp.asarray(0, dtype=jnp.int32),
            episode_id=jnp.asarray(0, dtype=jnp.int32),
            reward_phase=jnp.asarray(0, dtype=jnp.int32),
        )
        return self._observation(state), state

    def get_obs(self, state, params=None, key=None):
        del params, key
        return self._observation(state)

    def _path_score(self, recorded_path, reward_path):
        similarity = 1.0 - (
            jnp.abs(recorded_path - reward_path).astype(jnp.float32)
            / float(self.branching - 1)
        )
        if self.reward_distribution == "needle_in_haystack":
            return jnp.all(recorded_path == reward_path).astype(jnp.float32)
        # Earlier decisions carry more weight, matching CT-graph's linear
        # distribution while keeping the score in [0, 1] for B > 2.
        weights = jnp.arange(self.depth, 0, -1, dtype=jnp.float32)
        return jnp.sum(weights * similarity) / jnp.sum(weights)

    def step_env(self, key, state, action, params):
        del key
        action = jnp.asarray(action, dtype=jnp.int32)
        next_global_step = state.global_step + 1
        scheduled_phase = next_global_step // self.reward_switch_steps
        # Keep the target fixed for the complete root-to-leaf attempt.  A
        # step-based schedule may cross its threshold in the middle of an
        # episode; applying that new target only at the next root avoids
        # scoring a path against a goal that changed while it was traversed.
        reward_path = self._reward_path(state.reward_phase)

        is_root = state.stage == ROOT
        is_wait = state.stage == WAIT
        is_decision = state.stage == DECISION
        valid_wait = action == 0
        valid_decision = action > 0

        crash = (is_wait & ~valid_wait) | (is_decision & ~valid_decision)
        at_last_wait = is_wait & valid_wait & (state.depth_index >= self.depth)

        chosen_branch = jnp.clip(action - 1, 0, self.branching - 1)
        write_mask = jax.nn.one_hot(
            jnp.minimum(state.depth_index, self.depth - 1), self.depth, dtype=jnp.int32
        )
        updated_path = jnp.where(
            is_decision & valid_decision,
            state.recorded_path * (1 - write_mask) + chosen_branch * write_mask,
            state.recorded_path,
        )
        next_depth = state.depth_index + (is_decision & valid_decision).astype(jnp.int32)

        next_stage = jnp.where(is_root, WAIT, state.stage)
        next_stage = jnp.where(is_wait & valid_wait & (state.depth_index < self.depth), DECISION, next_stage)
        next_stage = jnp.where(is_decision & valid_decision, WAIT, next_stage)
        next_stage = jnp.where(at_last_wait, END, next_stage)

        path_score = self._path_score(updated_path, reward_path)
        reward = jnp.where(
            crash,
            params.fail_reward,
            jnp.where(at_last_wait, params.high_reward * path_score, 0.0),
        ).astype(jnp.float32)
        # Terminate on the rewarding leaf transition.  The upstream
        # CT-graph adds an extra action-independent END transition; removing
        # that no-information step keeps returns identical and makes the
        # terminal success/path score available to generic episode logging.
        done = crash | at_last_wait
        reward_switched = done & (scheduled_phase != state.reward_phase)

        next_state = CTGraphState(
            stage=next_stage.astype(jnp.int32),
            depth_index=next_depth.astype(jnp.int32),
            recorded_path=updated_path.astype(jnp.int32),
            global_step=next_global_step.astype(jnp.int32),
            episode_id=state.episode_id,
            reward_phase=state.reward_phase,
        )
        info = {
            "success": at_last_wait & (path_score >= 1.0 - 1e-6),
            "path_score": jnp.where(at_last_wait, path_score, 0.0),
            "reward_phase": state.reward_phase,
            "next_reward_phase": jnp.where(done, scheduled_phase, state.reward_phase),
            "reward_switched": reward_switched,
            "episode_boundary": done,
        }
        return self._observation(next_state), next_state, reward, done, info

    def step(self, key, state, action, params=None):
        """Auto-reset episodic fields without erasing the lifelong clock."""
        if params is None:
            params = self.default_params
        obs, next_state, reward, done, info = self.step_env(key, state, action, params)
        reset_state = CTGraphState(
            stage=jnp.asarray(ROOT, dtype=jnp.int32),
            depth_index=jnp.asarray(0, dtype=jnp.int32),
            recorded_path=jnp.full((self.depth,), -1, dtype=jnp.int32),
            global_step=next_state.global_step,
            episode_id=state.episode_id + done.astype(jnp.int32),
            reward_phase=jnp.where(
                done,
                next_state.global_step // self.reward_switch_steps,
                next_state.reward_phase,
            ).astype(jnp.int32),
        )
        selected_state = jax.tree.map(lambda new, reset: jnp.where(done, reset, new), next_state, reset_state)
        selected_obs = jnp.where(done, self._observation(reset_state), obs)
        # Lifelong experiments are one continuing process.  The environment
        # still resets the route at every leaf/crash, while the false agent
        # termination preserves recurrent state, eligibility traces, the
        # terminal reward input, and value bootstrapping across attempts.
        agent_done = jnp.asarray(False) if self.continuing_task else done
        return selected_obs, selected_state, reward, agent_done, info

    def is_terminal(self, state, params):
        del params
        return state.stage == END
