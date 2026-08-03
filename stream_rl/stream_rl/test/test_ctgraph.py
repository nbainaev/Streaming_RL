import unittest

import jax
import numpy as np

from stream_rl.src.env.ctgraph import CTGraph
from stream_rl.src.env.factory import make_env


class CTGraphTests(unittest.TestCase):
    def test_optimal_path_gets_linear_reward_and_lifelong_clock_survives_reset(self):
        env = CTGraph(depth=2, branching=2, reward_switch_steps=64, reward_seed=0)
        params = env.default_params
        obs, state = env.reset(jax.random.key(0), params)
        self.assertEqual(obs.shape, (7,))

        # phase 0 target is [0, 0]: root, wait, first decision, wait,
        # second decision, final wait.
        done = False
        reward = 0.0
        info = {}
        for action in [0, 0, 1, 0, 1, 0]:
            obs, state, reward, done, info = env.step(jax.random.key(1), state, action, params)

        self.assertTrue(bool(done))
        self.assertAlmostEqual(float(reward), 1.0)
        self.assertTrue(bool(info["success"]))
        self.assertEqual(int(state.global_step), 6)
        self.assertEqual(int(state.episode_id), 1)

    def test_reward_path_switches_only_between_complete_attempts(self):
        env = CTGraph(depth=1, branching=2, reward_switch_steps=3, reward_seed=0)
        params = env.default_params
        _, state = env.reset(jax.random.key(0), params)
        switched = []
        phases = []
        rewards = []
        for action in [0, 0, 1, 0, 0]:
            _, state, reward, _, info = env.step(jax.random.key(2), state, action, params)
            switched.append(bool(info["reward_switched"]))
            phases.append(int(info["reward_phase"]))
            rewards.append(float(reward))
        # The threshold is crossed on step 3, but phase 0 scores the complete
        # route and phase 1 becomes active only after its terminal wait.
        np.testing.assert_array_equal(switched, [False, False, False, True, False])
        np.testing.assert_array_equal(phases, [0, 0, 0, 0, 1])
        self.assertAlmostEqual(rewards[3], 1.0)

    def test_continuing_mode_preserves_agent_sequence_but_logs_attempts(self):
        env, params = make_env(
            {
                "namespace": "ctgraph",
                "env_id": "ctgraph_lifelong",
                "kwargs": {
                    "depth": 1,
                    "branching": 2,
                    "reward_switch_steps": 128,
                    "continuing_task": True,
                },
            },
            num_envs=1,
        )
        _, state = env.reset(jax.random.key(0), params)
        for action in [0, 0, 1, 0]:
            _, state, reward, done, info = env.step(
                jax.random.key(3), state, action, params
            )
        self.assertFalse(bool(done))
        self.assertTrue(bool(info["episode_boundary"]))
        self.assertTrue(bool(info["returned_episode"]))
        self.assertAlmostEqual(float(info["returned_episode_returns"]), 1.0)
        self.assertAlmostEqual(float(reward), 1.0)
        self.assertEqual(int(state.env_state.episode_id), 1)


if __name__ == "__main__":
    unittest.main()
