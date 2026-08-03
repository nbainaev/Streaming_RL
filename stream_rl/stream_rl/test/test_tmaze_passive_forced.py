import unittest

import jax

from stream_rl.src.env.tmaze import TMazePassiveForced


class TMazePassiveForcedTests(unittest.TestCase):
    def test_forced_traversal_leaves_only_delayed_binary_decision(self):
        corridor_length = 5
        env = TMazePassiveForced(corridor_length=corridor_length)
        params = env.default_params
        obs, state = env.reset_env(jax.random.PRNGKey(7), params)

        self.assertEqual(env.num_actions, 2)
        self.assertEqual(float(obs[0]), 0.0)
        self.assertEqual(int(obs[1]), int(state.goal_y))

        for expected_x in range(1, corridor_length + 1):
            obs, state, reward, done, _ = env.step_env(
                jax.random.PRNGKey(expected_x), state, expected_x % 2, params
            )
            self.assertEqual(int(state.x), expected_x)
            self.assertFalse(bool(done))
            self.assertEqual(float(reward), 0.0)

        self.assertEqual(float(obs[0]), 1.0)
        self.assertEqual(float(obs[1]), 0.0)
        correct_action = 1 if int(state.goal_y) == 1 else 0
        _, state, reward, done, info = env.step_env(
            jax.random.PRNGKey(99), state, correct_action, params
        )
        self.assertTrue(bool(done))
        self.assertTrue(bool(info["success"]))
        self.assertEqual(float(reward), 1.0)
        self.assertEqual(int(state.y), int(state.goal_y))


if __name__ == "__main__":
    unittest.main()
