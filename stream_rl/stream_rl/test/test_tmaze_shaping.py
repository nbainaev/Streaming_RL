import unittest

import jax

from stream_rl.src.env.factory import make_env


class TMazeShapingTests(unittest.TestCase):
    def test_oracle_bonus_does_not_expose_or_replace_hidden_goal(self):
        env, params = make_env(
            {
                "namespace": "tmaze",
                "env_id": "tmaze_active",
                "kwargs": {"corridor_length": 2, "oracle_reward": 0.25},
            },
            num_envs=1,
        )
        _, state = env.reset(jax.random.key(0), params)
        # Active T-maze starts one cell right of the oracle. Action 2 moves
        # left, earning only the navigation bonus; the hidden goal remains a
        # property of the environment state and is not appended to the obs.
        obs, _, reward, done, _ = env.step(
            jax.random.key(1), state, 2, params
        )
        self.assertFalse(bool(done))
        self.assertAlmostEqual(float(reward), 0.25)
        self.assertEqual(obs.shape, (2,))


if __name__ == "__main__":
    unittest.main()
