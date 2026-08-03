import unittest

import jax
import jax.numpy as jnp

from stream_rl.src.env.delayed_cue import DelayedCue


class DelayedCueTests(unittest.TestCase):
    def test_cue_is_hidden_during_delay_and_rewarded_at_decision(self):
        env = DelayedCue(delay=2)
        params = env.default_params
        obs, state = env.reset(jax.random.key(0), params)
        cue = int(state.cue)
        self.assertNotEqual(float(obs[0]), 0.0)
        for _ in range(3):
            obs, state, reward, done, _ = env.step(jax.random.key(1), state, 1 - cue, params)
            if not done:
                self.assertEqual(float(obs[0]), 0.0)
        self.assertFalse(bool(done))
        obs, state, reward, done, info = env.step(jax.random.key(1), state, cue, params)
        self.assertTrue(bool(done))
        self.assertEqual(float(reward), 1.0)
        self.assertTrue(bool(info["success"]))

    def test_dense_variant_rewards_memory_during_blank_observations(self):
        env = DelayedCue(delay=2, dense_rewards=True)
        params = env.default_params
        _, state = env.reset(jax.random.key(0), params)
        cue = int(state.cue)
        _, state, reward, done, _ = env.step(jax.random.key(1), state, cue, params)
        self.assertEqual(float(reward), 0.0)
        obs, state, reward, done, _ = env.step(jax.random.key(1), state, cue, params)
        self.assertEqual(float(obs[0]), 0.0)
        self.assertEqual(float(reward), 1.0)

    def test_noise_is_a_nuisance_not_part_of_ground_truth_state(self):
        env = DelayedCue(delay=2, noise_dim=4, noise_std=0.5)
        params = env.default_params
        obs, state = env.reset(jax.random.key(0), params)
        self.assertEqual(obs.shape, (6,))
        obs_a = env.get_obs(state, key=jax.random.key(2))
        obs_b = env.get_obs(state, key=jax.random.key(3))
        self.assertEqual(float(obs_a[0]), float(obs_b[0]))
        self.assertFalse(bool(jnp.allclose(obs_a[2:], obs_b[2:])))


if __name__ == "__main__":
    unittest.main()
