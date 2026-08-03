import unittest

import jax

from stream_rl.src.agents.factory import build_agent
from stream_rl.src.env.factory import make_env


class ArchitectureVariantTests(unittest.TestCase):
    def setUp(self):
        self.env, self.params = make_env(
            {
                "namespace": "tmaze",
                "env_id": "tmaze_active",
                "kwargs": {"corridor_length": 2},
            },
            num_envs=1,
        )

    def _architecture(self):
        return [
            {"type": "rnn", "cell": "gru", "features": 8},
            {"type": "layernorm"},
            {"type": "rnn", "cell": "gru", "features": 8},
            {"type": "layernorm"},
            {"type": "fc", "features": 8, "activation": "tanh"},
            {"type": "fc", "features": -1},
        ]

    def test_stream_ac_accepts_stacked_recurrence_and_mlp_head(self):
        cfg = {
            "name": "stream_ac",
            "num_envs": 1,
            "embed_dim": 8,
            "gamma": 0.99,
            "trace_lambda": 0.9,
            "actor_lr": 1.0,
            "critic_lr": 1.0,
            "actor_kappa": 0.2,
            "critic_kappa": 0.5,
            "entropy_coefficient": 0.01,
            "adaptive": False,
            "tbptt_steps": 1,
            "actor_architecture": self._architecture(),
            "critic_architecture": self._architecture(),
        }
        agent = build_agent("stream_ac", cfg, self.env, self.params)
        state = agent.init(jax.random.key(0))
        self.assertEqual(len(state.actor_carry), 6)
        self.assertIsNotNone(state.actor_carry[0])
        self.assertIsNotNone(state.actor_carry[2])

    def test_direct_entropy_stream_ac_runs_stacked_update(self):
        cfg = {
            "name": "stream_ac",
            "num_envs": 1,
            "embed_dim": 8,
            "gamma": 0.99,
            "trace_lambda": 0.95,
            "actor_lr": 0.03,
            "critic_lr": 0.03,
            "actor_kappa": 3.0,
            "critic_kappa": 2.0,
            "entropy_coefficient": 0.1,
            "adaptive": False,
            "direct_entropy_update": True,
            "tbptt_steps": 1,
            "actor_architecture": self._architecture(),
            "critic_architecture": self._architecture(),
        }
        agent = build_agent("stream_ac", cfg, self.env, self.params)
        state = agent.init(jax.random.key(7))
        trained = agent.train(jax.random.key(8), state, num_steps=8)
        self.assertEqual(int(trained.step), 8)

    def test_recurrent_ppo_num_envs_one_uses_single_minibatch(self):
        cfg = {
            "name": "ppo",
            "num_envs": 1,
            "num_steps": 16,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 1,
            "update_epochs": 2,
            "normalize_advantage": True,
            "clip_coefficient": 0.2,
            "clip_value_loss": True,
            "entropy_coefficient": 0.01,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "max_grad_norm": 0.5,
            "embed_dim": 8,
            "actor_architecture": self._architecture(),
            "critic_architecture": self._architecture(),
        }
        agent = build_agent("ppo", cfg, self.env, self.params)
        state = agent.init(jax.random.key(1))
        self.assertEqual(agent.cfg.num_envs, 1)
        self.assertEqual(agent.cfg.num_minibatches, 1)
        self.assertEqual(len(state.actor_carry), 6)

    def test_rtu_rtrl_stream_ac_runs_online_update(self):
        architecture = [
            {"type": "rnn", "cell": "rtu_rtrl", "features": 4},
            {"type": "layernorm"},
            {"type": "fc", "features": 8, "activation": "tanh"},
            {"type": "fc", "features": -1},
        ]
        cfg = {
            "name": "stream_ac",
            "num_envs": 1,
            "embed_dim": 8,
            "gamma": 0.99,
            "trace_lambda": 0.95,
            "actor_lr": 0.01,
            "critic_lr": 0.03,
            "actor_kappa": 3.0,
            "critic_kappa": 2.0,
            "entropy_coefficient": 0.001,
            "adaptive": False,
            "direct_entropy_update": True,
            "tbptt_steps": 1,
            "actor_architecture": architecture,
            "critic_architecture": architecture,
        }
        agent = build_agent("stream_ac", cfg, self.env, self.params)
        state = agent.init(jax.random.key(11))
        dynamics, sensitivity = state.actor_carry[0]
        self.assertEqual(dynamics.real.shape[-1], 4)
        self.assertIn("B_real", sensitivity)
        trained = agent.train(jax.random.key(12), state, num_steps=8)
        self.assertEqual(int(trained.step), 8)

    def test_delta_rule_memory_runs_one_step_online_update(self):
        architecture = [
            {
                "type": "rnn",
                "cell": "delta_rule",
                "features": 8,
                "key_dim": 4,
            },
            {"type": "fc", "features": -1},
        ]
        cfg = {
            "name": "stream_ac",
            "num_envs": 1,
            "embed_dim": 8,
            "gamma": 0.99,
            "trace_lambda": 0.9,
            "actor_lr": 0.03,
            "critic_lr": 0.03,
            "actor_kappa": 3.0,
            "critic_kappa": 2.0,
            "entropy_coefficient": 0.001,
            "adaptive": False,
            "direct_entropy_update": True,
            "tbptt_steps": 1,
            "actor_architecture": architecture,
            "critic_architecture": architecture,
        }
        agent = build_agent("stream_ac", cfg, self.env, self.params)
        state = agent.init(jax.random.key(21))
        self.assertEqual(state.actor_carry[0].shape[-2:], (4, 8))
        trained = agent.train(jax.random.key(22), state, num_steps=8)
        self.assertEqual(int(trained.step), 8)

    def test_windowed_rtu_tbptt_uses_full_five_step_sequence(self):
        architecture = [
            {"type": "rnn", "cell": "rtu_bptt", "features": 4},
            {"type": "fc", "features": -1},
        ]
        cfg = {
            "name": "stream_tbptt",
            "num_envs": 1,
            "embed_dim": 8,
            "gamma": 0.99,
            "trace_lambda": 0.9,
            "actor_lr": 0.03,
            "critic_lr": 0.03,
            "actor_kappa": 3.0,
            "critic_kappa": 2.0,
            "entropy_coefficient": 0.001,
            "adaptive": False,
            "tbptt_steps": 5,
            "actor_architecture": architecture,
            "critic_architecture": architecture,
        }
        agent = build_agent("stream_tbptt", cfg, self.env, self.params)
        state = agent.init(jax.random.key(31))
        trained = agent.train(jax.random.key(32), state, num_steps=10)
        self.assertEqual(int(trained.step), 10)
        self.assertEqual(int(trained.update_step), 10)


if __name__ == "__main__":
    unittest.main()
