import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from stream_rl.src.agents.factory import build_agent
from stream_rl.src.env.factory import make_env
from stream_rl.src.models.eprop_layers import EpropCarry, GRU, LSTM, VanillaRNN


class EpropTraceTests(unittest.TestCase):
    def test_lstm_carry_checkpoint_roundtrip(self):
        carry = EpropCarry(
            h=(jnp.asarray([[1.0, 2.0]]), jnp.asarray([[3.0, 4.0]])),
            traces={"e_kernel": jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3)},
        )
        restored = serialization.from_bytes(carry, serialization.to_bytes(carry))
        for expected, actual in zip(jax.tree.leaves(carry), jax.tree.leaves(restored)):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def _assert_instantaneous_traces_match_autodiff(self, cell, hidden):
        batch, input_size, hidden_size = 2, 3, 2
        x = jnp.asarray([[0.2, -0.4, 0.7], [-0.3, 0.6, 0.1]], dtype=jnp.float32)
        carry = cell.initialize_carry(None, (batch, input_size))
        if isinstance(cell, LSTM):
            c = jnp.asarray([[0.15, -0.2], [0.3, 0.1]], dtype=jnp.float32)
            carry = EpropCarry(h=(c, hidden), traces=carry.traces)
        else:
            carry = EpropCarry(h=hidden, traces=carry.traces)

        variables = cell.init(jax.random.key(4), x, initial_carry=carry)
        new_carry, _ = cell.apply(variables, x, initial_carry=carry)

        def output_from_params(params):
            _, output = cell.apply({"params": params}, x, initial_carry=carry)
            return output

        jac = jax.jacobian(output_from_params)(variables["params"])
        for parameter_name, parameter in variables["params"].items():
            trace_name = "e_" + parameter_name
            self.assertIn(trace_name, new_carry.traces)
            derivative = np.asarray(jac[parameter_name])
            expected = np.zeros((batch, *parameter.shape), dtype=np.float32)
            if parameter.ndim == 2:
                for b in range(batch):
                    for out_idx in range(hidden_size):
                        expected[b, :, out_idx] = derivative[b, out_idx, :, out_idx]
            else:
                for b in range(batch):
                    for out_idx in range(hidden_size):
                        expected[b, out_idx] = derivative[b, out_idx, out_idx]
            np.testing.assert_allclose(
                np.asarray(new_carry.traces[trace_name]), expected, rtol=2e-5, atol=2e-6,
                err_msg=f"incorrect instantaneous trace for {type(cell).__name__}.{parameter_name}",
            )

    def test_vanilla_rnn_traces(self):
        hidden = jnp.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32)
        self._assert_instantaneous_traces_match_autodiff(
            VanillaRNN(hidden_size=2, trace_decay=0.0, use_layernorm=False, use_sparse_init=False), hidden
        )

    def test_gru_traces(self):
        hidden = jnp.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32)
        self._assert_instantaneous_traces_match_autodiff(
            GRU(hidden_size=2, trace_decay=0.0, use_layernorm=False, use_sparse_init=False), hidden
        )

    def test_lstm_traces(self):
        hidden = jnp.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32)
        self._assert_instantaneous_traces_match_autodiff(
            LSTM(hidden_size=2, trace_decay=0.0, use_layernorm=False, use_sparse_init=False), hidden
        )


class StreamEpropStateTests(unittest.TestCase):
    def _build_agent(self, feedback_seed=17, **overrides):
        env, params = make_env(
            {"namespace": "tmaze", "env_id": "tmaze_passive", "kwargs": {"corridor_length": 2}},
            num_envs=2,
        )
        cfg = {
            "name": "stream_eprop", "num_envs": 2, "gamma": 0.99,
            "embed_dim": 4, "hidden_size": 4, "cell": "eprop_gru", "activation": "tanh",
            "trace_decay": 0.9, "trace_lambda": 0.9, "actor_lr": 1.0, "critic_lr": 1.0,
            "actor_kappa": 0.2, "critic_kappa": 0.5, "entropy_coefficient": 0.01,
            "adaptive": False, "feedback_mode": "random", "feedback_seed": feedback_seed,
            "feedback_lr": 0.05, "use_layernorm": True, "use_sparse_init": True, "sparsity": 0.9,
        }
        cfg.update(overrides)
        return build_agent("stream_eprop", cfg, env, params)

    def test_feedback_seed_is_independent_of_training_seed(self):
        agent = self._build_agent(feedback_seed=23)
        state_a = agent.init(jax.random.key(0))
        state_b = agent.init(jax.random.key(999))
        np.testing.assert_array_equal(state_a.actor_feedback, state_b.actor_feedback)
        np.testing.assert_array_equal(state_a.critic_feedback, state_b.critic_feedback)

    def test_done_resets_hidden_state_and_local_traces_per_env(self):
        agent = self._build_agent()
        carry = agent.actor_cells[0].initialize_carry(None, (2, 4))
        carry = jax.tree.map(jnp.ones_like, carry)
        reset = agent._reset_carry(carry, jnp.asarray([True, False]))
        for leaf in jax.tree.leaves(reset):
            np.testing.assert_array_equal(np.asarray(leaf[0]), np.zeros_like(np.asarray(leaf[0])))
            np.testing.assert_array_equal(np.asarray(leaf[1]), np.ones_like(np.asarray(leaf[1])))

    def test_nonlinear_head_has_valid_effective_feedback_target(self):
        agent = self._build_agent(
            feedback_mode="adaptive",
            head_hidden_sizes=[8, 6],
            head_activation="tanh",
        )
        state = agent.init(jax.random.key(3))
        h = jnp.asarray(
            [[0.2, -0.1, 0.4, 0.3], [-0.2, 0.5, 0.1, -0.4]],
            dtype=jnp.float32,
        )
        target = agent._effective_feedback_target(
            agent.actor_head, state.actor_params["head"], h, is_actor=True
        )
        self.assertEqual(target.shape, state.actor_feedback[0].shape)
        self.assertTrue(np.isfinite(np.asarray(target)).all())

    def test_symmetric_mode_uses_accumulated_cell_trace(self):
        agent = self._build_agent(
            feedback_mode="symmetric", use_layernorm=False, use_sparse_init=False
        )
        state = agent.init(jax.random.key(9))
        obs = state.timestep.obs
        carry = agent._reset_carry(state.actor_carry, state.timestep.done)
        x = agent.actor_embed.apply(state.actor_params["embed"], obs)
        new_carry, h = agent._apply_stack(
            agent.actor_cells, state.actor_params["cells"], carry, x
        )
        action = jnp.zeros((2,), dtype=jnp.int32)
        td_error = jnp.ones((2,), dtype=jnp.float32)

        grads, feedback = agent._role_grads(
            is_actor=True,
            embed=agent.actor_embed,
            cells=agent.actor_cells,
            head=agent.actor_head,
            params=state.actor_params,
            obs=obs,
            carry_in=carry,
            new_carry=new_carry,
            x=x,
            h=h,
            action=action,
            td_error=td_error,
            feedback=state.actor_feedback,
        )
        expected_feedback = agent._effective_feedback_target(
            agent.actor_head, state.actor_params["head"], h, is_actor=True
        )
        np.testing.assert_allclose(feedback[0], expected_feedback, rtol=1e-6, atol=1e-6)

        traced_grad_norm = sum(
            jnp.linalg.norm(grads["cells"][0]["params"][name])
            for name in ("wz", "wr", "wn")
        )
        self.assertGreater(float(traced_grad_norm), 0.0)


if __name__ == "__main__":
    unittest.main()
