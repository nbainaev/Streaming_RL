import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import unittest

from stream_rl.src.models.blocks import FrozenSSMMemoryBlock, build_block
from stream_rl.src.models.networks import FrozenProjectionEncoder, build_actor_network


class FrozenSSMMemoryTest(unittest.TestCase):
    def test_frozen_ssm_has_no_parameters_and_resets_on_done(self):
        block = FrozenSSMMemoryBlock(features=8, state_dim=16, seed=7)
        x = jnp.ones((2, 4, 3), dtype=jnp.float32)
        done = jnp.asarray([[True, False, True, False], [True, False, False, False]])
        carry = block.initialize_carry(jax.random.key(0), (2, None))
        variables = block.init(jax.random.key(1), x, done=done, initial_carry=carry)

        self.assertNotIn("params", variables)
        next_carry, output = block.apply(
            variables, x, done=done, initial_carry=carry
        )
        self.assertEqual(next_carry.shape, (2, 1, 16))
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertTrue(np.isfinite(np.asarray(output)).all())

        # A reset at t=2 makes that suffix independent of inputs at t<2.
        changed = x.at[0, :2].set(100.0)
        _, changed_output = block.apply(
            variables, changed, done=done, initial_carry=carry
        )
        np.testing.assert_allclose(output[0, 2:], changed_output[0, 2:], atol=1e-6)

    def test_readout_only_network_contains_only_final_projection_params(self):
        architecture = [
            {"type": "frozen_ssm", "features": 8, "state_dim": 16, "seed": 3},
            {"type": "fc", "features": -1},
        ]
        network = build_actor_network(
            architecture_cfg=architecture,
            action_dim=4,
            embed_dim=8,
            readout_only=True,
            memory_seed=9,
        )
        observation = jnp.ones((2, 3, 2), dtype=jnp.float32)
        action = jnp.zeros((2, 3), dtype=jnp.int32)
        reward = jnp.zeros((2, 3), dtype=jnp.float32)
        done = jnp.zeros((2, 3), dtype=jnp.bool_)
        carry = network.initialize_carry((2, None))
        variables = network.init(
            jax.random.key(0), observation, done, action, reward, initial_carry=carry
        )

        leaves = jax.tree.leaves(variables["params"])
        # Final Dense kernel and bias only; encoder and SSM have no parameters.
        self.assertEqual(len(leaves), 2)
        self.assertEqual(sum(x.size for x in leaves), 8 * 4 + 4)

    def test_readout_only_rejects_hidden_trainable_layers(self):
        with self.assertRaisesRegex(ValueError, "readout_only"):
            build_actor_network(
                architecture_cfg=[
                    {"type": "frozen_ssm", "features": 8},
                    {"type": "layernorm"},
                    {"type": "fc", "features": -1},
                ],
                action_dim=4,
                embed_dim=8,
                readout_only=True,
            )

    def test_auxiliary_readout_shares_only_downstream_parameters(self):
        network = build_actor_network(
            architecture_cfg=[
                {"type": "frozen_ssm", "features": 8, "state_dim": 16, "seed": 3}
            ],
            action_dim=4,
            embed_dim=8,
            readout_only=True,
            memory_seed=9,
            auxiliary_cue=True,
            auxiliary_readout_dim=5,
        )
        observation = jnp.ones((2, 3, 2), dtype=jnp.float32)
        action = jnp.zeros((2, 3), dtype=jnp.int32)
        reward = jnp.zeros((2, 3), dtype=jnp.float32)
        done = jnp.zeros((2, 3), dtype=jnp.bool_)
        carry = network.initialize_carry((2, None))
        variables = network.init(
            jax.random.key(0), observation, done, action, reward, initial_carry=carry
        )
        _, (distribution, aux) = network.apply(
            variables, observation, done, action, reward, initial_carry=carry
        )

        leaves = jax.tree.leaves(variables["params"])
        self.assertEqual(sum(x.size for x in leaves), 75)
        self.assertEqual(distribution.logits.shape, (2, 3, 4))
        self.assertEqual(aux["cue_prediction"].shape, (2, 3))

    def test_memoryless_readout_only_control_is_allowed(self):
        network = build_actor_network(
            architecture_cfg=[{"type": "fc", "features": -1}],
            action_dim=4,
            embed_dim=8,
            readout_only=True,
            memory_seed=9,
        )
        observation = jnp.ones((2, 3, 2), dtype=jnp.float32)
        action = jnp.zeros((2, 3), dtype=jnp.int32)
        reward = jnp.zeros((2, 3), dtype=jnp.float32)
        done = jnp.zeros((2, 3), dtype=jnp.bool_)
        carry = network.initialize_carry((2, None))
        variables = network.init(
            jax.random.key(0), observation, done, action, reward, initial_carry=carry
        )
        self.assertEqual(sum(x.size for x in jax.tree.leaves(variables["params"])), 36)

    def test_external_pretrained_npz_is_loaded_as_frozen_constants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/memory.npz"
            np.savez(
                path,
                decay=np.asarray([0.8, 0.9, 0.95, 0.99], dtype=np.float32),
                B=np.ones((4, 3), dtype=np.float32),
                C=np.ones((2, 4), dtype=np.float32),
                D=np.zeros((2, 3), dtype=np.float32),
            )
            block = build_block(
                {
                    "type": "frozen_ssm",
                    "features": 2,
                    "state_dim": 4,
                    "checkpoint_path": path,
                },
                resolved_features=2,
            )
            x = jnp.ones((1, 2, 3), dtype=jnp.float32)
            variables = block.init(jax.random.key(0), x)
            self.assertNotIn("params", variables)
            _, output = block.apply(variables, x)
            self.assertEqual(output.shape, (1, 2, 2))

    def test_frozen_ssm_can_preserve_current_input_as_a_fixed_skip(self):
        block = FrozenSSMMemoryBlock(
            features=8, state_dim=16, seed=7, concatenate_input=True
        )
        x = jnp.ones((2, 4, 3), dtype=jnp.float32)
        variables = block.init(jax.random.key(0), x)
        self.assertNotIn("params", variables)
        _, output = block.apply(variables, x)
        self.assertEqual(output.shape, (2, 4, 11))
        np.testing.assert_allclose(np.asarray(output[..., :3]), np.asarray(x))

    def test_frozen_encoder_can_preserve_observed_phase_channel(self):
        encoder = FrozenProjectionEncoder(
            embed_dim=8, seed=9, preserve_observation_prefix=1
        )
        observation = jnp.asarray(
            [[[0.0, -1.0], [1.0, 0.0]]], dtype=jnp.float32
        )
        variables = encoder.init(jax.random.key(0), observation)
        output = encoder.apply(variables, observation)
        self.assertNotIn("params", variables)
        np.testing.assert_allclose(output[..., 0], observation[..., 0])

    def test_phase_gated_memory_network_trains_only_two_readouts(self):
        network = build_actor_network(
            architecture_cfg=[
                {
                    "type": "frozen_ssm",
                    "features": 8,
                    "state_dim": 16,
                    "seed": 3,
                    "concatenate_input": True,
                },
                {
                    "type": "gated_readout",
                    "features": -1,
                    "split_index": 8,
                    "fixed_gate_index": 0,
                    "navigation_prefix": 1,
                },
            ],
            action_dim=4,
            embed_dim=8,
            readout_only=True,
            memory_seed=9,
            preserve_observation_prefix=1,
        )
        observation = jnp.asarray(
            [[[0.0, -1.0], [1.0, 0.0]]], dtype=jnp.float32
        )
        action = jnp.zeros((1, 2), dtype=jnp.int32)
        reward = jnp.zeros((1, 2), dtype=jnp.float32)
        done = jnp.zeros((1, 2), dtype=jnp.bool_)
        carry = network.initialize_carry((1, None))
        variables = network.init(
            jax.random.key(0),
            observation,
            done,
            action,
            reward,
            initial_carry=carry,
        )
        # Navigation: 1x4 kernel + bias; memory: 8x4 kernel + bias.
        self.assertEqual(
            sum(x.size for x in jax.tree.leaves(variables["params"])), 44
        )


if __name__ == "__main__":
    unittest.main()
