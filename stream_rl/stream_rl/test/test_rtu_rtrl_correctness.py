import unittest

import jax
import jax.numpy as jnp
import numpy as np

from memorax.networks.sequence_models.rnn import RNN
from memorax.networks.sequence_models.rtu import RTUCell, RTUConfig


class RTURTRLCorrectnessTests(unittest.TestCase):
    def test_single_layer_sensitivity_matches_full_bptt_jacobian(self):
        """The compact RTU sensitivities must equal BPTT's nonzero diagonal."""
        cell = RTUCell(RTUConfig(features=3, hidden_dim=4))
        model = RNN(cell=cell)
        inputs = jax.random.normal(jax.random.key(1), (1, 5, 3)) * 0.2
        done = jnp.zeros((1, 5), dtype=jnp.bool_)
        carry = model.initialize_carry(jax.random.key(2), (1, 3))
        sensitivity = model.initialize_sensitivity(jax.random.key(3), (1, 3))
        variables = model.init(jax.random.key(4), inputs, done, carry)
        _, _, rtrl_sensitivity = model.apply(
            variables,
            inputs,
            done,
            carry,
            sensitivity=sensitivity,
            method=model.local_jacobian,
        )

        params = variables["params"]
        for name, parameter in params["cell"].items():
            def final_state(candidate):
                cell_params = dict(params["cell"])
                cell_params[name] = candidate
                final, _ = model.apply(
                    {"params": {"cell": cell_params}}, inputs, done, carry
                )
                return jnp.stack([final.real[0], final.imaginary[0]], axis=0)

            bptt = jax.jacrev(final_state)(parameter)
            if parameter.ndim == 1:
                compact_bptt = jnp.einsum("ahh->ah", bptt)
            else:
                compact_bptt = jnp.einsum("ahhf->ahf", bptt)
            np.testing.assert_allclose(
                np.asarray(rtrl_sensitivity[name][0]),
                np.asarray(compact_bptt),
                rtol=2e-5,
                atol=2e-6,
                err_msg=f"RTRL sensitivity mismatch for {name}",
            )


if __name__ == "__main__":
    unittest.main()
