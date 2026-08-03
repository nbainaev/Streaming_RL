import unittest

import jax
import jax.numpy as jnp
import lox

from stream_rl.src.utils.lox_compat import patch_lox_scan_metadata


class LoxCompatibilityTests(unittest.TestCase):
    def test_spooling_logs_from_scan(self):
        patch_lox_scan_metadata()

        def body(carry, x):
            value = carry + x
            lox.log({"value": value})
            return value, None

        def run(initial):
            result, _ = jax.lax.scan(body, initial, jnp.arange(4.0))
            return result

        result, logs = lox.spool(run)(jnp.asarray(0.0))
        self.assertAlmostEqual(float(result), 6.0)
        self.assertEqual(logs["value"].tolist(), [0.0, 1.0, 3.0, 6.0])


if __name__ == "__main__":
    unittest.main()
