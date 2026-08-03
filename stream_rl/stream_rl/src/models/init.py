"""Sparse weight initialization (Elsayed et al., 2024, "Streaming Deep RL
Finally Works") — a flax-compatible port of the legacy torch
`stream_rl.src.utils.sparse_init.sparse_init`.

Each output unit keeps only a `1 - sparsity` fraction of its incoming
(fan_in) connections nonzero; the rest are drawn from a fan_in-scaled
uniform/normal distribution and then zeroed out per-output-unit. This is one
of the two stabilizing ingredients (along with LayerNorm on hidden
activations, see `blocks.LayerNormBlock`) that the streaming-RL literature
uses to make single-sample online TD updates work, and that this repo's
active JAX pipeline was missing (it only exists in the legacy torch path).

Note the transposed convention versus the torch original: flax `Dense`
kernels are `(fan_in, fan_out)` (torch stores `(fan_out, fan_in)`), so
"zero a sparsity-fraction of each output unit's incoming weights" here means
zeroing along axis 0 per column, not per row.
"""
import math
from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array
Initializer = Callable[[Array, tuple, jnp.dtype], Array]


def sparse_init(sparsity: float = 0.9, dist: str = "uniform") -> Initializer:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity!r}")

    def init(key: Array, shape: tuple, dtype: jnp.dtype = jnp.float32) -> Array:
        if len(shape) != 2:
            raise ValueError(f"sparse_init only supports 2D kernels, got shape={shape}")
        fan_in, fan_out = shape
        num_zeros = math.ceil(sparsity * fan_in)

        val_key, perm_key = jax.random.split(key)
        bound = math.sqrt(1.0 / fan_in)
        if dist == "uniform":
            weights = jax.random.uniform(val_key, shape, dtype, minval=-bound, maxval=bound)
        elif dist == "normal":
            weights = (jax.random.normal(val_key, shape, dtype) * bound).astype(dtype)
        else:
            raise ValueError(f"Unknown initialization type: {dist!r}")

        if num_zeros == 0:
            return weights

        perm_keys = jax.random.split(perm_key, fan_out)

        def zero_column(w_col: Array, k: Array) -> Array:
            perm = jax.random.permutation(k, fan_in)
            mask = jnp.ones((fan_in,), dtype=bool).at[perm[:num_zeros]].set(False)
            return jnp.where(mask, w_col, jnp.zeros((), dtype=dtype))

        return jax.vmap(zero_column, in_axes=(1, 0), out_axes=1)(weights, perm_keys)

    return init
