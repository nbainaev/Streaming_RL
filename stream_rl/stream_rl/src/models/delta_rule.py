"""Small recurrent delta-rule memory for online POMDP experiments.

The layer keeps a fixed-size fast-weight matrix ``M``.  Each observation
produces a key, query and value.  The delta update first reads the value
currently associated with the key and writes only the residual::

    M_t = decay_t * M_{t-1} + beta_t * k_t (v_t - M_{t-1}^T k_t)^T

The implementation deliberately exposes an ordinary Flax recurrent cell.
When it is used by Memorax ``StreamAC``, the previous carry is detached before
the per-step Jacobian is formed.  Training is therefore the requested
one-step RTRL/TBPTT approximation; no claim of exact RTRL is made.
"""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class DeltaRuleCell(nn.RNNCellBase):
    """Gated linear-attention memory with a delta-rule write."""

    features: int
    key_dim: int = 16
    min_decay: float = 0.90
    max_decay: float = 0.999
    eps: float = 1e-6

    @nn.compact
    def __call__(self, carry, inputs):
        memory = carry
        query = nn.Dense(self.key_dim, name="query")(inputs)
        key = nn.Dense(self.key_dim, name="key")(inputs)
        value = nn.tanh(nn.Dense(self.features, name="value")(inputs))
        beta = nn.sigmoid(nn.Dense(1, name="write_gate")(inputs))
        decay_gate = nn.sigmoid(nn.Dense(1, name="decay_gate")(inputs))

        query = query / jnp.maximum(
            jnp.linalg.norm(query, axis=-1, keepdims=True), self.eps
        )
        key = key / jnp.maximum(
            jnp.linalg.norm(key, axis=-1, keepdims=True), self.eps
        )
        decay = self.min_decay + (self.max_decay - self.min_decay) * decay_gate

        previous = jnp.einsum("...kv,...k->...v", memory, key)
        residual = value - previous
        write = jnp.einsum("...k,...v->...kv", key, beta * residual)
        new_memory = decay[..., None] * memory + write

        readout = jnp.einsum("...kv,...k->...v", new_memory, query)
        output = nn.LayerNorm(name="readout_norm")(readout)
        return new_memory, output

    @nn.nowrap
    def initialize_carry(self, rng, input_shape: tuple[int, ...]):
        del rng
        batch_dims = input_shape[:-1]
        return jnp.zeros(
            (*batch_dims, self.key_dim, self.features), dtype=jnp.float32
        )

    @property
    def num_feature_axes(self) -> int:
        return 1
