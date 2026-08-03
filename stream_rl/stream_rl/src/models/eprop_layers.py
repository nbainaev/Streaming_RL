# eprop_layers.py
"""Non-spiking e-prop recurrent cells (Bellec et al., 2020) for the
`stream_eprop` algorithm (see `stream_rl/src/agents/stream_eprop.py`).

Each cell maintains **local, learning-signal-free eligibility traces** —
a pure function of the forward pass, exactly Bellec et al.'s factorization:

    trace_t = trace_decay * trace_{t-1} + outer(presynaptic_t, pseudo_derivative_t)
    delta_theta = learning_signal (x) trace          # applied by the algorithm

One trace is kept per weight matrix, both "input" (x_t -> hidden) and
"recurrent" (h_{t-1} -> hidden) — both need eligibility, since both feed a
unit whose effect on the loss can only be felt in the future.

The learning signal (broadcast from the readout error through a feedback
"connectivity" matrix — symmetric / random / adaptive) is deliberately *not*
computed here: it lives in `stream_eprop.py`, which is the only place that
knows about the TD error and the readout. Keeping the two separate mirrors
Bellec et al.'s own factorization and is what makes `random`/`adaptive`
feedback meaningfully different from plain backprop (folding the learning
signal into the trace, as an earlier version of this file did, silently
collapses everything back into ordinary backprop, since the recursive
"multiply-by-decay-and-add-gradient" pattern is then indistinguishable from
a standard eligibility-trace optimizer).

trace_decay is a single configurable knob (rather than deriving each gate's
decay from the LSTM forget gate the way Bellec et al.'s exact formula does)
— a deliberate simplification so the same hyperparameter applies uniformly
across VanillaRNN/GRU/LSTM, per this project's config surface.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import flax.serialization
from flax import linen as nn
from jax.tree_util import register_pytree_node

from stream_rl.src.models.init import sparse_init

Array = jax.Array


@dataclass
class EpropCarry:
    h: Any  # hidden state: array, or (c, h) tuple for LSTM
    traces: Dict[str, Array]  # always a dict, never None


def _flatten_eprop_carry(carry):
    h_flat = carry.h if isinstance(carry.h, tuple) else (carry.h,)
    trace_keys = tuple(sorted(carry.traces.keys()))
    trace_values = tuple(carry.traces[k] for k in trace_keys)
    children = h_flat + trace_values
    metadata = {"h_is_tuple": isinstance(carry.h, tuple), "trace_keys": trace_keys}
    return children, metadata


def _unflatten_eprop_carry(metadata, children):
    h_is_tuple = metadata["h_is_tuple"]
    trace_keys = metadata["trace_keys"]
    if h_is_tuple:
        h_state = (children[0], children[1])
        idx = 2
    else:
        h_state = children[0]
        idx = 1
    traces = {key: children[idx + i] for i, key in enumerate(trace_keys)}
    return EpropCarry(h=h_state, traces=traces)


register_pytree_node(EpropCarry, _flatten_eprop_carry, _unflatten_eprop_carry)


def _epropcarry_to_state_dict(carry: "EpropCarry") -> dict:
    # Flax's msgpack encoder does not accept a bare Python tuple inside a
    # custom serialization handler.  Store LSTM (c, h) as a keyed mapping;
    # the target object supplied to from_bytes tells us whether to rebuild a
    # tuple or retain the single GRU/RNN array.
    hidden = (
        {"cell": carry.h[0], "hidden": carry.h[1]}
        if isinstance(carry.h, tuple)
        else carry.h
    )
    return {"h": hidden, "traces": carry.traces}


def _epropcarry_from_state_dict(carry: "EpropCarry", state_dict: dict) -> "EpropCarry":
    h = state_dict["h"]
    if isinstance(carry.h, tuple):
        h = (h["cell"], h["hidden"])
    return EpropCarry(h=h, traces=dict(state_dict["traces"]))


# JAX pytree registration (above) lets EpropCarry flow through jax.lax.scan /
# jax.jit; that's a separate registry from flax's msgpack (de)serialization
# (base_runner.py's checkpointing, via flax.serialization.to_bytes/from_bytes)
# -- without this, saving a stream_eprop checkpoint raises "can not
# serialize 'EpropCarry' object".
flax.serialization.register_serialization_state(
    EpropCarry, _epropcarry_to_state_dict, _epropcarry_from_state_dict
)


class _BaseEpropCell(nn.Module):
    hidden_size: int
    activation: str = "tanh"
    trace_decay: float = 0.9
    stop_gradients: bool = True
    param_dtype: jnp.dtype = jnp.float32
    # The two stabilizing ingredients from the streaming-RL literature that
    # this pipeline was missing (see stream_rl.src.models.blocks.LayerNormBlock
    # / stream_rl.src.models.init.sparse_init for the generic-block-chain
    # equivalents used by stream_ac/ppo). `use_layernorm` only normalizes the
    # value returned to the caller (the head's input) -- the raw hidden state
    # `h` kept in the carry (and used to compute the eligibility traces) is
    # left untouched, so recurrent dynamics/traces are unaffected.
    use_layernorm: bool = True
    use_sparse_init: bool = True
    sparsity: float = 0.9

    _trace_kind: str = "vanilla"  # overridden by subclasses

    def _input_kernel_init(self):
        return sparse_init(self.sparsity) if self.use_sparse_init else nn.initializers.xavier_uniform()

    def _maybe_layernorm(self, h: Array) -> Array:
        return nn.LayerNorm()(h) if self.use_layernorm else h

    def _act(self, x: Array) -> Array:
        a = self.activation.lower()
        if a == "tanh":
            return jnp.tanh(x)
        if a == "relu":
            return jax.nn.relu(x)
        if a in {"linear", "identity", "lin"}:
            return x
        raise ValueError(f"Unknown activation kind: {self.activation!r}")

    def _act_deriv(self, pre: Array, post: Array) -> Array:
        a = self.activation.lower()
        if a == "tanh":
            return 1.0 - post**2
        if a == "relu":
            return (pre > 0).astype(pre.dtype)
        if a in {"linear", "identity", "lin"}:
            return jnp.ones_like(pre)
        raise ValueError(f"Unknown activation kind: {self.activation!r}")

    @nn.nowrap
    def initialize_carry(self, key: Optional[jax.Array], input_shape: tuple) -> EpropCarry:
        batch_size = input_shape[0]
        in_dim = input_shape[-1]
        if in_dim is None:
            raise ValueError(
                "Eprop cells need a concrete input feature dim to size their "
                f"eligibility traces up front; got input_shape={input_shape}. "
                "Call initialize_carry with (batch_size, embed_dim), not "
                "(batch_size, None)."
            )
        h = jnp.zeros((batch_size, self.hidden_size), dtype=self.param_dtype)
        traces = self._init_traces(batch_size, in_dim)
        return EpropCarry(h=h, traces=traces)

    def _init_traces(self, batch_size: int, in_dim: int) -> Dict[str, Array]:
        hs = self.hidden_size
        zi = lambda: jnp.zeros((batch_size, in_dim, hs), dtype=self.param_dtype)
        zh = lambda: jnp.zeros((batch_size, hs, hs), dtype=self.param_dtype)
        zb = lambda: jnp.zeros((batch_size, hs), dtype=self.param_dtype)
        if self._trace_kind == "vanilla":
            return {"e_wxh": zi(), "e_whh": zh(), "e_b": zb()}
        if self._trace_kind == "gru":
            return {
                "e_wz": zi(), "e_wr": zi(), "e_wn": zi(),
                "e_uz": zh(), "e_ur": zh(), "e_un": zh(),
                "e_bz": zb(), "e_br": zb(), "e_bn": zb(), "e_bhn": zb(),
            }
        if self._trace_kind == "lstm":
            return {
                "e_wi": zi(), "e_wf": zi(), "e_wg": zi(), "e_wo": zi(),
                "e_ui": zh(), "e_uf": zh(), "e_ug": zh(), "e_uo": zh(),
                "e_bi": zb(), "e_bf": zb(), "e_bg": zb(), "e_bo": zb(),
            }
        raise ValueError(f"Unknown trace kind: {self._trace_kind!r}")

    def __call__(self, inputs: Array, done=None, initial_carry: Optional[EpropCarry] = None, **kwargs):
        if initial_carry is None:
            initial_carry = self.initialize_carry(None, inputs.shape)
        new_carry, output = self._step(initial_carry, inputs)
        return new_carry, output


class VanillaRNN(_BaseEpropCell):
    _trace_kind: str = "vanilla"

    @nn.compact
    def _step(self, carry: EpropCarry, inputs: Array):
        x = inputs
        h_prev = jax.lax.stop_gradient(carry.h) if self.stop_gradients else carry.h

        wxh = self.param("wxh", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        whh = self.param("whh", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        b = self.param("b", nn.initializers.zeros, (self.hidden_size,))

        pre = x @ wxh + h_prev @ whh + b
        h = self._act(pre)
        dphi = self._act_deriv(pre, h)

        decay = self.trace_decay
        traces = {
            "e_wxh": decay * carry.traces["e_wxh"] + jnp.einsum("bi,bj->bij", x, dphi),
            "e_whh": decay * carry.traces["e_whh"] + jnp.einsum("bi,bj->bij", h_prev, dphi),
            "e_b": decay * carry.traces["e_b"] + dphi,
        }
        return EpropCarry(h=h, traces=traces), self._maybe_layernorm(h)


class GRU(_BaseEpropCell):
    _trace_kind: str = "gru"
    forget_bias: float = 1.0

    @nn.compact
    def _step(self, carry: EpropCarry, inputs: Array):
        # Matches flax.linen.GRUCell's gate wiring exactly (Cho et al.'s
        # standard GRU): the reset gate multiplies the *recurrent matmul's
        # output* (h_prev @ un + bhn), not h_prev itself before the matmul —
        # getting this backwards changes the cell's actual dynamics, not
        # just its parameterization. z/r have a single (input-side) bias;
        # n's recurrent path gets its own bias since it isn't summed
        # directly with the input path before the gate multiply.
        x = inputs
        h_prev = jax.lax.stop_gradient(carry.h) if self.stop_gradients else carry.h

        wz = self.param("wz", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        wr = self.param("wr", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        wn = self.param("wn", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        uz = self.param("uz", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        ur = self.param("ur", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        un = self.param("un", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        bz = self.param("bz", nn.initializers.zeros, (self.hidden_size,))
        br = self.param("br", nn.initializers.zeros, (self.hidden_size,))
        bn = self.param("bn", nn.initializers.zeros, (self.hidden_size,))
        bhn = self.param("bhn", nn.initializers.zeros, (self.hidden_size,))

        z_pre = x @ wz + h_prev @ uz + bz + self.forget_bias
        r_pre = x @ wr + h_prev @ ur + br
        z = jax.nn.sigmoid(z_pre)
        r = jax.nn.sigmoid(r_pre)
        hn_pre = h_prev @ un + bhn
        n_pre = x @ wn + bn + r * hn_pre
        n = self._act(n_pre)
        # z=1 retains h_prev, z=0 fully updates to n (flax/Cho et al.
        # convention) — forget_bias biases z_pre positive so the cell
        # favors *retaining* memory early in training, matching the
        # LSTM forget-gate-bias trick this is modeled on. Swapping this
        # (as an earlier version of this file did) silently inverts what
        # forget_bias does.
        h = (1.0 - z) * n + z * h_prev

        # Each eligibility must be a derivative of the *cell output h*, not
        # merely a derivative of the corresponding gate.  Broadcasting a
        # dL/dh learning signal through bare dz/dr/dn (the previous
        # implementation) drops the chain-rule factors below and gives an
        # incorrectly scaled -- and for the update gate often incorrectly
        # signed -- recurrent update.
        dz = z * (1.0 - z)
        dr = r * (1.0 - r)
        dn = self._act_deriv(n_pre, n)
        z_sensitivity = (h_prev - n) * dz
        n_sensitivity = (1.0 - z) * dn
        r_sensitivity = n_sensitivity * hn_pre * dr
        recurrent_n_sensitivity = n_sensitivity * r

        decay = self.trace_decay
        traces = {
            "e_wz": decay * carry.traces["e_wz"] + jnp.einsum("bi,bj->bij", x, z_sensitivity),
            "e_wr": decay * carry.traces["e_wr"] + jnp.einsum("bi,bj->bij", x, r_sensitivity),
            "e_wn": decay * carry.traces["e_wn"] + jnp.einsum("bi,bj->bij", x, n_sensitivity),
            "e_uz": decay * carry.traces["e_uz"] + jnp.einsum("bi,bj->bij", h_prev, z_sensitivity),
            "e_ur": decay * carry.traces["e_ur"] + jnp.einsum("bi,bj->bij", h_prev, r_sensitivity),
            "e_un": decay * carry.traces["e_un"] + jnp.einsum("bi,bj->bij", h_prev, recurrent_n_sensitivity),
            "e_bz": decay * carry.traces["e_bz"] + z_sensitivity,
            "e_br": decay * carry.traces["e_br"] + r_sensitivity,
            "e_bn": decay * carry.traces["e_bn"] + n_sensitivity,
            "e_bhn": decay * carry.traces["e_bhn"] + recurrent_n_sensitivity,
        }
        return EpropCarry(h=h, traces=traces), self._maybe_layernorm(h)


class LSTM(_BaseEpropCell):
    _trace_kind: str = "lstm"
    forget_bias: float = 1.0

    @nn.nowrap
    def initialize_carry(self, key: Optional[jax.Array], input_shape: tuple) -> EpropCarry:
        batch_size = input_shape[0]
        in_dim = input_shape[-1]
        if in_dim is None:
            raise ValueError(
                "Eprop cells need a concrete input feature dim to size their "
                f"eligibility traces up front; got input_shape={input_shape}."
            )
        c = jnp.zeros((batch_size, self.hidden_size), dtype=self.param_dtype)
        h = jnp.zeros((batch_size, self.hidden_size), dtype=self.param_dtype)
        traces = self._init_traces(batch_size, in_dim)
        return EpropCarry(h=(c, h), traces=traces)

    @nn.compact
    def _step(self, carry: EpropCarry, inputs: Array):
        x = inputs
        c_prev, h_prev = carry.h
        if self.stop_gradients:
            c_prev = jax.lax.stop_gradient(c_prev)
            h_prev = jax.lax.stop_gradient(h_prev)

        wi = self.param("wi", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        wf = self.param("wf", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        wg = self.param("wg", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        wo = self.param("wo", self._input_kernel_init(), (x.shape[-1], self.hidden_size))
        ui = self.param("ui", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        uf = self.param("uf", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        ug = self.param("ug", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        uo = self.param("uo", nn.initializers.orthogonal(), (self.hidden_size, self.hidden_size))
        bi = self.param("bi", nn.initializers.zeros, (self.hidden_size,))
        bf = self.param("bf", nn.initializers.zeros, (self.hidden_size,))
        bg = self.param("bg", nn.initializers.zeros, (self.hidden_size,))
        bo = self.param("bo", nn.initializers.zeros, (self.hidden_size,))

        i_pre = x @ wi + h_prev @ ui + bi
        f_pre = x @ wf + h_prev @ uf + bf + self.forget_bias
        g_pre = x @ wg + h_prev @ ug + bg
        o_pre = x @ wo + h_prev @ uo + bo
        i = jax.nn.sigmoid(i_pre)
        f = jax.nn.sigmoid(f_pre)
        g = self._act(g_pre)
        c = f * c_prev + i * g
        o = jax.nn.sigmoid(o_pre)
        h = o * jnp.tanh(c)

        # Local (non-recursive) sensitivity of c_t to each gate's weights,
        # holding c_{t-1} fixed — the "presynaptic x pseudo-derivative" term
        # of the eligibility trace, uniformly low-pass filtered by
        # trace_decay (a simplification of Bellec et al.'s exact per-gate
        # LSTM trace, which instead decays e_i/e_f/e_g through the forget
        # gate f_t itself; trace_decay is kept as the single configurable
        # knob per this project's config surface). o_t doesn't feed c_t at
        # all (only h_t = o_t*tanh(c_t) directly), so its "trace" is really
        # instantaneous — decaying it too is a harmless simplification for
        # implementation uniformity.
        # The learning signal consumed by StreamEprop is dL/dh.  Therefore
        # all gate traces must also describe dh/dtheta.  The previous code
        # stored dc/dtheta for i/f/g but dh/dtheta for o, then multiplied all
        # four by the same learning signal.  Apply dh/dc here so the trace
        # dictionary has one consistent meaning.
        tanh_c = jnp.tanh(c)
        dh_dc = o * (1.0 - tanh_c**2)
        di_g = dh_dc * (i * (1.0 - i)) * g
        df_c = dh_dc * (f * (1.0 - f)) * c_prev
        i_dg = dh_dc * i * self._act_deriv(g_pre, g)
        do_tanh_c = (o * (1.0 - o)) * tanh_c

        decay = self.trace_decay
        traces = {
            "e_wi": decay * carry.traces["e_wi"] + jnp.einsum("bi,bj->bij", x, di_g),
            "e_wf": decay * carry.traces["e_wf"] + jnp.einsum("bi,bj->bij", x, df_c),
            "e_wg": decay * carry.traces["e_wg"] + jnp.einsum("bi,bj->bij", x, i_dg),
            "e_wo": decay * carry.traces["e_wo"] + jnp.einsum("bi,bj->bij", x, do_tanh_c),
            "e_ui": decay * carry.traces["e_ui"] + jnp.einsum("bi,bj->bij", h_prev, di_g),
            "e_uf": decay * carry.traces["e_uf"] + jnp.einsum("bi,bj->bij", h_prev, df_c),
            "e_ug": decay * carry.traces["e_ug"] + jnp.einsum("bi,bj->bij", h_prev, i_dg),
            "e_uo": decay * carry.traces["e_uo"] + jnp.einsum("bi,bj->bij", h_prev, do_tanh_c),
            "e_bi": decay * carry.traces["e_bi"] + di_g,
            "e_bf": decay * carry.traces["e_bf"] + df_c,
            "e_bg": decay * carry.traces["e_bg"] + i_dg,
            "e_bo": decay * carry.traces["e_bo"] + do_tanh_c,
        }
        return EpropCarry(h=(c, h), traces=traces), self._maybe_layernorm(h)


CELL_TYPES = {"eprop_rnn": VanillaRNN, "eprop_gru": GRU, "eprop_lstm": LSTM}


def build_eprop_cell(
    cell_name: str,
    hidden_size: int,
    trace_decay: float = 0.9,
    activation: str = "tanh",
    use_layernorm: bool = True,
    use_sparse_init: bool = True,
    sparsity: float = 0.9,
) -> _BaseEpropCell:
    if cell_name not in CELL_TYPES:
        raise ValueError(f"Unknown eprop cell {cell_name!r}. Expected one of: {list(CELL_TYPES)}")
    return CELL_TYPES[cell_name](
        hidden_size=hidden_size,
        trace_decay=trace_decay,
        activation=activation,
        use_layernorm=use_layernorm,
        use_sparse_init=use_sparse_init,
        sparsity=sparsity,
    )


class EpropTorso(nn.Module):
    """embedding (Dense+tanh) -> single e-prop recurrent cell.

    Exposes the cell's raw output `h` directly (not hidden inside a fused
    apply call) so `stream_eprop.py` can split "local eligibility trace"
    from "learning signal" at exactly this boundary.
    """

    embed_dim: int
    cell: _BaseEpropCell

    @nn.compact
    def __call__(self, observation: Array, initial_carry: Optional[EpropCarry] = None):
        x = nn.tanh(nn.Dense(self.embed_dim, name="embed")(observation))
        carry, h = self.cell(x, initial_carry=initial_carry)
        return carry, h

    @nn.nowrap
    def initialize_carry(self, key: Optional[jax.Array], num_envs: int) -> EpropCarry:
        return self.cell.initialize_carry(key, (num_envs, self.embed_dim))
