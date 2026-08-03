import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from memorax.networks.blocks.ffn import FFN, Projection, GLU
from memorax.networks.sequence_models.rnn import RNN, RNNCellBase
from memorax.networks.sequence_models.rtrl import RTRL
from memorax.networks.sequence_models.rtu import RTUCell, RTUConfig

from stream_rl.src.models.init import sparse_init
from stream_rl.src.models.delta_rule import DeltaRuleCell


class LayerNormBlock(nn.Module):
    """Normalizes hidden activations before the next block — one of the two
    stabilizing ingredients (with `init: sparse`, see `sparse_init`) that
    streaming/online TD updates need and that this pipeline was missing.
    Stateless (Block protocol: returns `(carry=None, output)`), features-less
    passthrough (output width == input width), so it slots in anywhere in an
    `architecture_cfg` list without needing a `features` key."""

    @nn.compact
    def __call__(self, inputs, done=None, initial_carry=None, **kwargs):
        return None, nn.LayerNorm()(inputs)

    @nn.nowrap
    def initialize_carry(self, key, input_shape: tuple):
        return None


class GatedReadoutBlock(nn.Module):
    """Small phase-gated readout for ``[current_features, memory_features]``.

    The current-observation path learns navigation while the memory path learns
    delayed decisions.  A scalar gate inferred from the current features keeps
    gradients from the two phases from destructively sharing one linear head.
    No history is stored here; the block is a strictly per-transition readout.
    """

    features: int
    split_index: int
    fixed_gate_index: int = -1
    navigation_prefix: int = 0

    @nn.compact
    def __call__(self, inputs, done=None, initial_carry=None, **kwargs):
        del done, initial_carry, kwargs
        if not 0 < self.split_index < inputs.shape[-1]:
            raise ValueError(
                "gated_readout split_index must lie inside the feature axis; "
                f"got split_index={self.split_index}, width={inputs.shape[-1]}."
            )
        current = inputs[..., : self.split_index]
        memory = inputs[..., self.split_index :]
        if self.fixed_gate_index >= 0:
            gate = jnp.clip(
                current[..., self.fixed_gate_index : self.fixed_gate_index + 1],
                0.0,
                1.0,
            )
        else:
            gate = nn.sigmoid(nn.Dense(1, name="phase_gate")(current))
        navigation_input = (
            current[..., : self.navigation_prefix]
            if self.navigation_prefix > 0
            else current
        )
        navigation = nn.Dense(self.features, name="navigation_readout")(
            navigation_input
        )
        recalled = nn.Dense(self.features, name="memory_readout")(memory)
        return None, navigation + gate * recalled

    @nn.nowrap
    def initialize_carry(self, key, input_shape: tuple):
        del key, input_shape
        return None


class FrozenSSMMemoryBlock(nn.Module):
    """Parameter-free, multi-timescale SSM used as a frozen memory prior.

    The transition/input/readout matrices are deterministic functions of
    ``seed`` and are deliberately *not* Flax parameters.  Consequently PPO
    and streaming AC only see parameters in the downstream readout.  This is
    the SSM analogue of using a frozen sequence model as a history compressor:

        h_t = A h_{t-1} + B x_t,   y_t = tanh(C h_t + D x_t)

    ``A`` is a stable diagonal spectrum spread between ``min_decay`` and
    ``max_decay``.  The broad spectrum gives the fixed state short and long
    memory channels without task-specific RL training.

    Important: this block is a structured frozen prior, not an LLM checkpoint.
    Keeping that distinction explicit prevents random fixed features from
    being reported as pretrained-language-model evidence.
    """

    features: int
    state_dim: int
    seed: int = 0
    min_decay: float = 0.5
    max_decay: float = 0.999
    input_scale: float = 1.0
    residual_scale: float = 0.1
    concatenate_input: bool = False
    decay_values: tuple | None = None
    input_matrix: tuple | None = None
    readout_matrix: tuple | None = None
    residual_matrix: tuple | None = None

    def _constants(self, input_dim: int, dtype):
        if self.decay_values is not None:
            decay_np = np.asarray(self.decay_values)
            b_np = np.asarray(self.input_matrix)
            c_np = np.asarray(self.readout_matrix)
            d_np = np.asarray(self.residual_matrix)
            expected = {
                "decay": (self.state_dim,),
                "B": (self.state_dim, input_dim),
                "C": (self.features, self.state_dim),
                "D": (self.features, input_dim),
            }
            actual = {
                "decay": decay_np.shape,
                "B": b_np.shape,
                "C": c_np.shape,
                "D": d_np.shape,
            }
            if actual != expected:
                raise ValueError(
                    f"Frozen SSM checkpoint shape mismatch: expected {expected}, got {actual}."
                )
            if not np.all((decay_np >= 0.0) & (decay_np < 1.0)):
                raise ValueError("Frozen SSM checkpoint decay must lie in [0, 1).")
            decay = jnp.asarray(decay_np, dtype=dtype)
            b = jnp.asarray(b_np, dtype=dtype)
            c = jnp.asarray(c_np, dtype=dtype)
            d = jnp.asarray(d_np, dtype=dtype)
            return decay, b, c, d

        if not 0.0 <= self.min_decay < self.max_decay < 1.0:
            raise ValueError(
                "Frozen SSM requires 0 <= min_decay < max_decay < 1; "
                f"got {self.min_decay}, {self.max_decay}."
            )

        key = jax.random.key(self.seed)
        b_key, c_key, d_key = jax.random.split(key, 3)
        decay = jnp.geomspace(
            max(self.min_decay, 1e-4), self.max_decay, self.state_dim
        ).astype(dtype)
        # Normalizing B by sqrt(input_dim) and each state channel by its
        # innovation variance keeps slow modes from dominating numerically.
        b = jax.random.normal(b_key, (self.state_dim, input_dim), dtype=dtype)
        b = self.input_scale * b / jnp.sqrt(float(max(input_dim, 1)))
        b = b * jnp.sqrt(jnp.maximum(1.0 - decay**2, 1e-6))[:, None]
        c = jax.random.normal(c_key, (self.features, self.state_dim), dtype=dtype)
        c = c / jnp.sqrt(float(max(self.state_dim, 1)))
        d = jax.random.normal(d_key, (self.features, input_dim), dtype=dtype)
        d = self.residual_scale * d / jnp.sqrt(float(max(input_dim, 1)))
        return decay, b, c, d

    @nn.compact
    def __call__(self, inputs, done=None, initial_carry=None, **kwargs):
        del kwargs
        if inputs.ndim != 3:
            raise ValueError(
                f"FrozenSSMMemoryBlock expects [batch, time, features], got {inputs.shape}."
            )
        batch_size, _, input_dim = inputs.shape
        dtype = inputs.dtype
        decay, b, c, d = self._constants(input_dim, dtype)

        if initial_carry is None:
            state = jnp.zeros((batch_size, self.state_dim), dtype=dtype)
        else:
            state = jnp.asarray(initial_carry, dtype=dtype)
            if state.ndim == 3 and state.shape[-2] == 1:
                state = state[..., 0, :]

        if done is None:
            done = jnp.zeros(inputs.shape[:2], dtype=jnp.bool_)

        x_time = jnp.swapaxes(inputs, 0, 1)
        done_time = jnp.swapaxes(done.astype(jnp.bool_), 0, 1)

        def step(h, xs):
            x_t, reset_t = xs
            h = jnp.where(reset_t[:, None], jnp.zeros_like(h), h)
            h = decay[None, :] * h + jnp.einsum("hi,bi->bh", b, x_t)
            y = jnp.tanh(
                jnp.einsum("oh,bh->bo", c, h)
                + jnp.einsum("oi,bi->bo", d, x_t)
            )
            return h, y

        state, outputs = jax.lax.scan(step, state, (x_time, done_time))
        outputs = jnp.swapaxes(outputs, 0, 1)
        if self.concatenate_input:
            outputs = jnp.concatenate([inputs, outputs], axis=-1)
        return state[:, None, :], outputs

    @nn.nowrap
    def initialize_carry(self, key, input_shape: tuple):
        del key
        batch_size = int(input_shape[0])
        return jnp.zeros((batch_size, 1, self.state_dim), dtype=jnp.float32)

# NOTE: eprop_rnn/eprop_gru/eprop_lstm (stream_rl.src.models.eprop_layers)
# are intentionally NOT wired in here. The generic StreamAC path stops the
# previous carry and therefore produces a one-step truncated gradient, not
# e-prop. Explicit eligibility traces also need the concrete input feature
# width up front, which this block chain cannot supply at carry init. Use
# agent: stream_eprop; it builds its torso with a concrete embed_dim.
ACTIVATIONS = {"tanh": nn.tanh, "relu": nn.relu, "gelu": nn.gelu, None: None}


CELL_REGISTRY = {
    "gru": lambda features, in_features: nn.GRUCell(features=features),
    "lstm": lambda features, in_features: nn.LSTMCell(features=features),
}


RTRL_CELLS = {"rtu_rtrl"}
NEEDS_INPUT_DIM = {"rtu_rtrl", "rtu_bptt"}


def _build_rtu_cell(cfg: dict, hidden_dim: int, in_features: int) -> RTUCell:
    rtu_cfg = RTUConfig(
        features=in_features,
        hidden_dim=hidden_dim,
        r_min=cfg.get("r_min", 0.0),
        r_max=cfg.get("r_max", 1.0),
        max_phase=cfg.get("max_phase", 6.28),
        eps=cfg.get("eps", 1e-8),
        activation_fn=ACTIVATIONS[cfg.get("activation", "tanh")],
    )
    return RTUCell(config=rtu_cfg)


def _filter_cfg_for_cell(cfg: dict) -> dict:
    forbidden = {"type", "cell", "features"}
    return {k: v for k, v in cfg.items() if k not in forbidden}


def _kernel_init_kwargs(cfg: dict) -> dict:
    """`init: sparse` (optionally `sparsity: <float>`, default 0.9) opts a
    fc/ffn/glu block into Elsayed et al.'s sparse initialization instead of
    the memorax default (lecun_normal); omit `init` to keep the default."""
    if cfg.get("init") != "sparse":
        return {}
    return {"kernel_init": sparse_init(sparsity=cfg.get("sparsity", 0.9))}


def _load_frozen_ssm_checkpoint(path: str | None) -> dict:
    """Load non-trainable SSM arrays exported as decay/B/C/D in an NPZ."""
    if not path:
        return {}
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Frozen SSM checkpoint not found: {checkpoint}")
    with np.load(checkpoint) as data:
        missing = {"decay", "B", "C", "D"} - set(data.files)
        if missing:
            raise ValueError(
                f"Frozen SSM checkpoint {checkpoint} is missing arrays: {sorted(missing)}"
            )
        # Hashable tuples keep the Flax module definition static under JIT.
        return {
            "decay_values": tuple(data["decay"].tolist()),
            "input_matrix": tuple(map(tuple, data["B"].tolist())),
            "readout_matrix": tuple(map(tuple, data["C"].tolist())),
            "residual_matrix": tuple(map(tuple, data["D"].tolist())),
        }


def build_block(cfg: dict, resolved_features: int, in_features: int | None = None, tbptt_steps: int = 1) -> nn.Module:
    block_type = cfg["type"]

    if block_type == "layernorm":
        return LayerNormBlock()
    if block_type == "frozen_ssm":
        return FrozenSSMMemoryBlock(
            features=resolved_features,
            state_dim=int(cfg.get("state_dim", resolved_features)),
            seed=int(cfg.get("seed", 0)),
            min_decay=float(cfg.get("min_decay", 0.5)),
            max_decay=float(cfg.get("max_decay", 0.999)),
            input_scale=float(cfg.get("input_scale", 1.0)),
            residual_scale=float(cfg.get("residual_scale", 0.1)),
            concatenate_input=bool(cfg.get("concatenate_input", False)),
            **_load_frozen_ssm_checkpoint(cfg.get("checkpoint_path")),
        )
    if block_type == "fc":
        return Projection(
            features=resolved_features,
            activation_fn=ACTIVATIONS[cfg.get("activation")],
            **_kernel_init_kwargs(cfg),
        )
    if block_type == "gated_readout":
        return GatedReadoutBlock(
            features=resolved_features,
            split_index=int(cfg.get("split_index", in_features or 0)),
            fixed_gate_index=int(cfg.get("fixed_gate_index", -1)),
            navigation_prefix=int(cfg.get("navigation_prefix", 0)),
        )
    if block_type == "ffn":
        return FFN(features=resolved_features, expansion_factor=cfg.get("expansion_factor", 4), **_kernel_init_kwargs(cfg))
    if block_type == "glu":
        return GLU(features=resolved_features, expansion_factor=cfg.get("expansion_factor", 4), **_kernel_init_kwargs(cfg))

    if block_type == "rnn":
        cell_name = cfg["cell"]

        # RTU / RTRL
        if cell_name in RTRL_CELLS:
            if in_features is None:
                raise ValueError(
                    f"cell={cell_name!r} needs the running input width; build_block_chain must pass in_features for RTRL cells."
                )
            cell = _build_rtu_cell(cfg, hidden_dim=resolved_features, in_features=in_features)
            assert isinstance(cell, RNNCellBase), (
                "RTU cell must implement the RTRL protocol (RNNCellBase) to be wrapped in RTRL(); got a cell without local_jacobian support."
            )
            return RTRL(sequence_model=RNN(cell=cell))

        if cell_name == "rtu_bptt":
            if in_features is None:
                raise ValueError("rtu_bptt needs in_features")
            cell = _build_rtu_cell(cfg, hidden_dim=resolved_features, in_features=in_features)
            # Для BPTT-ячеек используем переданный tbptt_steps
            return RNN(cell=cell, unroll=tbptt_steps)

        if cell_name == "delta_rule":
            cell = DeltaRuleCell(
                features=resolved_features,
                key_dim=cfg.get("key_dim", min(16, resolved_features)),
                min_decay=cfg.get("min_decay", 0.90),
                max_decay=cfg.get("max_decay", 0.999),
            )
            return RNN(cell=cell, unroll=1)

        # Обычные GRU/LSTM
        if cell_name not in CELL_REGISTRY:
            raise ValueError(
                f"Unknown cell type: {cell_name}. Expected GRU, LSTM, "
                "delta_rule, RTU-BPTT or RTU-RTRL."
            )
        cell = CELL_REGISTRY[cell_name](resolved_features, in_features)
        # Для GRU/LSTM используем tbptt_steps
        return RNN(cell=cell, unroll=tbptt_steps)

    raise ValueError(f"Unknown block type: {block_type}")

def _block_output_width(
    layer: dict, resolved_features: int, input_width: int
) -> int:
    block_type = layer["type"]
    if block_type == "frozen_ssm" and layer.get("concatenate_input", False):
        return input_width + resolved_features
    if block_type == "rnn":
        return resolved_features
    return resolved_features


# Block types that take no `features` key and don't change the running
# width flowing through the chain (e.g. LayerNorm normalizes in place).
FEATURELESS_TYPES = {"layernorm"}


def resolve_architecture(layers: list[dict], output_dim: int) -> list[dict]:
    resolved = []
    for i, layer in enumerate(layers):
        layer = dict(layer)
        if layer["type"] in FEATURELESS_TYPES:
            resolved.append(layer)
            continue
        is_last = i == len(layers) - 1
        if layer["features"] == -1:
            if not is_last:
                raise ValueError(
                    f"features=-1 is only allowed on the last block (block {i} is not last)."
                )
            layer["features"] = output_dim
        resolved.append(layer)
    return resolved


def build_block_chain(architecture_cfg: list[dict], output_dim: int, embed_dim: int, tbptt_steps: int = 1):
    from stream_rl.src.models.block_chain import BlockChain

    resolved = resolve_architecture(architecture_cfg, output_dim)
    blocks = []
    running_width = embed_dim

    for layer in resolved:
        if layer["type"] in FEATURELESS_TYPES:
            blocks.append(build_block(layer, running_width, tbptt_steps=tbptt_steps))
            continue
        needs_dim = (
            (layer["type"] == "rnn" and layer.get("cell") in NEEDS_INPUT_DIM)
            or layer["type"] == "gated_readout"
        )
        block = build_block(
            layer,
            layer["features"],
            in_features=running_width if needs_dim else None,
            tbptt_steps=tbptt_steps,      # передаём
        )
        blocks.append(block)
        # обновление running_width (без изменений)
        if layer["type"] == "rnn" and layer.get("cell") in {"rtu_rtrl", "rtu_bptt"}:
            running_width = 2 * layer["features"]
        else:
            running_width = _block_output_width(layer, layer["features"], running_width)

    return BlockChain(blocks=blocks)
