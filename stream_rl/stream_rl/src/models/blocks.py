import flax.linen as nn

from memorax.networks.blocks.ffn import FFN, Projection, GLU
from memorax.networks.sequence_models.rnn import RNN, RNNCellBase
from memorax.networks.sequence_models.rtrl import RTRL
from memorax.networks.sequence_models.rtu import RTUCell, RTUConfig

from stream_rl.src.models.init import sparse_init


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

# NOTE: eprop_rnn/eprop_gru/eprop_lstm (stream_rl.src.models.eprop_layers)
# are intentionally NOT wired in here. Plugging one of those cells into the
# generic stream_ac path would give you "symmetric e-prop" for free (see
# stream_eprop.py's module docstring for why), but their eligibility traces
# need to know the *actual* input feature width up front, which this
# generic block chain can't supply (BlockChain.initialize_carry always
# calls initialize_carry(key, (num_envs, None))). Use agent: stream_eprop
# for eprop cells; it builds its own torso with a concrete embed_dim.
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


def build_block(cfg: dict, resolved_features: int, in_features: int | None = None, tbptt_steps: int = 1) -> nn.Module:
    block_type = cfg["type"]

    if block_type == "layernorm":
        return LayerNormBlock()
    if block_type == "fc":
        return Projection(
            features=resolved_features,
            activation_fn=ACTIVATIONS[cfg.get("activation")],
            **_kernel_init_kwargs(cfg),
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

        # Обычные GRU/LSTM
        if cell_name not in CELL_REGISTRY:
            raise ValueError(f"Unknown cell type: {cell_name}")
        cell = CELL_REGISTRY[cell_name](resolved_features, in_features)
        # Для GRU/LSTM используем tbptt_steps
        return RNN(cell=cell, unroll=tbptt_steps)

    raise ValueError(f"Unknown block type: {block_type}")

def _block_output_width(block_type: str, resolved_features: int) -> int:
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
        needs_dim = layer["type"] == "rnn" and layer.get("cell") in NEEDS_INPUT_DIM
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
            running_width = _block_output_width(layer["type"], layer["features"])

    return BlockChain(blocks=blocks)