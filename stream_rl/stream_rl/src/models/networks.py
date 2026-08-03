"""Builds actor/critic Networks directly from block-chain architectures.
The last block's raw output IS the logits (actor) / value (critic); no
separate head module is applied, avoiding a redundant final projection.
"""
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from memorax.networks.feature_extractor import FeatureExtractor
from memorax.networks.blocks.ffn import Projection
from stream_rl.src.models.blocks import build_block_chain


class OneHotObservationEncoder(nn.Module):
    """One-hot encodes each integer-valued observation feature by its known
    per-feature cardinality before projecting to embed_dim.

    POPGym observations are category IDs (e.g. which card is face-up, which
    symbol was shown) packed into small-integer arrays. Feeding those raw
    integers straight into a Dense layer treats unrelated categories as
    ordered magnitudes -- e.g. card-color 0 vs 1 vs "face-down"=2 read as a
    numeric ramp instead of three unrelated states -- which starves the
    encoder of any usable structure. One-hot recovers the categorical
    structure at negligible extra cost given how small these cardinalities
    are (<=8 here).
    """
    low: tuple
    sizes: tuple
    embed_dim: int

    @nn.compact
    def __call__(self, obs):
        obs = obs.astype(jnp.int32) - jnp.asarray(self.low, dtype=jnp.int32)
        parts = [
            jax.nn.one_hot(obs[..., i], self.sizes[i])
            for i in range(len(self.sizes))
        ]
        x = jnp.concatenate(parts, axis=-1)
        x = nn.Dense(self.embed_dim)(x)
        return nn.tanh(x)


class FrozenProjectionEncoder(nn.Module):
    """Fixed random observation projection with no trainable parameters."""

    embed_dim: int
    seed: int = 0
    preserve_observation_prefix: int = 0

    @nn.compact
    def __call__(self, obs):
        obs = obs.astype(jnp.float32)
        input_dim = obs.shape[-1]
        key = jax.random.key(self.seed)
        w = jax.random.normal(key, (input_dim, self.embed_dim), dtype=obs.dtype)
        w = w / jnp.sqrt(float(max(input_dim, 1)))
        # No bias: a zero observation in a POMDP should be a genuine
        # no-signal input.  A fixed bias would be re-written at every blank
        # T-maze corridor step and can swamp the one-off cue being retained.
        projected = nn.tanh(jnp.einsum("...i,io->...o", obs, w))
        prefix = min(
            int(self.preserve_observation_prefix), int(input_dim), int(self.embed_dim)
        )
        if prefix:
            projected = projected.at[..., :prefix].set(obs[..., :prefix])
        return projected


class FrozenOneHotObservationEncoder(nn.Module):
    """Categorical one-hot encoder followed by a fixed random projection."""

    low: tuple
    sizes: tuple
    embed_dim: int
    seed: int = 0

    @nn.compact
    def __call__(self, obs):
        obs = obs.astype(jnp.int32) - jnp.asarray(self.low, dtype=jnp.int32)
        parts = [
            jax.nn.one_hot(obs[..., i], self.sizes[i])
            for i in range(len(self.sizes))
        ]
        x = jnp.concatenate(parts, axis=-1)
        return FrozenProjectionEncoder(self.embed_dim, self.seed)(x)


def _build_observation_extractor(
    observation_space,
    embed_dim: int,
    *,
    frozen: bool = False,
    seed: int = 0,
    preserve_observation_prefix: int = 0,
):
    """Integer-dtype spaces (POPGym's Discrete/MultiDiscrete/Tuple obs, all
    surfaced as int32 arrays by ExtendedPopGymWrapper) get one-hot encoded;
    everything else (continuous Box obs, e.g. TMaze/cartpole/pendulum) keeps
    the original raw Dense+tanh path unchanged."""
    if observation_space is not None and np.issubdtype(np.dtype(observation_space.dtype), np.integer):
        low = np.broadcast_to(np.asarray(observation_space.low), observation_space.shape).reshape(-1)
        high = np.broadcast_to(np.asarray(observation_space.high), observation_space.shape).reshape(-1)
        sizes = tuple((high - low + 1).astype(int).tolist())
        encoder_cls = FrozenOneHotObservationEncoder if frozen else OneHotObservationEncoder
        kwargs = dict(low=tuple(low.tolist()), sizes=sizes, embed_dim=embed_dim)
        if frozen:
            kwargs["seed"] = seed
        return encoder_cls(**kwargs)
    if frozen:
        return FrozenProjectionEncoder(
            embed_dim=embed_dim,
            seed=seed,
            preserve_observation_prefix=preserve_observation_prefix,
        )
    return nn.Sequential([nn.Dense(embed_dim), nn.tanh])


class RawLogitsHead(nn.Module):
    """Wraps the torso's raw output (features=action_dim via -1) as a
    Categorical distribution, matching the (dist, aux_dict) contract that
    ppo.py's stochastic_action expects from actor_network.apply."""

    @nn.compact
    def __call__(self, x, **kwargs):
        return distrax.Categorical(logits=x), {}


class RawValueHead(nn.Module):
    """Returns the torso's raw output unchanged, shape (..., 1) since the
    last block has features=1 via -1. ppo.py squeezes this itself via
    remove_feature_axis, and calls .loss(...) directly on this module —
    both match VNetwork's contract in heads.py."""

    @nn.compact
    def __call__(self, x, **kwargs):
        return x, {}

    def loss(self, output, aux, targets, **kwargs):
        return 0.5 * jnp.square(output - targets)


class AuxiliaryCategoricalHead(nn.Module):
    """Shared downstream readout for policy logits and cue reconstruction.

    The frozen encoder and memory remain parameter-free.  A small shared
    bottleneck is necessary: two independent linear heads would give the cue
    objective no gradient path into the policy readout and therefore could not
    improve how the policy consumes the frozen representation.
    """

    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x, **kwargs):
        shared = nn.tanh(nn.Dense(self.hidden_dim, name="shared_readout")(x))
        logits = nn.Dense(self.action_dim, name="policy_readout")(shared)
        cue = nn.tanh(nn.Dense(1, name="cue_readout")(shared))[..., 0]
        return distrax.Categorical(logits=logits), {"cue_prediction": cue}


class AuxiliaryValueHead(nn.Module):
    """Shared value/cue readout used with a frozen history representation."""

    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x, **kwargs):
        shared = nn.tanh(nn.Dense(self.hidden_dim, name="shared_readout")(x))
        value = nn.Dense(1, name="value_readout")(shared)
        cue = nn.tanh(nn.Dense(1, name="cue_readout")(shared))[..., 0]
        return value, {"cue_prediction": cue}

    def loss(self, output, aux, targets, **kwargs):
        return 0.5 * jnp.square(output - targets)
from memorax.networks.network import Network

def _validate_readout_only_architecture(
    architecture_cfg: list[dict], *, auxiliary_cue: bool = False
) -> None:
    """Ensure configs advertised as readout-only contain no hidden trainables."""
    if auxiliary_cue:
        if not architecture_cfg:
            return
        if len(architecture_cfg) != 1 or architecture_cfg[0].get("type") != "frozen_ssm":
            raise ValueError(
                "auxiliary readout_only architecture must be [] or [frozen_ssm]; "
                "all trainable downstream layers live in the network head."
            )
        return
    if len(architecture_cfg) == 1 and architecture_cfg[0].get("type") == "fc":
        return
    if len(architecture_cfg) != 2:
        raise ValueError(
            "readout_only requires exactly [frozen_ssm, fc] so the trainable "
            "parameter count really is limited to the final projection."
        )
    memory, readout = architecture_cfg
    if memory.get("type") != "frozen_ssm" or readout.get("type") not in {
        "fc",
        "gated_readout",
    }:
        raise ValueError(
            "readout_only architecture must be [frozen_ssm, fc] or "
            "[frozen_ssm, gated_readout]."
        )


def build_actor_network(
    architecture_cfg: list[dict],
    action_dim: int,
    embed_dim: int,
    tbptt_steps: int = 1,
    observation_space=None,
    readout_only: bool = False,
    frozen_encoder: bool = False,
    memory_seed: int = 0,
    auxiliary_cue: bool = False,
    auxiliary_readout_dim: int = 32,
    preserve_observation_prefix: int = 0,
):
    from memorax.networks.network import Network

    if readout_only:
        _validate_readout_only_architecture(
            architecture_cfg, auxiliary_cue=auxiliary_cue
        )

    return Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=_build_observation_extractor(
                observation_space,
                embed_dim,
                frozen=readout_only or frozen_encoder,
                seed=memory_seed,
                preserve_observation_prefix=preserve_observation_prefix,
            ),
        ),
        torso=build_block_chain(architecture_cfg, output_dim=action_dim, embed_dim=embed_dim, tbptt_steps=tbptt_steps),
        head=(
            AuxiliaryCategoricalHead(
                action_dim=action_dim, hidden_dim=auxiliary_readout_dim
            )
            if auxiliary_cue
            else RawLogitsHead()
        ),
    )


def build_critic_network(
    architecture_cfg: list[dict],
    embed_dim: int,
    tbptt_steps: int = 1,
    observation_space=None,
    readout_only: bool = False,
    frozen_encoder: bool = False,
    memory_seed: int = 0,
    auxiliary_cue: bool = False,
    auxiliary_readout_dim: int = 32,
    preserve_observation_prefix: int = 0,
):
    from memorax.networks.network import Network

    if readout_only:
        _validate_readout_only_architecture(
            architecture_cfg, auxiliary_cue=auxiliary_cue
        )

    return Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=_build_observation_extractor(
                observation_space,
                embed_dim,
                frozen=readout_only or frozen_encoder,
                seed=memory_seed,
                preserve_observation_prefix=preserve_observation_prefix,
            ),
        ),
        torso=build_block_chain(architecture_cfg, output_dim=1, embed_dim=embed_dim, tbptt_steps=tbptt_steps),
        head=(
            AuxiliaryValueHead(hidden_dim=auxiliary_readout_dim)
            if auxiliary_cue
            else RawValueHead()
        ),
    )
