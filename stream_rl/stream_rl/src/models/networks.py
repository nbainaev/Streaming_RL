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


def _build_observation_extractor(observation_space, embed_dim: int):
    """Integer-dtype spaces (POPGym's Discrete/MultiDiscrete/Tuple obs, all
    surfaced as int32 arrays by ExtendedPopGymWrapper) get one-hot encoded;
    everything else (continuous Box obs, e.g. TMaze/cartpole/pendulum) keeps
    the original raw Dense+tanh path unchanged."""
    if observation_space is not None and np.issubdtype(np.dtype(observation_space.dtype), np.integer):
        low = np.broadcast_to(np.asarray(observation_space.low), observation_space.shape).reshape(-1)
        high = np.broadcast_to(np.asarray(observation_space.high), observation_space.shape).reshape(-1)
        sizes = tuple((high - low + 1).astype(int).tolist())
        return OneHotObservationEncoder(low=tuple(low.tolist()), sizes=sizes, embed_dim=embed_dim)
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
from memorax.networks.network import Network

def build_actor_network(architecture_cfg: list[dict], action_dim: int, embed_dim: int, tbptt_steps: int = 1, observation_space=None):
    from memorax.networks.network import Network

    return Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=_build_observation_extractor(observation_space, embed_dim),
        ),
        torso=build_block_chain(architecture_cfg, output_dim=action_dim, embed_dim=embed_dim, tbptt_steps=tbptt_steps),
        head=RawLogitsHead(),
    )


def build_critic_network(architecture_cfg: list[dict], embed_dim: int, tbptt_steps: int = 1, observation_space=None):
    from memorax.networks.network import Network

    return Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=_build_observation_extractor(observation_space, embed_dim),
        ),
        torso=build_block_chain(architecture_cfg, output_dim=1, embed_dim=embed_dim, tbptt_steps=tbptt_steps),
        head=RawValueHead(),
    )