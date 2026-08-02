# stream_rl/src/agents/stream_eprop.py
"""Streaming actor-critic with genuine e-prop recurrent layers.

`memorax.algorithms.stream_ac.StreamAC` computes per-step parameter
gradients via `jax.jacobian(loss_fn)(params)` with the previous recurrent
carry `stop_gradient`-ed. That is mathematically "symmetric e-prop" already
(single-step-truncated BPTT with an exact learning signal — see Bellec et
al., 2020, and their own `CustomLSTM` for e-prop, which does the exact same
stop-gradient-on-h_{t-1} trick). Plugging a plain recurrent cell into stock
`StreamAC` (e.g. `stream_ac_gru.yaml`) already gets you that case for free.

What stock `StreamAC` cannot give you — and what this module exists for —
is **random/feedback-alignment e-prop**: replacing the exact backprop-
computed learning signal with a broadcast through a fixed (or slowly
adapting) matrix, decoupled from the true readout weights. That requires
bypassing `jax.jacobian` for the recurrent layer's own weights, so this is
a full standalone algorithm (not a `StreamAC` subclass) — but it still
lives entirely above `memorax` (no library files touched) and reuses
`memorax.utils.{Timestep,Transition}` plus a copy of `StreamAC`'s ObGD
step-size normalization (`_obgd_update`, identical to
`memorax.algorithms.stream_ac.StreamAC._obgd_update`).

Three parameter groups are updated every step:
  - embedding (Dense+tanh, before the recurrent cell(s)): ordinary exact
    gradient (feedforward, single step — no approximation needed).
  - readout (head): ordinary exact gradient, same reasoning.
  - each recurrent cell's own weights: `learning_signal (x) local_trace`,
    where `local_trace` comes from the cell's own forward pass
    (`stream_rl.src.models.eprop_layers`) and `learning_signal` is the
    feedback-mode-dependent broadcast of the *readout-level* error
    (`d(loss)/d(readout_output)` — not the TD error itself, which is only
    ever applied once, uniformly, inside `_obgd_update`, exactly like stock
    `stream_ac.py`; folding it into the learning signal too would double
    -count it).

Feedback modes (the "connectivity matrix" from the project brief):
  - "symmetric": cell weights also get the ordinary exact gradient (no
    approximation at all — equivalent to stock StreamAC + these cells).
  - "random": a fixed matrix B per layer, sampled once at init from
    `feedback_seed`, broadcasts the readout-level error into that layer's
    hidden space (Bellec et al.'s reward-modulated / random e-prop).
  - "adaptive": every layer's B starts random and is nudged toward the true
    readout weights via a slow EMA (`feedback_lr`) -- for layers other than
    the last, this is a pragmatic approximation (there's no single "true"
    target for an intermediate layer's feedback matrix under multi-layer
    credit assignment; nudging every layer toward the head's kernel is the
    common shallow feedback-alignment simplification).

Stacking (`cfg.num_layers > 1`): `num_layers` identical cells (same `cell`
type, `hidden_size`, `trace_decay`, `activation`) are chained sequentially,
embed -> cell_0 -> cell_1 -> ... -> head. Each cell's own optional LayerNorm
(`use_layernorm`) already normalizes its output before it becomes the next
layer's input, so no separate inter-layer normalization is needed. Layer 0
consumes `embed_dim`-wide input; every later layer consumes the previous
layer's `hidden_size`-wide (post-LayerNorm) output.
"""
from typing import Any

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import struct

from memorax.utils import Timestep
from memorax.utils.axes import remove_feature_axis
from memorax.utils.typing import Array, Discrete, Environment, EnvParams, EnvState, Key, PyTree

from stream_rl.src.models.eprop_layers import build_eprop_cell


# --------------------------------------------------------------------------
# Small, transparent heads (unlike memorax.networks.heads.Categorical/
# Gaussian/VNetwork, these expose the raw pre-distribution output as a
# plain array, which is exactly the boundary the feedback matrix needs).
# --------------------------------------------------------------------------
class _DiscretePolicyHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, h: Array) -> Array:
        return nn.Dense(self.action_dim, name="out")(h)


class _ContinuousPolicyHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, h: Array) -> tuple[Array, Array]:
        mean = nn.Dense(self.action_dim, name="out")(h)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(jnp.clip(log_std, -5.0, 2.0))
        return mean, jnp.broadcast_to(std, mean.shape)


class _ValueHead(nn.Module):
    @nn.compact
    def __call__(self, h: Array) -> Array:
        return nn.Dense(1, name="out")(h)


@struct.dataclass(frozen=True)
class StreamEpropConfig:
    num_envs: int
    gamma: float
    embed_dim: int
    hidden_size: int
    cell: str = "eprop_gru"                # "eprop_rnn" | "eprop_gru" | "eprop_lstm"
    num_layers: int = 1                    # stack this many identical eprop cells sequentially
    activation: str = "tanh"
    trace_decay: float = 0.9               # e-prop cell's own eligibility trace decay
    use_layernorm: bool = True             # LayerNorm on each cell's readout-facing output (not the carried state)
    use_sparse_init: bool = True           # sparse init (Elsayed et al., 2024) for each cell's input-side kernels
    sparsity: float = 0.9
    trace_lambda: float = 0.9              # outer streaming (TD-lambda-style) trace, as in stock StreamAC
    actor_lr: float = 1.0
    critic_lr: float = 1.0
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.01
    adaptive: bool = False                 # ObGD's Adam-like (v_hat) step-size normalization
    beta2: float = 0.999
    eps: float = 1e-8
    feedback_mode: str = "symmetric"       # "symmetric" | "random" | "adaptive"
    feedback_seed: int = 0
    feedback_lr: float = 0.05              # only used when feedback_mode == "adaptive"


@struct.dataclass(frozen=True)
class StreamEpropState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: PyTree
    critic_params: PyTree
    actor_carry: Any                       # tuple of per-layer EpropCarry
    critic_carry: Any                      # tuple of per-layer EpropCarry
    actor_v: PyTree
    critic_v: PyTree
    actor_traces: PyTree
    critic_traces: PyTree
    actor_feedback: Any                    # tuple of per-layer feedback matrices
    critic_feedback: Any                   # tuple of per-layer feedback matrices


class StreamEprop:
    def __init__(self, cfg: StreamEpropConfig, env: Environment, env_params: EnvParams):
        self.cfg = cfg
        self.env = env
        self.env_params = env_params

        if cfg.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {cfg.num_layers!r}")

        action_space = env.action_space(env_params)
        self.is_discrete = isinstance(action_space, Discrete) or hasattr(action_space, "n")
        self.action_dim = int(action_space.n) if self.is_discrete else int(action_space.shape[0])

        self.actor_embed = nn.Dense(cfg.embed_dim, name="embed")
        self.critic_embed = nn.Dense(cfg.embed_dim, name="embed")

        def _build_cells():
            return [
                build_eprop_cell(
                    cfg.cell, cfg.hidden_size, cfg.trace_decay, cfg.activation,
                    use_layernorm=cfg.use_layernorm, use_sparse_init=cfg.use_sparse_init, sparsity=cfg.sparsity,
                )
                for _ in range(cfg.num_layers)
            ]

        self.actor_cells = _build_cells()
        self.critic_cells = _build_cells()
        self.actor_head = (
            _DiscretePolicyHead(self.action_dim) if self.is_discrete else _ContinuousPolicyHead(self.action_dim)
        )
        self.critic_head = _ValueHead()

        self._out_dim = self.action_dim  # width of the readout the feedback matrices broadcast from

    # ---- distribution helper ----
    def _make_dist(self, actor_out):
        if self.is_discrete:
            return distrax.Categorical(logits=actor_out)
        mean, std = actor_out
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    # ---- layer-stack forward pass ----
    def _layer_input_dims(self) -> list[int]:
        return [self.cfg.embed_dim] + [self.cfg.hidden_size] * (self.cfg.num_layers - 1)

    def _init_carries(self, cells: list) -> tuple:
        return tuple(
            cell.initialize_carry(None, (self.cfg.num_envs, in_dim))
            for cell, in_dim in zip(cells, self._layer_input_dims())
        )

    def _apply_stack(self, cells: list, params_list: list, carries_in: tuple, x: Array) -> tuple[tuple, Array]:
        h = x
        new_carries = []
        for cell, params, carry in zip(cells, params_list, carries_in):
            new_carry, h = cell.apply(params, h, initial_carry=carry)
            new_carries.append(new_carry)
        return tuple(new_carries), h

    # ---- init ----
    def init(self, key: Key) -> StreamEpropState:
        n = self.cfg.num_layers
        keys = list(jax.random.split(key, 7 + 2 * n))
        it = iter(keys)
        env_key, ae_key = next(it), next(it)
        ac_keys = [next(it) for _ in range(n)]
        ah_key, ce_key = next(it), next(it)
        cc_keys = [next(it) for _ in range(n)]
        ch_key, afb_key, cfb_key = next(it), next(it), next(it)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(env_keys, self.env_params)
        action = jnp.zeros((self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
                            dtype=self.env.action_space(self.env_params).dtype)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        actor_carry = self._init_carries(self.actor_cells)
        critic_carry = self._init_carries(self.critic_cells)

        actor_embed_params = self.actor_embed.init(ae_key, obs)
        x0 = self.actor_embed.apply(actor_embed_params, obs)
        actor_cell_params = []
        h = x0
        for cell, k, carry in zip(self.actor_cells, ac_keys, actor_carry):
            p = cell.init(k, h, initial_carry=carry)
            _, h = cell.apply(p, h, initial_carry=carry)
            actor_cell_params.append(p)
        actor_head_params = self.actor_head.init(ah_key, h)

        critic_embed_params = self.critic_embed.init(ce_key, obs)
        xc0 = self.critic_embed.apply(critic_embed_params, obs)
        critic_cell_params = []
        hc = xc0
        for cell, k, carry in zip(self.critic_cells, cc_keys, critic_carry):
            p = cell.init(k, hc, initial_carry=carry)
            _, hc = cell.apply(p, hc, initial_carry=carry)
            critic_cell_params.append(p)
        critic_head_params = self.critic_head.init(ch_key, hc)

        actor_params = {"embed": actor_embed_params, "cells": actor_cell_params, "head": actor_head_params}
        critic_params = {"embed": critic_embed_params, "cells": critic_cell_params, "head": critic_head_params}

        actor_traces = jax.tree.map(lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=jnp.float32), actor_params)
        critic_traces = jax.tree.map(lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=jnp.float32), critic_params)
        actor_v = jax.tree.map(jnp.zeros_like, actor_traces)
        critic_v = jax.tree.map(jnp.zeros_like, critic_traces)

        afb_keys = jax.random.split(afb_key, n)
        cfb_keys = jax.random.split(cfb_key, n)
        actor_feedback = tuple(
            jax.random.normal(k, (self.cfg.hidden_size, self._out_dim)) / jnp.sqrt(self.cfg.hidden_size)
            for k in afb_keys
        )
        critic_feedback = tuple(
            jax.random.normal(k, (self.cfg.hidden_size, 1)) / jnp.sqrt(self.cfg.hidden_size)
            for k in cfb_keys
        )

        return StreamEpropState(
            step=0, update_step=0, timestep=timestep, env_state=env_state,
            actor_params=actor_params, critic_params=critic_params,
            actor_carry=actor_carry, critic_carry=critic_carry,
            actor_v=actor_v, critic_v=critic_v,
            actor_traces=actor_traces, critic_traces=critic_traces,
            actor_feedback=actor_feedback, critic_feedback=critic_feedback,
        )

    # ---- per-role gradient computation ----
    def _role_grads(self, *, is_actor: bool, embed, cells, head, params, obs, carry_in, new_carry,
                     x, h, action, td_error, feedback):
        embed_params, cells_params, head_params = params["embed"], params["cells"], params["head"]
        num_layers = len(cells)

        def scalar_loss(out):
            if is_actor:
                dist = self._make_dist(out)
                return dist.log_prob(action) + self.cfg.entropy_coefficient * jnp.sign(td_error) * dist.entropy()
            return remove_feature_axis(out)

        def embed_loss(embed_params_):
            x_ = embed.apply(embed_params_, obs)
            _, h_ = self._apply_stack(cells, cells_params, carry_in, x_)
            return scalar_loss(head.apply(head_params, h_))

        def head_loss(head_params_):
            return scalar_loss(head.apply(head_params_, h))

        embed_grad = jax.jacobian(embed_loss)(embed_params)
        head_grad = jax.jacobian(head_loss)(head_params)

        if self.cfg.feedback_mode == "symmetric":
            cells_grad = []
            for i in range(num_layers):
                def cell_i_loss(cell_i_params, i=i):
                    layer_params = list(cells_params)
                    layer_params[i] = cell_i_params
                    _, h_ = self._apply_stack(cells, layer_params, carry_in, x)
                    return scalar_loss(head.apply(head_params, h_))

                cells_grad.append(jax.jacobian(cell_i_loss)(cells_params[i]))
            new_feedback = feedback
        else:
            # Output-space error e_readout[n] = d(loss_n)/d(primary_output_n),
            # where primary_output is the h-dependent part of the head's raw
            # output (logits for a discrete actor, mean for a continuous
            # actor, value for the critic). Computed WITHOUT the TD error —
            # delta is applied exactly once, uniformly, inside _obgd_update;
            # folding it in here too would double-count it.
            out = head.apply(head_params, h)
            if is_actor and not self.is_discrete:
                primary, std = out
            else:
                primary, std = out, jnp.zeros_like(out)

            def per_example_loss(primary_i, action_i, td_error_i, std_i):
                p_i = primary_i[None, ...]
                if is_actor:
                    dist = (
                        distrax.Categorical(logits=p_i)
                        if self.is_discrete
                        else distrax.MultivariateNormalDiag(loc=p_i, scale_diag=std_i[None, ...])
                    )
                    a_i = action_i[None, ...]
                    loss_i = dist.log_prob(a_i) + self.cfg.entropy_coefficient * jnp.sign(td_error_i[None]) * dist.entropy()
                else:
                    loss_i = remove_feature_axis(p_i)
                return loss_i[0]

            action_arg = action if is_actor else jnp.zeros((primary.shape[0],), dtype=jnp.int32)
            e_readout = jax.vmap(jax.grad(per_example_loss, argnums=0), in_axes=(0, 0, 0, 0))(
                primary, action_arg, td_error, std
            )

            def trace_based_cell_grad(cell_params_dict, traces, learning_signal):
                grad = {}
                for pname, pval in cell_params_dict.items():
                    trace_key = "e_" + pname
                    if trace_key not in traces:
                        # Params with no e-prop eligibility trace (e.g. the
                        # cell's optional LayerNorm scale/bias, added by
                        # `use_layernorm` -- LayerNorm has no temporal
                        # dependency, so there's nothing to broadcast the
                        # learning signal through here). Left untrained under
                        # random/adaptive feedback; `symmetric` mode doesn't
                        # go through this path at all (uses jax.jacobian over
                        # every cell param instead, LayerNorm included).
                        grad[pname] = jax.tree.map(jnp.zeros_like, pval)
                        continue
                    trace = traces[trace_key]
                    if pval.ndim == 2:
                        grad[pname] = trace * learning_signal[:, None, :]
                    else:
                        grad[pname] = trace * learning_signal
                return grad

            cells_grad = []
            new_feedback_list = []
            for i in range(num_layers):
                fb_i = feedback[i]
                learning_signal_i = jnp.einsum("do,no->nd", fb_i, e_readout)
                cells_grad.append({
                    "params": trace_based_cell_grad(cells_params[i]["params"], new_carry[i].traces, learning_signal_i)
                })
                if self.cfg.feedback_mode == "adaptive":
                    # No single "true" target exists for an intermediate
                    # layer's feedback matrix; nudging every layer toward the
                    # head's own kernel (shape-compatible since all layers
                    # share hidden_size) is the standard shallow
                    # feedback-alignment simplification for this case.
                    out_kernel = head_params["params"]["out"]["kernel"]
                    new_feedback_list.append((1.0 - self.cfg.feedback_lr) * fb_i + self.cfg.feedback_lr * out_kernel)
                else:
                    new_feedback_list.append(fb_i)
            new_feedback = tuple(new_feedback_list)

        grads = {"embed": embed_grad, "cells": cells_grad, "head": head_grad}
        return grads, new_feedback

    # ---- ObGD (copied from memorax.algorithms.stream_ac.StreamAC._obgd_update) ----
    def _obgd_update(self, traces: PyTree, v: PyTree, td_error: Array, lr: float, kappa: float, step: int):
        beta2 = self.cfg.beta2
        eps = self.cfg.eps

        def _broadcast_delta(td_error, z):
            n_trailing = z.ndim - 1
            return td_error[(slice(None),) + (None,) * n_trailing]

        new_v = jax.tree.map(
            lambda vi, z: beta2 * vi + (1 - beta2) * jnp.square(_broadcast_delta(td_error, z) * z), v, traces,
        )

        if self.cfg.adaptive:
            v_hat = jax.tree.map(lambda vi: vi / (1.0 - beta2 ** step), new_v)
            norm_leaves = jax.tree.leaves(jax.tree.map(
                lambda z, vh: jnp.abs(z) / (jnp.sqrt(vh) + eps), traces, v_hat,
            ))
            z_sum = sum(jnp.sum(z, axis=tuple(range(1, z.ndim))) for z in norm_leaves)
        else:
            v_hat = None
            z_leaves = jax.tree.leaves(traces)
            z_sum = sum(jnp.sum(jnp.abs(z), axis=tuple(range(1, z.ndim))) for z in z_leaves)

        delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        step_size = lr / jnp.maximum(1.0, delta_bar * z_sum * lr * kappa)

        if self.cfg.adaptive:
            def compute_update(z, vh):
                n_trailing = z.ndim - 1
                ss = step_size[(slice(None),) + (None,) * n_trailing]
                delta = td_error[(slice(None),) + (None,) * n_trailing]
                return (ss * delta * z / (jnp.sqrt(vh) + eps)).mean(axis=0)
            updates = jax.tree.map(compute_update, traces, v_hat)
        else:
            def compute_update(z):
                n_trailing = z.ndim - 1
                ss = step_size[(slice(None),) + (None,) * n_trailing]
                delta = td_error[(slice(None),) + (None,) * n_trailing]
                return (ss * delta * z).mean(axis=0)
            updates = jax.tree.map(compute_update, traces)

        return updates, new_v

    # ---- action selection (deterministic, for evaluate()) ----
    def _deterministic_action(self, key, state):
        x = self.actor_embed.apply(state.actor_params["embed"], state.timestep.obs)
        actor_carry, h = self._apply_stack(self.actor_cells, state.actor_params["cells"], state.actor_carry, x)
        out = self.actor_head.apply(state.actor_params["head"], h)
        dist = self._make_dist(out)
        action = jnp.argmax(out, axis=-1) if self.is_discrete else dist.mode()
        state = state.replace(actor_carry=actor_carry)
        return state, action

    def _env_step(self, state, key):
        action_key, step_key = jax.random.split(key)
        state, action = self._deterministic_action(action_key, state)
        num_envs = state.timestep.obs.shape[0]
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            step_keys, state.env_state, action, self.env_params
        )
        lox.log({"info": info})
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            env_state=env_state,
        )
        return state, None

    def _update_step(self, state: StreamEpropState, key: Key):
        action_key, step_key = jax.random.split(key)
        obs = state.timestep.obs
        prev_done = state.timestep.done

        actor_carry_in = jax.lax.stop_gradient(state.actor_carry)
        critic_carry_in = jax.lax.stop_gradient(state.critic_carry)

        x_actor = self.actor_embed.apply(state.actor_params["embed"], obs)
        new_actor_carry, h_actor = self._apply_stack(self.actor_cells, state.actor_params["cells"], actor_carry_in, x_actor)
        actor_out = self.actor_head.apply(state.actor_params["head"], h_actor)
        dist = self._make_dist(actor_out)
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        entropy = dist.entropy().mean()

        x_critic = self.critic_embed.apply(state.critic_params["embed"], obs)
        new_critic_carry, h_critic = self._apply_stack(self.critic_cells, state.critic_params["cells"], critic_carry_in, x_critic)
        value = remove_feature_axis(self.critic_head.apply(state.critic_params["head"], h_critic))

        num_envs = obs.shape[0]
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        critic_params_sg = jax.lax.stop_gradient(state.critic_params)
        x_critic_next = self.critic_embed.apply(critic_params_sg["embed"], next_obs)
        _, h_critic_next = self._apply_stack(
            self.critic_cells, critic_params_sg["cells"], jax.lax.stop_gradient(new_critic_carry), x_critic_next
        )
        next_value = remove_feature_axis(self.critic_head.apply(critic_params_sg["head"], h_critic_next))

        gamma = self.cfg.gamma
        td_error = next_reward + gamma * (1 - next_done) * next_value - value

        actor_grads, new_actor_feedback = self._role_grads(
            is_actor=True, embed=self.actor_embed, cells=self.actor_cells, head=self.actor_head,
            params=state.actor_params, obs=obs, carry_in=actor_carry_in, new_carry=new_actor_carry,
            x=x_actor, h=h_actor, action=action, td_error=td_error, feedback=state.actor_feedback,
        )
        critic_grads, new_critic_feedback = self._role_grads(
            is_actor=False, embed=self.critic_embed, cells=self.critic_cells, head=self.critic_head,
            params=state.critic_params, obs=obs, carry_in=critic_carry_in, new_carry=new_critic_carry,
            x=x_critic, h=h_critic, action=None, td_error=td_error, feedback=state.critic_feedback,
        )

        outer_trace_decay = gamma * self.cfg.trace_lambda

        def update_outer_trace(z, g):
            n_trailing = z.ndim - 1
            not_done = (1 - prev_done)[(slice(None),) + (None,) * n_trailing]
            return outer_trace_decay * not_done * z + g

        new_actor_traces = jax.tree.map(update_outer_trace, state.actor_traces, actor_grads)
        new_critic_traces = jax.tree.map(update_outer_trace, state.critic_traces, critic_grads)

        current_step = state.update_step + 1
        critic_updates, critic_v = self._obgd_update(
            new_critic_traces, state.critic_v, td_error, self.cfg.critic_lr, self.cfg.critic_kappa, current_step
        )
        actor_updates, actor_v = self._obgd_update(
            new_actor_traces, state.actor_v, td_error, self.cfg.actor_lr, self.cfg.actor_kappa, current_step
        )

        new_critic_params = jax.tree.map(lambda p, u: p + u, state.critic_params, critic_updates)
        new_actor_params = jax.tree.map(lambda p, u: p + u, state.actor_params, actor_updates)

        lox.log({
            "info": info,
            "critic/td_error": td_error.mean(),
            "actor/entropy": entropy,
            "critic/value": value.mean(),
            "actor/feedback_norm": sum(jnp.linalg.norm(fb) for fb in new_actor_feedback),
            "critic/feedback_norm": sum(jnp.linalg.norm(fb) for fb in new_critic_feedback),
        })

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            update_step=current_step,
            timestep=Timestep(obs=next_obs, action=action, reward=next_reward, done=next_done),
            env_state=env_state,
            actor_params=new_actor_params, actor_traces=new_actor_traces, actor_v=actor_v,
            actor_carry=new_actor_carry, actor_feedback=new_actor_feedback,
            critic_params=new_critic_params, critic_traces=new_critic_traces, critic_v=critic_v,
            critic_carry=new_critic_carry, critic_feedback=new_critic_feedback,
        )
        return state, None

    def train(self, key: Key, state: StreamEpropState, num_steps: int) -> StreamEpropState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(self._update_step, state, keys)
        return state

    def warmup(self, key: Key, state: StreamEpropState, num_steps: int) -> StreamEpropState:
        return state

    def evaluate(self, key: Key, state: StreamEpropState, num_steps: int) -> StreamEpropState:
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_keys, self.env_params)
        action = jnp.zeros((self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
                            dtype=self.env.action_space(self.env_params).dtype)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        actor_carry = self._init_carries(self.actor_cells)
        critic_carry = self._init_carries(self.critic_cells)

        state = state.replace(
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done),
            env_state=env_state, actor_carry=actor_carry, critic_carry=critic_carry,
        )
        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(self._env_step, state, step_keys)
        return state
