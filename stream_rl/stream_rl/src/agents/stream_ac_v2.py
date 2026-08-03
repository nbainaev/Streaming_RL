"""StreamAC variant with a direct, always-positive entropy update.

Stock Memorax inserts ``sign(delta) * beta * grad(H)`` into the actor's
eligibility trace and later multiplies the whole trace by another TD error.
Once that trace persists, old entropy gradients can therefore receive either
sign.  Here only ``grad(log pi)`` enters the TD(lambda) trace; ``grad(H)`` is
applied immediately with its own ObGD-style bound.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import lox

from memorax.algorithms.stream_ac import StreamAC, StreamACState
from memorax.utils import Timestep, Transition
from memorax.utils.axes import add_time_axis, remove_feature_axis, remove_time_axis


class DirectEntropyStreamAC(StreamAC):
    def _bounded_entropy_update(self, grads: Any):
        coefficient = self.cfg.entropy_coefficient
        effective_lr = self.cfg.actor_lr * coefficient
        leaves = jax.tree.leaves(grads)
        grad_sum = sum(
            jnp.sum(jnp.abs(g), axis=tuple(range(1, g.ndim))) for g in leaves
        )
        step_size = effective_lr / jnp.maximum(
            1.0, effective_lr * self.cfg.actor_kappa * grad_sum
        )

        def update(g):
            trailing = g.ndim - 1
            scale = step_size[(slice(None),) + (None,) * trailing]
            return (scale * g).mean(axis=0)

        return jax.tree.map(update, grads)

    def _update_step(self, state: StreamACState, key):
        action_key, step_key, actor_torso_key, critic_torso_key = jax.random.split(
            key, 4
        )
        obs, done, ts_action, reward = state.timestep.to_sequence()

        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            action=ts_action,
            reward=reward,
            done=done,
            initial_carry=state.actor_carry,
            rngs={"torso": actor_torso_key},
            mutable=["intermediates"],
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)
        entropy = remove_time_axis(probs.entropy()).mean()
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        critic_carry, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=obs,
            action=ts_action,
            reward=reward,
            done=done,
            initial_carry=state.critic_carry,
            rngs={"torso": critic_torso_key},
        )
        value = remove_feature_axis(remove_time_axis(value))

        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        next_sequence = Timestep(
            obs=next_obs, action=action, reward=next_reward, done=next_done
        ).to_sequence()
        next_obs_s, next_done_s, next_action_s, next_reward_s = next_sequence
        _, (next_value, _) = self.critic_network.apply(
            jax.lax.stop_gradient(state.critic_params),
            observation=next_obs_s,
            action=next_action_s,
            reward=next_reward_s,
            done=next_done_s,
            initial_carry=jax.lax.stop_gradient(critic_carry),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value))
        td_error = (
            next_reward
            + self.cfg.gamma * (1 - next_done) * next_value
            - value
        )

        initial_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        initial_critic_carry = jax.lax.stop_gradient(state.critic_carry)

        def critic_loss_fn(params):
            _, (v, _) = self.critic_network.apply(
                params,
                observation=obs,
                action=ts_action,
                reward=reward,
                done=done,
                initial_carry=initial_critic_carry,
            )
            return remove_feature_axis(remove_time_axis(v))

        def actor_score_fn(params):
            _, (dist, _) = self.actor_network.apply(
                params,
                observation=obs,
                action=ts_action,
                reward=reward,
                done=done,
                initial_carry=initial_actor_carry,
            )
            return remove_time_axis(dist.log_prob(add_time_axis(action)))

        def actor_entropy_fn(params):
            _, (dist, _) = self.actor_network.apply(
                params,
                observation=obs,
                action=ts_action,
                reward=reward,
                done=done,
                initial_carry=initial_actor_carry,
            )
            return remove_time_axis(dist.entropy())

        critic_grads = jax.jacobian(critic_loss_fn)(state.critic_params)
        actor_grads = jax.jacobian(actor_score_fn)(state.actor_params)
        entropy_grads = jax.jacobian(actor_entropy_fn)(state.actor_params)

        trace_decay = self.cfg.gamma * self.cfg.trace_lambda

        def update_trace(trace, grad):
            trailing = trace.ndim - 1
            active = (1 - state.timestep.done)[
                (slice(None),) + (None,) * trailing
            ]
            return trace_decay * active * trace + grad

        critic_traces = jax.tree.map(
            update_trace, state.critic_traces, critic_grads
        )
        actor_traces = jax.tree.map(
            update_trace, state.actor_traces, actor_grads
        )
        current_step = state.update_step + 1
        critic_updates, critic_v = self._obgd_update(
            critic_traces,
            state.critic_v,
            td_error,
            self.cfg.critic_lr,
            self.cfg.critic_kappa,
            current_step,
        )
        policy_updates, actor_v = self._obgd_update(
            actor_traces,
            state.actor_v,
            td_error,
            self.cfg.actor_lr,
            self.cfg.actor_kappa,
            current_step,
        )
        entropy_updates = self._bounded_entropy_update(entropy_grads)
        actor_updates = jax.tree.map(
            lambda policy, exploration: policy + exploration,
            policy_updates,
            entropy_updates,
        )
        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.stack(x)),
            intermediates.get("intermediates", {}),
            is_leaf=lambda x: isinstance(x, tuple),
        )
        lox.log(
            {
                "info": info,
                "intermediates": intermediates,
                "critic/td_error": td_error.mean(),
                "actor/entropy": entropy,
                "critic/value": value.mean(),
            }
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        first = Timestep(
            obs=state.timestep.obs,
            action=state.timestep.action,
            reward=state.timestep.reward,
            done=state.timestep.done,
        )
        second = Timestep(
            obs=None, action=action, reward=next_reward, done=next_done
        )
        transition = Transition(
            first=first,
            second=second,
            aux={"log_prob": log_prob, "value": value},
        )
        del transition

        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            update_step=current_step,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    jnp.expand_dims(next_done, axis=broadcast_dims),
                    jnp.zeros_like(action),
                    action,
                ),
                reward=jnp.where(
                    next_done, jnp.zeros_like(next_reward_f), next_reward_f
                ),
                done=next_done,
            ),
            env_state=env_state,
            actor_params=actor_params,
            actor_traces=actor_traces,
            actor_v=actor_v,
            actor_carry=actor_carry,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
        )
        return state, None
