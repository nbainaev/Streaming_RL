"""Windowed AC(lambda) with a genuine truncated recurrent gradient horizon.

The stock Memorax StreamAC detaches the recurrent carry before every loss.
Changing ``RNN.unroll`` therefore cannot turn it into TBPTT(k).  This class
collects a fixed k-step on-policy window, replays that whole sequence from a
detached window-start carry, and forms per-time Jacobians through the window.
Parameters are updated once per window while the outer AC(lambda) eligibility
trace is advanced in temporal order.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox

from memorax.algorithms.stream_ac import StreamAC, StreamACState
from memorax.utils.axes import add_time_axis, remove_feature_axis, remove_time_axis


def _batch_time(value):
    """Convert lax.scan's [time, batch, ...] output to [batch, time, ...]."""
    return jnp.swapaxes(value, 0, 1)


@dataclass
class WindowedStreamAC(StreamAC):
    window: int = 5

    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.cfg.num_envs != 1:
            raise ValueError("WindowedStreamAC is validated only for num_envs=1")

    def _update_window(self, state: StreamACState, key):
        initial_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        initial_critic_carry = jax.lax.stop_gradient(state.critic_carry)
        rollout_keys = jax.random.split(key, self.window)

        def rollout_step(current, step_key):
            return self._step(current, step_key, policy=self._stochastic_action)

        rolled, transitions = jax.lax.scan(rollout_step, state, rollout_keys)

        observations = _batch_time(transitions.first.obs)
        input_actions = _batch_time(transitions.first.action)
        input_rewards = _batch_time(transitions.first.reward)
        input_done = _batch_time(transitions.first.done)
        policy_actions = _batch_time(transitions.second.action)
        next_rewards = _batch_time(transitions.second.reward)
        next_done = _batch_time(transitions.second.done)

        actor_carry, (distribution, _) = self.actor_network.apply(
            state.actor_params,
            observation=observations,
            action=input_actions,
            reward=input_rewards,
            done=input_done,
            initial_carry=initial_actor_carry,
        )
        critic_carry, (values, _) = self.critic_network.apply(
            state.critic_params,
            observation=observations,
            action=input_actions,
            reward=input_rewards,
            done=input_done,
            initial_carry=initial_critic_carry,
        )
        values = remove_feature_axis(values)

        next_ts = rolled.timestep
        _, (bootstrap, _) = self.critic_network.apply(
            jax.lax.stop_gradient(state.critic_params),
            observation=add_time_axis(next_ts.obs),
            action=add_time_axis(next_ts.action),
            reward=add_time_axis(next_ts.reward),
            done=add_time_axis(next_ts.done),
            initial_carry=jax.lax.stop_gradient(critic_carry),
        )
        bootstrap = remove_feature_axis(remove_time_axis(bootstrap))
        next_values = jnp.concatenate([values[:, 1:], bootstrap[:, None]], axis=1)
        td_error = next_rewards + self.cfg.gamma * (1.0 - next_done) * next_values - values
        td_error = jax.lax.stop_gradient(td_error)

        def critic_objective(params):
            _, (candidate, _) = self.critic_network.apply(
                params,
                observation=observations,
                action=input_actions,
                reward=input_rewards,
                done=input_done,
                initial_carry=initial_critic_carry,
            )
            return remove_feature_axis(candidate)

        def actor_objective(params):
            _, (candidate, _) = self.actor_network.apply(
                params,
                observation=observations,
                action=input_actions,
                reward=input_rewards,
                done=input_done,
                initial_carry=initial_actor_carry,
            )
            return candidate.log_prob(policy_actions) + self.cfg.entropy_coefficient * jnp.sign(td_error) * candidate.entropy()

        critic_grads = jax.jacobian(critic_objective)(state.critic_params)
        actor_grads = jax.jacobian(actor_objective)(state.actor_params)
        critic_grads = jax.tree.map(_batch_time, critic_grads)
        actor_grads = jax.tree.map(_batch_time, actor_grads)
        done_time = jnp.swapaxes(input_done, 0, 1)
        td_time = jnp.swapaxes(td_error, 0, 1)

        zero_actor_update = jax.tree.map(jnp.zeros_like, state.actor_params)
        zero_critic_update = jax.tree.map(jnp.zeros_like, state.critic_params)
        trace_decay = self.cfg.gamma * self.cfg.trace_lambda

        def trace_step(carry, xs):
            actor_trace, critic_trace, actor_v, critic_v, actor_sum, critic_sum, update_step = carry
            actor_grad, critic_grad, done_t, delta_t = xs

            def advance(trace, grad):
                not_done = (1.0 - done_t)[
                    (slice(None),) + (None,) * (trace.ndim - 1)
                ]
                return trace_decay * not_done * trace + grad

            actor_trace = jax.tree.map(advance, actor_trace, actor_grad)
            critic_trace = jax.tree.map(advance, critic_trace, critic_grad)
            update_step = update_step + 1
            actor_update, actor_v = self._obgd_update(
                actor_trace,
                actor_v,
                delta_t,
                self.cfg.actor_lr,
                self.cfg.actor_kappa,
                update_step,
            )
            critic_update, critic_v = self._obgd_update(
                critic_trace,
                critic_v,
                delta_t,
                self.cfg.critic_lr,
                self.cfg.critic_kappa,
                update_step,
            )
            actor_sum = jax.tree.map(jnp.add, actor_sum, actor_update)
            critic_sum = jax.tree.map(jnp.add, critic_sum, critic_update)
            return (
                actor_trace,
                critic_trace,
                actor_v,
                critic_v,
                actor_sum,
                critic_sum,
                update_step,
            ), None

        trace_state = (
            state.actor_traces,
            state.critic_traces,
            state.actor_v,
            state.critic_v,
            zero_actor_update,
            zero_critic_update,
            state.update_step,
        )
        trace_state, _ = jax.lax.scan(
            trace_step,
            trace_state,
            (actor_grads, critic_grads, done_time, td_time),
        )
        (
            actor_traces,
            critic_traces,
            actor_v,
            critic_v,
            actor_updates,
            critic_updates,
            update_step,
        ) = trace_state

        actor_params = jax.tree.map(jnp.add, state.actor_params, actor_updates)
        critic_params = jax.tree.map(jnp.add, state.critic_params, critic_updates)
        entropy = distribution.entropy().mean()
        lox.log(
            {
                "critic/td_error": td_error.mean(),
                "critic/value": values.mean(),
                "actor/entropy": entropy,
                "tbptt/window": jnp.asarray(self.window, dtype=jnp.float32),
            }
        )
        return rolled.replace(
            update_step=update_step,
            actor_params=actor_params,
            actor_traces=actor_traces,
            actor_v=actor_v,
            actor_carry=actor_carry,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
        ), None

    def train(self, key, state: StreamACState, num_steps: int) -> StreamACState:
        if num_steps % self.window:
            raise ValueError(
                f"num_steps={num_steps} must be divisible by TBPTT window={self.window}"
            )
        keys = jax.random.split(key, num_steps // self.window)
        state, _ = jax.lax.scan(self._update_window, state, keys)
        return state
