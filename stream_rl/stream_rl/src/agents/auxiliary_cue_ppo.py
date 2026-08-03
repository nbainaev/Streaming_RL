"""PPO with privileged cue supervision applied only to downstream readouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import lox
import optax

from memorax.algorithms.ppo import PPO, PPOState
from memorax.utils import Timestep, Transition, utils
from memorax.utils.axes import remove_feature_axis
from memorax.utils.typing import Array, Carry, Key, PyTree


@dataclass
class AuxiliaryCuePPO(PPO):
    """Adds a cue-reconstruction loss without updating frozen memory weights.

    ``info['goal_y']`` is used only as a training target.  The policy never
    receives it as input, and evaluation remains unchanged.
    """

    auxiliary_cue_coefficient: float = 0.1

    def _step(
        self, state: PPOState, key: Key, *, policy: Callable
    ) -> tuple[PPOState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value, intermediates = policy(action_key, state)

        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.stack(x)),
            intermediates.get("intermediates", {}),
            is_leaf=lambda x: isinstance(x, tuple),
        )
        if "goal_y" not in info:
            raise ValueError(
                "auxiliary cue training requires env info['goal_y']; "
                "use it only on compatible memory environments"
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
        second = Timestep(obs=None, action=action, reward=reward, done=done)
        lox.log({"info": info, "intermediates": intermediates})
        transition = Transition(
            first=first,
            second=second,
            aux={
                "log_prob": log_prob,
                "value": value,
                "cue_target": jnp.asarray(info["goal_y"], dtype=jnp.float32),
            },
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    jnp.expand_dims(done, axis=broadcast_dims),
                    jnp.zeros_like(action),
                    action,
                ),
                reward=jnp.where(done, 0, jnp.asarray(reward, dtype=jnp.float32)),
                done=done,
            ),
            env_state=env_state,
        )
        return state, transition

    def _update_actor(
        self,
        key: Key,
        state: PPOState,
        initial_actor_carry: Carry,
        transitions: Transition,
    ) -> tuple[PPOState, Array, tuple[Array, Array, Array]]:
        torso_key, dropout_key = jax.random.split(key)
        initial_actor_carry = utils.burn_in(
            self.actor_network,
            state.actor_params,
            transitions.first,
            initial_actor_carry,
            self.cfg.burn_in_length,
        )
        transitions = jax.tree.map(
            lambda x: x[:, self.cfg.burn_in_length :], transitions
        )
        advantages = transitions.aux["advantages"]
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_loss_fn(params: PyTree):
            _, (probs, head_aux) = self.actor_network.apply(
                params,
                *transitions.first,
                initial_carry=initial_actor_carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            cue_prediction = head_aux["cue_prediction"]
            cue_target = transitions.aux["cue_target"]
            cue_loss = 0.5 * jnp.square(cue_prediction - cue_target).mean()
            cue_accuracy = jnp.mean(
                (jnp.sign(cue_prediction) == cue_target).astype(jnp.float32)
            )

            log_probs = probs.log_prob(transitions.second.action)
            entropy = probs.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = jnp.mean(transitions.aux["log_prob"] - log_probs)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
            )
            policy_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()
            total = (
                policy_loss
                - self.cfg.entropy_coefficient * entropy
                + self.auxiliary_cue_coefficient * cue_loss
            )
            return total, (
                entropy,
                approximate_kl,
                clip_fraction,
                cue_loss,
                cue_accuracy,
            )

        (actor_loss, aux), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        entropy, approximate_kl, clip_fraction, cue_loss, cue_accuracy = aux
        lox.log(
            {
                "actor/gradient_norm": optax.global_norm(actor_grads),
                "aux/actor_cue_loss": cue_loss,
                "aux/actor_cue_accuracy": cue_accuracy,
            }
        )
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)
        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
        )
        return state, actor_loss.mean(), (
            entropy.mean(),
            approximate_kl.mean(),
            clip_fraction.mean(),
        )

    def _update_critic(
        self,
        key: Key,
        state: PPOState,
        initial_critic_carry: Carry,
        transitions: Transition,
    ) -> tuple[PPOState, Array]:
        torso_key, dropout_key = jax.random.split(key)
        initial_critic_carry = utils.burn_in(
            self.critic_network,
            state.critic_params,
            transitions.first,
            initial_critic_carry,
            self.cfg.burn_in_length,
        )
        transitions = jax.tree.map(
            lambda x: x[:, self.cfg.burn_in_length :], transitions
        )
        returns = transitions.aux["returns"]

        def critic_loss_fn(params: PyTree):
            _, (values, head_aux) = self.critic_network.apply(
                params,
                *transitions.first,
                initial_carry=initial_critic_carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            values = remove_feature_axis(values)
            critic_loss = self.critic_network.head.loss(
                values, head_aux, returns, transitions=transitions
            )
            if self.cfg.clip_value_loss:
                clipped_value = transitions.aux["value"] + jnp.clip(
                    values - transitions.aux["value"],
                    -self.cfg.clip_coefficient,
                    self.cfg.clip_coefficient,
                )
                clipped_loss = self.critic_network.head.loss(
                    clipped_value, head_aux, returns, transitions=transitions
                )
                critic_loss = jnp.maximum(critic_loss, clipped_loss)
            critic_loss = critic_loss.mean()

            cue_prediction = head_aux["cue_prediction"]
            cue_target = transitions.aux["cue_target"]
            cue_loss = 0.5 * jnp.square(cue_prediction - cue_target).mean()
            cue_accuracy = jnp.mean(
                (jnp.sign(cue_prediction) == cue_target).astype(jnp.float32)
            )
            total = critic_loss + self.auxiliary_cue_coefficient * cue_loss
            return total, (values, cue_loss, cue_accuracy)

        (critic_loss, aux), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic_params)
        values, cue_loss, cue_accuracy = aux
        explained_variance = 1 - jnp.var(returns - values) / jnp.maximum(
            jnp.var(returns), 1e-8
        )
        lox.log(
            {
                "critic/gradient_norm": optax.global_norm(critic_grads),
                "critic/explained_variance": explained_variance,
                "critic/value": values.mean(),
                "aux/critic_cue_loss": cue_loss,
                "aux/critic_cue_accuracy": cue_accuracy,
            }
        )
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)
        state = state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return state, critic_loss.mean()
