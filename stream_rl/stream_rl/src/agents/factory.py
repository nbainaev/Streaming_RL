"""Builds PPO / StreamAC / StreamEprop agent instances from config + env."""
import optax

from memorax.algorithms.ppo import PPO, PPOConfig
from memorax.algorithms.stream_ac import StreamAC, StreamACConfig

from stream_rl.src.agents.stream_eprop import StreamEprop, StreamEpropConfig
from stream_rl.src.models.networks import build_actor_network, build_critic_network

def build_stream_eprop(cfg: dict, env, env_params):
    stream_eprop_cfg = StreamEpropConfig(
        num_envs=cfg["num_envs"],
        gamma=cfg["gamma"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        cell=cfg.get("cell", "eprop_gru"),
        activation=cfg.get("activation", "tanh"),
        trace_decay=cfg.get("trace_decay", 0.9),
        trace_lambda=cfg.get("trace_lambda", 0.9),
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        actor_kappa=cfg.get("actor_kappa", 3.0),
        critic_kappa=cfg.get("critic_kappa", 2.0),
        entropy_coefficient=cfg["entropy_coefficient"],
        adaptive=cfg.get("adaptive", False),
        feedback_mode=cfg.get("feedback_mode", "symmetric"),
        feedback_seed=cfg.get("feedback_seed", 0),
        feedback_lr=cfg.get("feedback_lr", 0.05),
    )

    return StreamEprop(cfg=stream_eprop_cfg, env=env, env_params=env_params)

def build_ppo(cfg: dict, env, env_params):
    observation_space = env.observation_space(env_params)
    actor_network = build_actor_network(
        architecture_cfg=cfg["actor_architecture"],
        action_dim=env.num_actions,
        embed_dim=cfg["embed_dim"],
        observation_space=observation_space,
    )
    critic_network = build_critic_network(
        architecture_cfg=cfg["critic_architecture"],
        embed_dim=cfg["embed_dim"],
        observation_space=observation_space,
    )

    ppo_cfg = PPOConfig(
        num_envs=cfg["num_envs"],
        num_steps=cfg["num_steps"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        num_minibatches=cfg["num_minibatches"],
        update_epochs=cfg["update_epochs"],
        normalize_advantage=cfg["normalize_advantage"],
        clip_coefficient=cfg["clip_coefficient"],
        clip_value_loss=cfg["clip_value_loss"],
        entropy_coefficient=cfg["entropy_coefficient"],
    )

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["actor_lr"]),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["critic_lr"]),
    )

    return PPO(
        cfg=ppo_cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )


def build_stream_ac(cfg: dict, env, env_params):
    tbptt_steps = cfg.get("tbptt_steps", 1)   # извлечение
    observation_space = env.observation_space(env_params)

    actor_network = build_actor_network(
        architecture_cfg=cfg["actor_architecture"],
        action_dim=env.num_actions,
        embed_dim=cfg["embed_dim"],
        tbptt_steps=tbptt_steps,          # передаём
        observation_space=observation_space,
    )
    critic_network = build_critic_network(
        architecture_cfg=cfg["critic_architecture"],
        embed_dim=cfg["embed_dim"],
        tbptt_steps=tbptt_steps,          # передаём
        observation_space=observation_space,
    )

    stream_ac_cfg = StreamACConfig(
        num_envs=cfg["num_envs"],
        gamma=cfg["gamma"],
        trace_lambda=cfg["trace_lambda"],
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        actor_kappa=cfg.get("actor_kappa", 3.0),
        critic_kappa=cfg.get("critic_kappa", 2.0),
        entropy_coefficient=cfg["entropy_coefficient"],
        adaptive=cfg.get("adaptive", False),
    )

    return StreamAC(
        cfg=stream_ac_cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
    )

BUILDERS = {
    "ppo": build_ppo,
    "stream_ac": build_stream_ac,
    "stream_eprop": build_stream_eprop,  # новый
}


def build_agent(name: str, cfg: dict, env, env_params):
    if name not in BUILDERS:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(BUILDERS)}")
    return BUILDERS[name](cfg, env, env_params)