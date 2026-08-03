"""Builds PPO / StreamAC / StreamEprop agent instances from config + env."""
import optax

from memorax.algorithms.ppo import PPO, PPOConfig
from memorax.algorithms.stream_ac import StreamAC, StreamACConfig

from stream_rl.src.agents.stream_eprop import StreamEprop, StreamEpropConfig
from stream_rl.src.agents.stream_ac_v2 import DirectEntropyStreamAC
from stream_rl.src.agents.auxiliary_cue_ppo import AuxiliaryCuePPO
from stream_rl.src.agents.auxiliary_cue_stream_ac import AuxiliaryCueStreamAC
from stream_rl.src.agents.stream_tbptt import WindowedStreamAC
from stream_rl.src.models.networks import build_actor_network, build_critic_network

def build_stream_eprop(cfg: dict, env, env_params):
    stream_eprop_cfg = StreamEpropConfig(
        num_envs=cfg["num_envs"],
        gamma=cfg["gamma"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        cell=cfg.get("cell", "eprop_gru"),
        num_layers=cfg.get("num_layers", 1),
        activation=cfg.get("activation", "tanh"),
        trace_decay=cfg.get("trace_decay", 0.9),
        use_layernorm=cfg.get("use_layernorm", True),
        use_sparse_init=cfg.get("use_sparse_init", True),
        sparsity=cfg.get("sparsity", 0.9),
        trace_lambda=cfg.get("trace_lambda", 0.9),
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        actor_kappa=cfg.get("actor_kappa", 3.0),
        critic_kappa=cfg.get("critic_kappa", 2.0),
        entropy_coefficient=cfg["entropy_coefficient"],
        adaptive=cfg.get("adaptive", False),
        beta2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-8),
        feedback_mode=cfg.get("feedback_mode", "symmetric"),
        feedback_seed=cfg.get("feedback_seed", 0),
        feedback_lr=cfg.get("feedback_lr", 0.05),
        head_hidden_sizes=tuple(cfg.get("head_hidden_sizes", ())),
        head_activation=cfg.get("head_activation", "tanh"),
        frozen_ssm=cfg.get("frozen_ssm", False),
        frozen_ssm_checkpoint=cfg.get("frozen_ssm_checkpoint"),
        frozen_encoder=cfg.get("frozen_encoder", False),
        memory_seed=cfg.get("memory_seed", 0),
        critic_memory_seed=cfg.get("critic_memory_seed", cfg.get("memory_seed", 0) + 1),
        ssm_features=cfg.get("ssm_features", cfg.get("embed_dim", 64)),
        ssm_state_dim=cfg.get("ssm_state_dim", 128),
        ssm_concatenate_input=cfg.get("ssm_concatenate_input", True),
    )

    return StreamEprop(cfg=stream_eprop_cfg, env=env, env_params=env_params)

def build_ppo(cfg: dict, env, env_params):
    observation_space = env.observation_space(env_params)
    readout_only = bool(cfg.get("readout_only", False))
    frozen_encoder = bool(cfg.get("frozen_encoder", False))
    memory_seed = int(cfg.get("memory_seed", 0))
    critic_memory_seed = int(cfg.get("critic_memory_seed", memory_seed + 1))
    auxiliary_cue = float(cfg.get("auxiliary_cue_coefficient", 0.0)) > 0.0
    auxiliary_readout_dim = int(cfg.get("auxiliary_readout_dim", 32))
    preserve_observation_prefix = int(cfg.get("preserve_observation_prefix", 0))
    actor_network = build_actor_network(
        architecture_cfg=cfg["actor_architecture"],
        action_dim=env.num_actions,
        embed_dim=cfg["embed_dim"],
        observation_space=observation_space,
        readout_only=readout_only,
        frozen_encoder=frozen_encoder,
        memory_seed=memory_seed,
        auxiliary_cue=auxiliary_cue,
        auxiliary_readout_dim=auxiliary_readout_dim,
        preserve_observation_prefix=preserve_observation_prefix,
    )
    critic_network = build_critic_network(
        architecture_cfg=cfg["critic_architecture"],
        embed_dim=cfg["embed_dim"],
        observation_space=observation_space,
        readout_only=readout_only,
        frozen_encoder=frozen_encoder,
        memory_seed=critic_memory_seed,
        auxiliary_cue=auxiliary_cue,
        auxiliary_readout_dim=auxiliary_readout_dim,
        preserve_observation_prefix=preserve_observation_prefix,
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

    agent_cls = AuxiliaryCuePPO if auxiliary_cue else PPO
    agent_kwargs = {}
    if auxiliary_cue:
        agent_kwargs["auxiliary_cue_coefficient"] = float(
            cfg["auxiliary_cue_coefficient"]
        )
    return agent_cls(
        cfg=ppo_cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        **agent_kwargs,
    )


def build_stream_ac(cfg: dict, env, env_params):
    tbptt_steps = cfg.get("tbptt_steps", 1)   # извлечение
    observation_space = env.observation_space(env_params)
    readout_only = bool(cfg.get("readout_only", False))
    frozen_encoder = bool(cfg.get("frozen_encoder", False))
    memory_seed = int(cfg.get("memory_seed", 0))
    critic_memory_seed = int(cfg.get("critic_memory_seed", memory_seed + 1))
    auxiliary_cue = float(cfg.get("auxiliary_cue_coefficient", 0.0)) > 0.0
    auxiliary_readout_dim = int(cfg.get("auxiliary_readout_dim", 32))
    preserve_observation_prefix = int(cfg.get("preserve_observation_prefix", 0))

    actor_network = build_actor_network(
        architecture_cfg=cfg["actor_architecture"],
        action_dim=env.num_actions,
        embed_dim=cfg["embed_dim"],
        tbptt_steps=tbptt_steps,          # передаём
        observation_space=observation_space,
        readout_only=readout_only,
        frozen_encoder=frozen_encoder,
        memory_seed=memory_seed,
        auxiliary_cue=auxiliary_cue,
        auxiliary_readout_dim=auxiliary_readout_dim,
        preserve_observation_prefix=preserve_observation_prefix,
    )
    critic_network = build_critic_network(
        architecture_cfg=cfg["critic_architecture"],
        embed_dim=cfg["embed_dim"],
        tbptt_steps=tbptt_steps,          # передаём
        observation_space=observation_space,
        readout_only=readout_only,
        frozen_encoder=frozen_encoder,
        memory_seed=critic_memory_seed,
        auxiliary_cue=auxiliary_cue,
        auxiliary_readout_dim=auxiliary_readout_dim,
        preserve_observation_prefix=preserve_observation_prefix,
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

    if auxiliary_cue:
        agent_cls = AuxiliaryCueStreamAC
    else:
        agent_cls = DirectEntropyStreamAC if cfg.get("direct_entropy_update", False) else StreamAC
    agent_kwargs = {}
    if auxiliary_cue:
        agent_kwargs.update(
            auxiliary_cue_coefficient=float(cfg["auxiliary_cue_coefficient"]),
            auxiliary_cue_lr=float(cfg.get("auxiliary_cue_lr", 0.05)),
            auxiliary_cue_kappa=float(cfg.get("auxiliary_cue_kappa", 1.0)),
        )
    return agent_cls(
        cfg=stream_ac_cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        **agent_kwargs,
    )


def build_stream_tbptt(cfg: dict, env, env_params):
    observation_space = env.observation_space(env_params)
    readout_only = bool(cfg.get("readout_only", False))
    frozen_encoder = bool(cfg.get("frozen_encoder", False))
    memory_seed = int(cfg.get("memory_seed", 0))
    preserve_observation_prefix = int(cfg.get("preserve_observation_prefix", 0))
    actor_network = build_actor_network(
        architecture_cfg=cfg["actor_architecture"],
        action_dim=env.num_actions,
        embed_dim=cfg["embed_dim"],
        tbptt_steps=1,
        observation_space=observation_space,
        readout_only=readout_only,
        frozen_encoder=frozen_encoder,
        memory_seed=memory_seed,
        preserve_observation_prefix=preserve_observation_prefix,
    )
    critic_network = build_critic_network(
        architecture_cfg=cfg["critic_architecture"],
        embed_dim=cfg["embed_dim"],
        tbptt_steps=1,
        observation_space=observation_space,
        readout_only=readout_only,
        frozen_encoder=frozen_encoder,
        memory_seed=memory_seed + 1,
        preserve_observation_prefix=preserve_observation_prefix,
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
    return WindowedStreamAC(
        cfg=stream_ac_cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        window=int(cfg.get("tbptt_steps", 5)),
    )

BUILDERS = {
    "ppo": build_ppo,
    "stream_ac": build_stream_ac,
    "stream_eprop": build_stream_eprop,  # новый
    "stream_tbptt": build_stream_tbptt,
}


def build_agent(name: str, cfg: dict, env, env_params):
    if name not in BUILDERS:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(BUILDERS)}")
    return BUILDERS[name](cfg, env, env_params)
