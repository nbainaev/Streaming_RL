"""Reproducible staged architecture search for recurrent RL agents.

The search keeps PPO at num_envs=1 and therefore num_minibatches=1.  Recurrent
PPO minibatches in memorax are split across environment trajectories, not
across time, so more than one minibatch is invalid when only one trajectory
is collected.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from stream_rl.experiments.runners.base_runner import (
    ExperimentRunner,
    NoOpScenario,
    save_yaml,
)


def recurrent_architecture(
    cell: str,
    width: int,
    recurrent_depth: int,
    head_sizes: tuple[int, ...],
    *,
    pre_size: int | None = None,
) -> list[dict]:
    layers: list[dict] = []
    if pre_size is not None:
        layers.append(
            {"type": "fc", "features": pre_size, "activation": "tanh", "init": "sparse"}
        )
    for _ in range(recurrent_depth):
        layers.extend(
            [
                {"type": "rnn", "cell": cell, "features": width},
                {"type": "layernorm"},
            ]
        )
    for size in head_sizes:
        layers.append(
            {"type": "fc", "features": size, "activation": "tanh", "init": "sparse"}
        )
    layers.append({"type": "fc", "features": -1, "init": "sparse"})
    return layers


def ppo_cfg(
    *,
    cell="gru",
    width=64,
    recurrent_depth=1,
    head_sizes=(),
    pre_size=None,
    num_steps=128,
    lr=3e-4,
    entropy=0.01,
) -> dict:
    arch = recurrent_architecture(
        cell, width, recurrent_depth, tuple(head_sizes), pre_size=pre_size
    )
    return {
        "name": "ppo",
        "device": "cpu",
        "embed_dim": width,
        "num_envs": 1,
        "num_steps": num_steps,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_minibatches": 1,
        "update_epochs": 4,
        "normalize_advantage": True,
        "clip_coefficient": 0.2,
        "clip_value_loss": True,
        "entropy_coefficient": entropy,
        "actor_lr": lr,
        "critic_lr": lr,
        "max_grad_norm": 0.5,
        "actor_architecture": copy.deepcopy(arch),
        "critic_architecture": copy.deepcopy(arch),
    }


def stream_cfg(
    *,
    cell="gru",
    width=64,
    recurrent_depth=1,
    head_sizes=(),
    pre_size=None,
    trace_lambda=0.9,
    actor_lr=1.0,
    critic_lr=1.0,
    actor_kappa=0.2,
    critic_kappa=0.5,
    entropy=0.01,
    adaptive=False,
) -> dict:
    arch = recurrent_architecture(
        cell, width, recurrent_depth, tuple(head_sizes), pre_size=pre_size
    )
    return {
        "name": "stream_ac",
        "device": "cpu",
        "embed_dim": width,
        "num_envs": 1,
        "gamma": 0.99,
        "trace_lambda": trace_lambda,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "actor_kappa": actor_kappa,
        "critic_kappa": critic_kappa,
        "entropy_coefficient": entropy,
        "adaptive": adaptive,
        # StreamAC consumes a one-step sequence per update.  Values above 1
        # only change scan unrolling and do not create multi-step BPTT.
        "tbptt_steps": 1,
        "actor_architecture": copy.deepcopy(arch),
        "critic_architecture": copy.deepcopy(arch),
    }


def windowed_stream_cfg(*, cell="gru", width=64, window=5, **kwargs) -> dict:
    """AC(lambda) whose recurrent Jacobian spans a real k-step window."""
    cfg = stream_cfg(cell=cell, width=width, **kwargs)
    cfg["name"] = "stream_tbptt"
    cfg["tbptt_steps"] = window
    return cfg


def eprop_cfg(
    *,
    cell="eprop_gru",
    width=64,
    head_sizes=(),
    trace_decay=0.9,
    trace_lambda=0.9,
    actor_lr=1.0,
    critic_lr=1.0,
    actor_kappa=0.2,
    critic_kappa=0.5,
    entropy=0.01,
    feedback_mode="symmetric",
    adaptive=False,
) -> dict:
    return {
        "name": "stream_eprop",
        "device": "cpu",
        "num_envs": 1,
        "embed_dim": width,
        "hidden_size": width,
        "cell": cell,
        "activation": "tanh",
        "gamma": 0.99,
        "trace_decay": trace_decay,
        "trace_lambda": trace_lambda,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "actor_kappa": actor_kappa,
        "critic_kappa": critic_kappa,
        "entropy_coefficient": entropy,
        "adaptive": adaptive,
        "feedback_mode": feedback_mode,
        "feedback_seed": 0,
        "feedback_lr": 0.05,
        "use_layernorm": True,
        "use_sparse_init": True,
        "sparsity": 0.9,
        "head_hidden_sizes": list(head_sizes),
        "head_activation": "tanh",
    }


def direct_entropy_stream_cfg(**kwargs) -> dict:
    cfg = stream_cfg(**kwargs)
    cfg["direct_entropy_update"] = True
    return cfg


def rtu_rtrl_stream_cfg(
    *,
    hidden_dim=32,
    recurrent_depth=1,
    head_sizes=(64,),
    embed_dim=64,
    trace_lambda=0.95,
    actor_lr=0.01,
    critic_lr=0.03,
    actor_kappa=3.0,
    critic_kappa=2.0,
    entropy=0.001,
    direct_entropy=True,
    r_min=0.0,
    r_max=1.0,
) -> dict:
    """AC(lambda) with an RTU trained by the Memorax RTRL wrapper.

    RTU emits real and imaginary components, so hidden_dim=32 exposes a
    64-dimensional recurrent representation, matching the output width of
    the LSTM-64 comparison model.  A depth above one is a *local-RTRL*
    hierarchy: each layer tracks derivatives for its own recurrent kernel,
    but cross-layer temporal sensitivities are not represented.
    """
    arch = recurrent_architecture(
        "rtu_rtrl", hidden_dim, recurrent_depth, tuple(head_sizes)
    )
    for layer in arch:
        if layer.get("cell") == "rtu_rtrl":
            layer["r_min"] = r_min
            layer["r_max"] = r_max
    return {
        "name": "stream_ac",
        "device": "cpu",
        "embed_dim": embed_dim,
        "num_envs": 1,
        "gamma": 0.99,
        "trace_lambda": trace_lambda,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "actor_kappa": actor_kappa,
        "critic_kappa": critic_kappa,
        "entropy_coefficient": entropy,
        "adaptive": False,
        "direct_entropy_update": direct_entropy,
        "tbptt_steps": 1,
        "rtrl_scope": "single_layer_exact" if recurrent_depth == 1 else "layer_local",
        "actor_architecture": copy.deepcopy(arch),
        "critic_architecture": copy.deepcopy(arch),
    }


def rtu_bptt_ppo_cfg(
    *, hidden_dim=32, recurrent_depth=1, head_sizes=(64,), embed_dim=64
) -> dict:
    """PPO control with the same RTU state/output widths, trained by BPTT."""
    cfg = ppo_cfg(
        cell="rtu_bptt",
        width=hidden_dim,
        recurrent_depth=recurrent_depth,
        head_sizes=head_sizes,
    )
    cfg["embed_dim"] = embed_dim
    return cfg


CANDIDATES = {
    # PPO controls and architectural variants.
    "ppo_gru1_linear": ppo_cfg(),
    "ppo_gru1_mlp64": ppo_cfg(head_sizes=(64,)),
    "ppo_gru1_mlp64x2": ppo_cfg(head_sizes=(64, 64)),
    "ppo_gru2_mlp64": ppo_cfg(recurrent_depth=2, head_sizes=(64,)),
    "ppo_lstm1_mlp64": ppo_cfg(cell="lstm", head_sizes=(64,)),
    "ppo_lstm2_mlp64": ppo_cfg(
        cell="lstm", recurrent_depth=2, head_sizes=(64,)
    ),
    "ppo_gru1_prepost64": ppo_cfg(pre_size=64, head_sizes=(64,)),
    "ppo_gru1_mlp64_ns64": ppo_cfg(head_sizes=(64,), num_steps=64),
    "ppo_gru1_mlp64_ns256": ppo_cfg(head_sizes=(64,), num_steps=256),
    "ppo_gru1_mlp64_lr7e4": ppo_cfg(head_sizes=(64,), lr=7e-4),
    "ppo_gru2_mlp64_ns64": ppo_cfg(
        recurrent_depth=2, head_sizes=(64,), num_steps=64
    ),
    "ppo_gru2_mlp64_ns256": ppo_cfg(
        recurrent_depth=2, head_sizes=(64,), num_steps=256
    ),
    "ppo_gru2_mlp64_lr7e4": ppo_cfg(
        recurrent_depth=2, head_sizes=(64,), lr=7e-4
    ),
    "ppo_gru2_mlp64_ent02": ppo_cfg(
        recurrent_depth=2, head_sizes=(64,), entropy=0.02
    ),
    "ppo_gru2_w32_mlp32": ppo_cfg(
        width=32, recurrent_depth=2, head_sizes=(32,)
    ),
    "ppo_gru2_w128_mlp128": ppo_cfg(
        width=128, recurrent_depth=2, head_sizes=(128,)
    ),
    "ppo_rtu1_h32_mlp64_bptt": rtu_bptt_ppo_cfg(),
    # Ordinary recurrent StreamAC.  Two recurrent layers test hierarchy.
    "stream_gru1_linear": stream_cfg(),
    "stream_gru1_mlp64": stream_cfg(head_sizes=(64,)),
    "stream_gru1_mlp64x2": stream_cfg(head_sizes=(64, 64)),
    "stream_gru2_mlp64": stream_cfg(recurrent_depth=2, head_sizes=(64,)),
    "stream_lstm1_mlp64": stream_cfg(cell="lstm", head_sizes=(64,)),
    "stream_lstm2_mlp64": stream_cfg(cell="lstm", recurrent_depth=2, head_sizes=(64,)),
    "stream_gru1_linear_tbptt5_true": windowed_stream_cfg(cell="gru", window=5),
    "stream_lstm1_linear_tbptt5_true": windowed_stream_cfg(cell="lstm", window=5),
    "stream_rtu1_h64_linear_bptt1": stream_cfg(cell="rtu_bptt", width=64),
    "stream_rtu1_h64_linear_tbptt5_true": windowed_stream_cfg(
        cell="rtu_bptt", width=64, window=5
    ),
    "stream_delta1_h64_k16_linear": stream_cfg(cell="delta_rule", width=64),
    "stream_gru1_linear_tbptt5_true_tuned": windowed_stream_cfg(
        cell="gru", window=5, trace_lambda=0.98, adaptive=True
    ),
    "stream_lstm1_linear_tuned": stream_cfg(
        cell="lstm", trace_lambda=0.98, adaptive=True
    ),
    "stream_rtu1_h64_linear_tbptt5_true_tuned": windowed_stream_cfg(
        cell="rtu_bptt", width=64, window=5, trace_lambda=0.98, adaptive=True
    ),
    "stream_rtu1_h64_linear_tbptt5_true_matched": windowed_stream_cfg(
        cell="rtu_bptt", width=64, window=5, trace_lambda=0.97, adaptive=True
    ),
    "stream_delta1_h64_k16_linear_tuned": stream_cfg(
        cell="delta_rule", width=64, trace_lambda=0.98, adaptive=True
    ),
    "stream_delta1_h64_k16_mlp64": stream_cfg(
        cell="delta_rule", width=64, head_sizes=(64,)
    ),
    "stream_delta1_h64_k16_directent": direct_entropy_stream_cfg(
        cell="delta_rule",
        width=64,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.01,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    "stream_lstm2_directent_alr01_clr03_ent001_k3": direct_entropy_stream_cfg(
        cell="lstm",
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.01,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    "stream_gru2_w128_directent_alr01_clr03_ent001_k3": direct_entropy_stream_cfg(
        cell="gru",
        width=128,
        recurrent_depth=2,
        head_sizes=(128,),
        trace_lambda=0.95,
        actor_lr=0.01,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    # RTU recurrent state contains real+imaginary components.  The one-layer
    # candidate is the primary exact-RTRL baseline.  The two-layer candidate
    # deliberately tests hierarchical memory but is only layer-local RTRL.
    "stream_rtu1_h32_mlp64_rtrl_directent": rtu_rtrl_stream_cfg(),
    "stream_rtu2_h32_mlp64_localrtrl_directent": rtu_rtrl_stream_cfg(
        recurrent_depth=2
    ),
    "stream_rtu1_h64_linear_rtrl_paper": rtu_rtrl_stream_cfg(
        hidden_dim=64,
        head_sizes=(),
        trace_lambda=0.9,
        actor_lr=1.0,
        critic_lr=1.0,
        actor_kappa=0.2,
        critic_kappa=0.5,
        entropy=0.01,
        direct_entropy=False,
    ),
    "stream_rtu1_h64_linear_rtrl_paper_directent": rtu_rtrl_stream_cfg(
        hidden_dim=64,
        head_sizes=(),
        trace_lambda=0.9,
        actor_lr=1.0,
        critic_lr=1.0,
        actor_kappa=0.2,
        critic_kappa=0.5,
        entropy=0.01,
        direct_entropy=True,
    ),
    "stream_rtu1_h64_linear_rtrl_alr03_clr03_ent001_k3": rtu_rtrl_stream_cfg(
        hidden_dim=64,
        head_sizes=(),
        actor_lr=0.03,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    "stream_rtu1_h64_linear_rtrl_alr10_clr10_ent001_k3": rtu_rtrl_stream_cfg(
        hidden_dim=64,
        head_sizes=(),
        actor_lr=0.10,
        critic_lr=0.10,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    "stream_rtu1_h64_linear_rtrl_paper_rmax99": rtu_rtrl_stream_cfg(
        hidden_dim=64,
        head_sizes=(),
        trace_lambda=0.9,
        actor_lr=1.0,
        critic_lr=1.0,
        actor_kappa=0.2,
        critic_kappa=0.5,
        entropy=0.01,
        direct_entropy=False,
        r_max=0.99,
    ),
    "stream_gru1_prepost64": stream_cfg(pre_size=64, head_sizes=(64,)),
    "stream_gru1_mlp64_lam95": stream_cfg(head_sizes=(64,), trace_lambda=0.95),
    "stream_gru1_mlp64_lr03": stream_cfg(head_sizes=(64,), actor_lr=0.3, critic_lr=0.3),
    "stream_gru1_mlp64_kappa1": stream_cfg(
        head_sizes=(64,), actor_kappa=1.0, critic_kappa=1.0
    ),
    "stream_gru2_mlp64_lam80": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), trace_lambda=0.8
    ),
    "stream_gru2_mlp64_lam95": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), trace_lambda=0.95
    ),
    "stream_gru2_mlp64_lr03": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), actor_lr=0.3, critic_lr=0.3
    ),
    "stream_gru2_mlp64_kappa1": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), actor_kappa=1.0, critic_kappa=1.0
    ),
    "stream_gru2_mlp64_ent02": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), entropy=0.02
    ),
    # CT-graph exploration controls.  The default streaming policy often
    # collapses to a fixed leaf before it can react to reward switches, so
    # vary entropy pressure and the effective actor step independently.
    "stream_gru2_mlp64_ent05": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), trace_lambda=0.95, entropy=0.05
    ),
    "stream_gru2_mlp64_ent10": stream_cfg(
        recurrent_depth=2, head_sizes=(64,), trace_lambda=0.95, entropy=0.10
    ),
    "stream_gru2_mlp64_lr01_ent10": stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.1,
        critic_lr=0.1,
        entropy=0.10,
    ),
    "stream_gru2_mlp64_lr003_ent10": stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.03,
        critic_lr=0.03,
        entropy=0.10,
    ),
    "stream_gru2_mlp64_k005_ent10": stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_kappa=0.05,
        critic_kappa=0.25,
        entropy=0.10,
    ),
    "stream_gru2_directent_lr003_k3": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.03,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.10,
    ),
    "stream_gru2_directent_lr01_k3": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.10,
        critic_lr=0.10,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.10,
    ),
    "stream_gru2_directent_lr003_k10": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.03,
        critic_lr=0.03,
        actor_kappa=10.0,
        critic_kappa=2.0,
        entropy=0.10,
    ),
    "stream_gru2_directent_lr003_ent01_k3": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.03,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.01,
    ),
    "stream_gru2_directent_lr003_ent001_k3": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.03,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    "stream_gru2_directent_lr003_ent003_k3": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.03,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.003,
    ),
    "stream_gru2_directent_alr01_clr03_ent001_k3": direct_entropy_stream_cfg(
        recurrent_depth=2,
        head_sizes=(64,),
        trace_lambda=0.95,
        actor_lr=0.01,
        critic_lr=0.03,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy=0.001,
    ),
    "stream_gru2_w32_mlp32": stream_cfg(
        width=32, recurrent_depth=2, head_sizes=(32,)
    ),
    "stream_gru2_w128_mlp128": stream_cfg(
        width=128, recurrent_depth=2, head_sizes=(128,)
    ),
    "stream_lstm1_mlp64_lam95": stream_cfg(
        cell="lstm", head_sizes=(64,), trace_lambda=0.95
    ),
    "stream_lstm1_mlp64_lr03": stream_cfg(
        cell="lstm", head_sizes=(64,), actor_lr=0.3, critic_lr=0.3
    ),
    "stream_gru1_mlp64x2_lam95": stream_cfg(
        head_sizes=(64, 64), trace_lambda=0.95
    ),
    # Dedicated e-prop cells with symmetric or random learning signals.
    # Eligibility propagation remains local to each recurrent cell; ordinary
    # StreamAC candidates above instead use a one-step truncated Jacobian.
    "eprop_gru_sym_linear": eprop_cfg(),
    "eprop_gru_sym_linear_tuned": eprop_cfg(trace_lambda=0.98, adaptive=True),
    "eprop_gru_sym_mlp64": eprop_cfg(head_sizes=(64,)),
    "eprop_gru_random_mlp64": eprop_cfg(head_sizes=(64,), feedback_mode="random"),
    "eprop_lstm_sym_mlp64": eprop_cfg(cell="eprop_lstm", head_sizes=(64,)),
    "eprop_lstm_sym_linear": eprop_cfg(cell="eprop_lstm"),
    "eprop_lstm_sym_linear_tuned": eprop_cfg(
        cell="eprop_lstm", trace_lambda=0.98, adaptive=True
    ),
    "eprop_lstm_random_mlp64": eprop_cfg(
        cell="eprop_lstm", head_sizes=(64,), feedback_mode="random"
    ),
    "eprop_gru_sym_mlp64_decay99": eprop_cfg(
        head_sizes=(64,), trace_decay=0.99, trace_lambda=0.95
    ),
}


EASY_ENVS = {
    "popgym": {
        "namespace": "popgym",
        "env_id": "popgym-RepeatPreviousEasy-v0",
        "kwargs": {},
    },
    "active_tmaze": {
        "namespace": "tmaze",
        "env_id": "tmaze_active",
        "kwargs": {"corridor_length": 5, "goal_reward": 1.0},
    },
    "active_tmaze_short": {
        "namespace": "tmaze",
        "env_id": "tmaze_active",
        "kwargs": {"corridor_length": 2, "goal_reward": 1.0},
    },
    "active_tmaze_shaped": {
        "namespace": "tmaze",
        "env_id": "tmaze_active",
        "kwargs": {
            "corridor_length": 5,
            "goal_reward": 1.0,
            "oracle_reward": 0.1,
        },
    },
    "active_tmaze_shaped_strong": {
        "namespace": "tmaze",
        "env_id": "tmaze_active",
        "kwargs": {
            "corridor_length": 5,
            "goal_reward": 1.0,
            "oracle_reward": 1.0,
        },
    },
    "passive_tmaze": {
        "namespace": "tmaze",
        "env_id": "tmaze_passive",
        "kwargs": {"corridor_length": 5, "goal_reward": 1.0},
    },
    "ctgraph": {
        "namespace": "ctgraph",
        "env_id": "ctgraph_lifelong",
        "kwargs": {
            "depth": 2,
            "branching": 2,
            "reward_switch_steps": 128,
            "reward_distribution": "linear",
            "reward_seed": 0,
            "high_reward": 1.0,
            "fail_reward": -1.0,
            "continuing_task": True,
        },
    },
}


COMPLEX_ENVS = {
    "popgym": {
        "namespace": "popgym",
        "env_id": "popgym-RepeatPreviousMedium-v0",
        "kwargs": {},
    },
    "active_tmaze": {
        "namespace": "tmaze",
        "env_id": "tmaze_active",
        "kwargs": {"corridor_length": 20, "goal_reward": 1.0},
    },
    "ctgraph": {
        "namespace": "ctgraph",
        "env_id": "ctgraph_lifelong",
        "kwargs": {
            "depth": 4,
            "branching": 2,
            # PPO has num_envs=1 and num_minibatches=1, so one 128-step
            # rollout is the relevant on-policy buffer lifetime.
            "reward_switch_steps": 128,
            "reward_distribution": "linear",
            "reward_seed": 0,
            "high_reward": 1.0,
            "fail_reward": -1.0,
            "continuing_task": True,
        },
    },
}


def read_monitor(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"episodes": 0}

    def values(key):
        return [float(row[key]) for row in rows if row.get(key, "") != ""]

    returns = values("info.returned_episode_returns")
    successes = values("info.success")
    tail = min(500, len(returns))
    summary = {
        "episodes": len(rows),
        "last_500_return": sum(returns[-tail:]) / tail,
        "last_500_success": (
            sum(successes[-min(500, len(successes)):]) / min(500, len(successes))
            if successes
            else math.nan
        ),
    }

    if rows[0].get("info.reward_phase", "") != "":
        phases = defaultdict(list)
        for row in rows:
            phases[int(float(row["info.reward_phase"]))].append(
                float(row["info.returned_episode_returns"])
            )
        gains = []
        for phase in sorted(phases)[2:-1]:
            phase_returns = phases[phase]
            if len(phase_returns) >= 8:
                n = min(4, len(phase_returns) // 2)
                gains.append(
                    sum(phase_returns[-n:]) / n - sum(phase_returns[:n]) / n
                )
        summary["phase_adaptation_gain"] = (
            sum(gains) / len(gains) if gains else math.nan
        )
    return summary


def run_one(
    *,
    env_name: str,
    env_cfg: dict,
    variant: str,
    agent_cfg: dict,
    seed: int,
    total_timesteps: int,
    log_root: Path,
    checkpoint_every: int,
) -> dict:
    experiment = f"{env_name}__{variant}"
    run_dir = log_root / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor = run_dir / f"monitor_seed_{seed}.csv"

    runner_cfg = {
        "experiment_name": experiment,
        "run_id": experiment,
        "log_root": str(log_root),
        "total_timesteps": total_timesteps,
        "log_every": 4096,
        "step_chunk": 2048,
        "checkpoint_every": checkpoint_every,
        "eval_every": -1,
        "eval_steps": -1,
    }
    save_yaml(run_dir / "agent.yaml", agent_cfg)
    save_yaml(run_dir / "env.yaml", env_cfg)
    save_yaml(run_dir / "runner.yaml", runner_cfg)

    if not monitor.exists():
        runner = ExperimentRunner(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            runner_cfg=runner_cfg,
            run_dir=run_dir,
            scenario=NoOpScenario({"type": "none"}),
        )
        runner.run(seed)

    return {
        "environment": env_name,
        "variant": variant,
        "agent": agent_cfg["name"],
        "seed": seed,
        **read_monitor(monitor),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("easy", "complex"), default="easy")
    parser.add_argument("--env", choices=tuple(EASY_ENVS), required=True)
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated candidate names or 'all'.",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--total-timesteps", type=int, default=20480)
    parser.add_argument("--checkpoint-every", type=int, default=-1)
    parser.add_argument("--log-root", type=Path, default=Path("logs/architecture_search"))
    args = parser.parse_args()

    envs = EASY_ENVS if args.suite == "easy" else COMPLEX_ENVS
    variants = list(CANDIDATES) if args.variants == "all" else args.variants.split(",")
    unknown = sorted(set(variants) - set(CANDIDATES))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    log_root = args.log_root.resolve() / args.suite
    log_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in variants:
        for seed in seeds:
            print(f"RUN {args.suite}/{args.env}/{variant}/seed_{seed}", flush=True)
            try:
                rows.append(
                    run_one(
                        env_name=args.env,
                        env_cfg=copy.deepcopy(envs[args.env]),
                        variant=variant,
                        agent_cfg=copy.deepcopy(CANDIDATES[variant]),
                        seed=seed,
                        total_timesteps=args.total_timesteps,
                        log_root=log_root,
                        checkpoint_every=args.checkpoint_every,
                    )
                )
            except Exception as exc:  # retain failures without losing completed candidates
                rows.append(
                    {
                        "environment": args.env,
                        "variant": variant,
                        "agent": CANDIDATES[variant]["name"],
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"FAILED {variant}: {type(exc).__name__}: {exc}", flush=True)

    summary_json = log_root / f"summary_{args.env}.json"
    summary_csv = log_root / f"summary_{args.env}.csv"
    summary_json.write_text(json.dumps(rows, indent=2, allow_nan=True))
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(summary_json)


if __name__ == "__main__":
    main()
