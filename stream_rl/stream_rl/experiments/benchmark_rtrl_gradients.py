"""Compare exact RTRL with online recurrent-gradient approximations.

The benchmark is deliberately independent of the policy optimizer.  It uses
the same recurrent state equations and measures the gradient of a scalar
state readout at every time step.  Full RTRL is propagated forward as

    S_t = d h_t / d theta = J_h S_{t-1} + J_theta.

TBPTT(k) retains only the last k terms of that expansion, while the one-step
approximation keeps only ``J_theta``.  The neuron-local approximation keeps
only within-unit blocks of ``J_h`` and is the structural part of e-prop; an
actual e-prop learner additionally approximates the learning signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree


KINDS = ("gru", "lstm", "rtu", "delta")


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    input_dim: int
    width: int
    state_dim: int
    param_dim: int
    flat_params: jax.Array
    unravel: object


def _init_model(kind: str, key: jax.Array, input_dim: int, width: int) -> ModelSpec:
    keys = iter(jax.random.split(key, 12))

    def normal(shape, scale=0.2):
        return scale * jax.random.normal(next(keys), shape)

    if kind == "gru":
        params = {
            "wx": normal((input_dim, 3 * width)),
            "wh": normal((width, 3 * width)),
            "b": jnp.zeros((3 * width,)),
        }
        state_dim = width
    elif kind == "lstm":
        params = {
            "wx": normal((input_dim, 4 * width)),
            "wh": normal((width, 4 * width)),
            "b": jnp.zeros((4 * width,)),
        }
        state_dim = 2 * width
    elif kind == "rtu":
        params = {
            "rho": normal((width,)),
            "phase": normal((width,)),
            "b_real": normal((input_dim, width)),
            "b_imag": normal((input_dim, width)),
        }
        state_dim = 2 * width
    elif kind == "delta":
        params = {
            "wq": normal((input_dim, width)),
            "wk": normal((input_dim, width)),
            "wv": normal((input_dim, width)),
            "wb": normal((input_dim, 1)),
            "wd": normal((input_dim, 1)),
        }
        state_dim = width * width
    else:
        raise ValueError(kind)

    flat, unravel = ravel_pytree(params)
    return ModelSpec(kind, input_dim, width, state_dim, flat.size, flat, unravel)


def _transition(spec: ModelSpec, flat_params, state, x):
    p = spec.unravel(flat_params)
    h = spec.width
    if spec.kind == "gru":
        gi = x @ p["wx"] + p["b"]
        gh = state @ p["wh"]
        ir, iz, inn = jnp.split(gi, 3)
        hr, hz, hn = jnp.split(gh, 3)
        reset = jax.nn.sigmoid(ir + hr)
        update = jax.nn.sigmoid(iz + hz)
        candidate = jnp.tanh(inn + reset * hn)
        return (1.0 - update) * candidate + update * state
    if spec.kind == "lstm":
        cell, hidden = state[:h], state[h:]
        gates = x @ p["wx"] + hidden @ p["wh"] + p["b"]
        ingate, forget, candidate, outgate = jnp.split(gates, 4)
        ingate = jax.nn.sigmoid(ingate)
        forget = jax.nn.sigmoid(forget)
        outgate = jax.nn.sigmoid(outgate)
        new_cell = forget * cell + ingate * jnp.tanh(candidate)
        new_hidden = outgate * jnp.tanh(new_cell)
        return jnp.concatenate([new_cell, new_hidden])
    if spec.kind == "rtu":
        real, imag = state[:h], state[h:]
        radius = 0.99 * jax.nn.sigmoid(p["rho"])
        phase = 2.0 * math.pi * jax.nn.sigmoid(p["phase"])
        cosine, sine = jnp.cos(phase), jnp.sin(phase)
        new_real = radius * (cosine * real - sine * imag) + x @ p["b_real"]
        new_imag = radius * (sine * real + cosine * imag) + x @ p["b_imag"]
        return jnp.concatenate([new_real, new_imag])

    memory = state.reshape(h, h)
    query = x @ p["wq"]
    key = x @ p["wk"]
    value = jnp.tanh(x @ p["wv"])
    query = query / jnp.maximum(jnp.linalg.norm(query), 1e-6)
    key = key / jnp.maximum(jnp.linalg.norm(key), 1e-6)
    beta = jax.nn.sigmoid(x @ p["wb"])[0]
    decay = 0.90 + 0.099 * jax.nn.sigmoid(x @ p["wd"])[0]
    residual = value - memory.T @ key
    new_memory = decay * memory + beta * jnp.outer(key, residual)
    return new_memory.reshape(-1)


def _local_mask(spec: ModelSpec) -> jax.Array:
    s, h = spec.state_dim, spec.width
    mask = np.zeros((s, s), dtype=np.float32)
    if spec.kind == "gru":
        np.fill_diagonal(mask, 1.0)
    elif spec.kind in {"lstm", "rtu"}:
        for idx in range(h):
            block = (idx, h + idx)
            mask[np.ix_(block, block)] = 1.0
    else:
        np.fill_diagonal(mask, 1.0)
    return jnp.asarray(mask)


def _cosine(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = np.linalg.norm(reference) * np.linalg.norm(estimate)
    return float(np.dot(reference, estimate) / denom) if denom > 1e-12 else math.nan


def _metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref_norm = np.linalg.norm(reference)
    active = np.abs(reference) > 1e-8
    return {
        "cosine": _cosine(reference, estimate),
        "relative_l2": float(np.linalg.norm(reference - estimate) / max(ref_norm, 1e-12)),
        "norm_ratio": float(np.linalg.norm(estimate) / max(ref_norm, 1e-12)),
        "sign_agreement": float(np.mean(np.sign(reference[active]) == np.sign(estimate[active])))
        if np.any(active)
        else math.nan,
    }


def _bptt_gradient(spec: ModelSpec, inputs, readout, target):
    def loss(flat_params):
        state = jnp.zeros((spec.state_dim,))
        for x in inputs:
            state = _transition(spec, flat_params, state, x)
        error = jnp.dot(readout, state) - target[-1]
        return 0.5 * error * error

    return jax.grad(loss)(spec.flat_params)


def benchmark_one(kind: str, seed: int, horizon: int, input_dim: int, width: int):
    key = jax.random.key(seed)
    model_key, x_key, readout_key, target_key = jax.random.split(key, 4)
    spec = _init_model(kind, model_key, input_dim, width)
    inputs = jax.random.normal(x_key, (horizon, input_dim))
    readout = jax.random.normal(readout_key, (spec.state_dim,)) / math.sqrt(spec.state_dim)
    targets = jax.random.normal(target_key, (horizon,))

    transition = jax.jit(lambda p, s, x: _transition(spec, p, s, x))
    jac_state = jax.jit(jax.jacrev(lambda s, p, x: _transition(spec, p, s, x), argnums=0))
    jac_params = jax.jit(jax.jacrev(lambda p, s, x: _transition(spec, p, s, x), argnums=0))

    state = jnp.zeros((spec.state_dim,))
    exact = jnp.zeros((spec.state_dim, spec.param_dim))
    local = jnp.zeros_like(exact)
    local_mask = _local_mask(spec)
    contributions: list[jax.Array] = []
    rows = []

    # Compile all three kernels before measuring recurrent propagation.
    x0 = inputs[0]
    transition(spec.flat_params, state, x0).block_until_ready()
    jac_state(state, spec.flat_params, x0).block_until_ready()
    jac_params(spec.flat_params, state, x0).block_until_ready()
    start = time.perf_counter()

    for step, x in enumerate(inputs, start=1):
        js = jac_state(state, spec.flat_params, x)
        jp = jac_params(spec.flat_params, state, x)
        state = transition(spec.flat_params, state, x)
        exact = js @ exact + jp
        one_step = jp
        local = (js * local_mask) @ local + jp
        contributions = [jp] + [js @ item for item in contributions[:4]]
        tbptt5 = sum(contributions)

        error = jnp.dot(readout, state) - targets[step - 1]
        loss_state_grad = error * readout
        gradients = {
            "one_step": loss_state_grad @ one_step,
            "tbptt5": loss_state_grad @ tbptt5,
            "local_eprop": loss_state_grad @ local,
        }
        reference = np.asarray(loss_state_grad @ exact)
        for method, gradient in gradients.items():
            values = _metrics(reference, np.asarray(gradient))
            rows.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "step": step,
                    "horizon": horizon,
                    "width": width,
                    "state_dim": spec.state_dim,
                    "param_dim": spec.param_dim,
                    "method": method,
                    **values,
                }
            )

    exact.block_until_ready()
    elapsed = time.perf_counter() - start
    bptt = np.asarray(_bptt_gradient(spec, inputs, readout, targets))
    final_reference = np.asarray((jnp.dot(readout, state) - targets[-1]) * readout @ exact)
    validation = _metrics(final_reference, bptt)
    validation["max_abs_error"] = float(np.max(np.abs(final_reference - bptt)))
    resource = {
        "kind": kind,
        "seed": seed,
        "width": width,
        "state_dim": spec.state_dim,
        "param_dim": spec.param_dim,
        "exact_sensitivity_mib": spec.state_dim * spec.param_dim * 4 / (1024**2),
        "one_step_persistent_mib": 0.0,
        "local_trace_upper_mib": spec.param_dim * (2 if kind in {"lstm", "rtu"} else 1) * 4 / (1024**2),
        "milliseconds_per_step": 1000.0 * elapsed / horizon,
        "bptt_rtrl_cosine": validation["cosine"],
        "bptt_rtrl_max_abs_error": validation["max_abs_error"],
    }
    return rows, resource


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]):
    grouped = {}
    for row in rows:
        key = (row["kind"], row["method"], row["step"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (kind, method, step), values in sorted(grouped.items()):
        result.append(
            {
                "kind": kind,
                "method": method,
                "step": step,
                **{
                    metric: float(np.nanmean([item[metric] for item in values]))
                    for metric in ("cosine", "relative_l2", "norm_ratio", "sign_agreement")
                },
            }
        )
    return result


def _plot(rows: list[dict], resources: list[dict], output: Path):
    aggregate = _aggregate(rows)
    colors = {"one_step": "#d95f02", "tbptt5": "#1b9e77", "local_eprop": "#7570b3"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, kind in zip(axes.flat, KINDS):
        for method, color in colors.items():
            subset = [row for row in aggregate if row["kind"] == kind and row["method"] == method]
            axis.plot([row["step"] for row in subset], [row["cosine"] for row in subset], label=method, color=color)
        axis.axhline(0.0, color="#888888", linewidth=0.7)
        axis.set_title(kind.upper())
        axis.set_ylim(-1.05, 1.05)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Sequence step")
    axes[1, 1].set_xlabel("Sequence step")
    axes[0, 0].set_ylabel("Cosine with exact RTRL")
    axes[1, 0].set_ylabel("Cosine with exact RTRL")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "gradient_cosine_by_horizon.png", dpi=180)
    plt.close(fig)

    summary = {}
    for row in resources:
        summary.setdefault(row["kind"], []).append(row)
    kinds = list(KINDS)
    exact = [np.mean([r["exact_sensitivity_mib"] for r in summary[k]]) for k in kinds]
    local = [np.mean([r["local_trace_upper_mib"] for r in summary[k]]) for k in kinds]
    x = np.arange(len(kinds))
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(x - 0.18, exact, width=0.36, label="exact RTRL tensor", color="#264653")
    axis.bar(x + 0.18, local, width=0.36, label="local trace upper bound", color="#2a9d8f")
    axis.set_xticks(x, [kind.upper() for kind in kinds])
    axis.set_yscale("log")
    axis.set_ylabel("Persistent sensitivity storage, MiB (log)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "gradient_memory_cost.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, resources = [], []
    for kind in args.kinds:
        for seed in range(args.seeds):
            kind_rows, resource = benchmark_one(
                kind, seed, args.horizon, args.input_dim, args.width
            )
            rows.extend(kind_rows)
            resources.append(resource)

    aggregate = _aggregate(rows)
    _write_csv(args.output / "gradient_agreement_raw.csv", rows)
    _write_csv(args.output / "gradient_agreement_mean.csv", aggregate)
    _write_csv(args.output / "gradient_resources.csv", resources)
    if not args.no_plots:
        _plot(rows, resources, args.output)

    final = [row for row in aggregate if row["step"] == args.horizon]
    payload = {
        "settings": vars(args) | {"output": str(args.output)},
        "final_step": final,
        "resources": resources,
    }
    (args.output / "gradient_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    print(args.output / "gradient_summary.json")


if __name__ == "__main__":
    main()
