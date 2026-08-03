"""Pretrain a compact SSM on offline delayed-recall sequences and freeze it.

Only ``decay/B/C/D`` are exported.  The temporary supervised decoder is
discarded, so downstream PPO and streaming agents train their own readouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


def fixed_projection(seed: int, input_dim: int, embed_dim: int):
    key = jax.random.key(seed)
    weight = jax.random.normal(key, (input_dim, embed_dim), dtype=jnp.float32)
    return weight / jnp.sqrt(float(max(input_dim, 1)))


def make_batch(key, batch_size: int, max_delay: int, projection):
    cue_key, delay_key, noise_key, mask_key = jax.random.split(key, 4)
    cue = jnp.where(
        jax.random.bernoulli(cue_key, 0.5, (batch_size,)), 1.0, -1.0
    )
    delay = jax.random.randint(delay_key, (batch_size,), 4, max_delay + 1)
    time_steps = max_delay + 1
    raw = 0.08 * jax.random.normal(
        noise_key, (batch_size, time_steps, 2), dtype=jnp.float32
    )
    # Initial observation matches passive T-maze: position=0, cue=+/-1.
    raw = raw.at[:, 0, 0].set(0.0)
    raw = raw.at[:, 0, 1].set(cue)
    query_mask = jax.nn.one_hot(delay, time_steps, dtype=jnp.float32)
    raw = raw.at[:, :, 0].set(
        jnp.where(query_mask > 0, 1.0, raw[:, :, 0])
    )
    # Half the sequences are clean; the other half contain weak distractors.
    noisy = jax.random.bernoulli(mask_key, 0.5, (batch_size, 1))
    raw = raw.at[:, 1:, 1].set(
        jnp.where(noisy, raw[:, 1:, 1], jnp.zeros_like(raw[:, 1:, 1]))
    )
    embedded = jnp.tanh(jnp.einsum("bti,ie->bte", raw, projection))
    return embedded, cue, delay


def init_params(key, embed_dim: int, state_dim: int):
    b_key, c_key, d_key, decoder_key = jax.random.split(key, 4)
    target_decay = jnp.geomspace(0.65, 0.999, state_dim)
    decay_logits = jnp.log(target_decay) - jnp.log1p(-target_decay)
    b = jax.random.normal(b_key, (state_dim, embed_dim))
    b = b / jnp.sqrt(float(embed_dim))
    b = b * jnp.sqrt(jnp.maximum(1.0 - target_decay**2, 1e-6))[:, None]
    c = jax.random.normal(c_key, (embed_dim, state_dim)) / jnp.sqrt(
        float(state_dim)
    )
    d = 0.5 * jnp.eye(embed_dim) + 0.01 * jax.random.normal(
        d_key, (embed_dim, embed_dim)
    ) / jnp.sqrt(float(embed_dim))
    decoder = jax.random.normal(decoder_key, (embed_dim,)) / jnp.sqrt(
        float(embed_dim)
    )
    return {
        "decay_logits": decay_logits,
        "B": b,
        "C": c,
        "D": d,
        "decoder": decoder,
        "decoder_bias": jnp.asarray(0.0, dtype=jnp.float32),
    }


def run_ssm(params, inputs):
    decay = 0.9999 * jax.nn.sigmoid(params["decay_logits"])
    x_time = jnp.swapaxes(inputs, 0, 1)
    initial = jnp.zeros((inputs.shape[0], params["B"].shape[0]))

    def step(state, x):
        state = decay[None, :] * state + jnp.einsum("hi,bi->bh", params["B"], x)
        output = jnp.tanh(
            jnp.einsum("oh,bh->bo", params["C"], state)
            + jnp.einsum("oi,bi->bo", params["D"], x)
        )
        return state, output

    _, output = jax.lax.scan(step, initial, x_time)
    return jnp.swapaxes(output, 0, 1)


def loss_and_metrics(params, inputs, cue, delay, reconstruction_coefficient=0.1):
    outputs = run_ssm(params, inputs)
    selected = outputs[jnp.arange(outputs.shape[0]), delay]
    prediction = jnp.tanh(selected @ params["decoder"] + params["decoder_bias"])
    recall_loss = 0.5 * jnp.square(prediction - cue).mean()
    # Preserve the current observation embedding as well as delayed context.
    # Without this term a cue-perfect SSM can still make corridor navigation
    # unnecessarily difficult for a small downstream policy readout.
    reconstruction_loss = 0.5 * jnp.square(outputs - inputs).mean()
    state_penalty = 1e-5 * jnp.square(selected).mean()
    loss = (
        recall_loss
        + reconstruction_coefficient * reconstruction_loss
        + state_penalty
    )
    accuracy = jnp.mean((jnp.sign(prediction) == cue).astype(jnp.float32))
    return loss, (accuracy, recall_loss, reconstruction_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-delay", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--encoder-seed", type=int, default=220512258)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--reconstruction-coefficient", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1 or args.max_delay < 4:
        parser.error("steps must be positive and max-delay must be >= 4")

    projection = fixed_projection(args.encoder_seed, 2, args.embed_dim)
    root_key = jax.random.key(args.seed)
    init_key, train_key, test_key = jax.random.split(root_key, 3)
    params = init_params(init_key, args.embed_dim, args.state_dim)
    optimizer = optax.adam(args.learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state, key):
        inputs, cue, delay = make_batch(
            key, args.batch_size, args.max_delay, projection
        )
        objective = lambda p: loss_and_metrics(
            p,
            inputs,
            cue,
            delay,
            reconstruction_coefficient=args.reconstruction_coefficient,
        )
        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss, metrics

    last = None
    for step, key in enumerate(jax.random.split(train_key, args.steps), start=1):
        params, optimizer_state, loss, metrics = train_step(
            params, optimizer_state, key
        )
        if step == 1 or step % 100 == 0 or step == args.steps:
            last = {
                "step": step,
                "loss": float(loss),
                "accuracy": float(metrics[0]),
            }
            print(json.dumps(last))

    test_inputs, test_cue, test_delay = make_batch(
        test_key, 4096, args.max_delay, projection
    )
    test_loss, (test_accuracy, _, test_reconstruction_loss) = loss_and_metrics(
        params,
        test_inputs,
        test_cue,
        test_delay,
        reconstruction_coefficient=args.reconstruction_coefficient,
    )
    decay = 0.9999 * jax.nn.sigmoid(params["decay_logits"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        decay=np.asarray(decay, dtype=np.float32),
        B=np.asarray(params["B"], dtype=np.float32),
        C=np.asarray(params["C"], dtype=np.float32),
        D=np.asarray(params["D"], dtype=np.float32),
        encoder_seed=np.asarray(args.encoder_seed, dtype=np.int64),
        pretrain_test_accuracy=np.asarray(float(test_accuracy), dtype=np.float32),
    )
    result = {
        "checkpoint": str(args.output.resolve()),
        "steps": args.steps,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_reconstruction_loss": float(test_reconstruction_loss),
        "min_decay": float(decay.min()),
        "max_decay": float(decay.max()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
