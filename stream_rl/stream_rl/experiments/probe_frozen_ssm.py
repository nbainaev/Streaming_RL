"""Linear cue probe for the frozen SSM representation on passive T-maze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from stream_rl.src.models.blocks import FrozenSSMMemoryBlock
from stream_rl.src.models.networks import FrozenProjectionEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--corridor-length", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    time_steps = args.corridor_length + 1
    key = jax.random.key(123)
    cues = jnp.where(
        jax.random.bernoulli(key, 0.5, (args.samples,)), 1.0, -1.0
    )
    observations = jnp.zeros((args.samples, time_steps, 2), dtype=jnp.float32)
    observations = observations.at[:, 0, 1].set(cues)
    observations = observations.at[:, -1, 0].set(1.0)
    done = jnp.zeros((args.samples, time_steps), dtype=jnp.bool_)

    encoder = FrozenProjectionEncoder(embed_dim=64, seed=220512258)
    encoder_vars = encoder.init(jax.random.key(0), observations)
    embeddings = encoder.apply(encoder_vars, observations)

    memory = FrozenSSMMemoryBlock(
        features=64,
        state_dim=128,
        seed=11,
        min_decay=0.5,
        max_decay=0.999,
    )
    memory_vars = memory.init(jax.random.key(0), embeddings, done=done)
    _, outputs = memory.apply(memory_vars, embeddings, done=done)
    features = np.asarray(outputs[:, -1])
    labels = np.asarray(cues)

    rng = np.random.default_rng(0)
    indices = rng.permutation(args.samples)
    split = args.samples // 2
    train, test = indices[:split], indices[split:]
    x_train = np.column_stack([features[train], np.ones(len(train))])
    x_test = np.column_stack([features[test], np.ones(len(test))])
    ridge = 1e-4 * np.eye(x_train.shape[1])
    weights = np.linalg.solve(
        x_train.T @ x_train + ridge, x_train.T @ labels[train]
    )
    predictions = np.sign(x_test @ weights)

    positive = features[labels > 0].mean(axis=0)
    negative = features[labels < 0].mean(axis=0)
    result = {
        "samples": args.samples,
        "corridor_length": args.corridor_length,
        "linear_probe_accuracy": float(np.mean(predictions == labels[test])),
        "class_mean_separation_l2": float(np.linalg.norm(positive - negative)),
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
