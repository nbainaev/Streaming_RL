import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from stream_rl.visual_utils.sweeps import SweepAggregator, load_sweep_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', action='append', required=True,
                        help='Repeatable run root containing seed_* directories and saved structure/config log.')
    parser.add_argument('--group-by', required=True,
                        help='Dotted path in saved config, e.g. env.corridor_length')
    parser.add_argument('--metric', default='final_mean_last_k_episodes',
                        choices=['final_mean_last_k_episodes', 'final_episode_return', 'final_smoothed_return', 'best_episode_return'])
    parser.add_argument('--last-k', type=int, default=100)
    parser.add_argument('--split-by', default=None,
                        help='Optional second dotted path for multiple lines, e.g. agent.name')
    parser.add_argument('--ci', type=float, default=0.95)
    parser.add_argument('--window-steps', type=int, default=5000)
    parser.add_argument('--xlabel', type=str, default=None)
    parser.add_argument('--ylabel', type=str, default=None)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--dpi', type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    points = load_sweep_points(
        run_roots=args.run_root,
        group_by=args.group_by,
        metric=args.metric,
        last_k=args.last_k,
        split_by=args.split_by,
        window_steps=args.window_steps,
    )
    summaries = SweepAggregator(confidence=args.ci).summarize(points)

    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = {}
    for s in summaries:
        grouped.setdefault(s.split_value, []).append(s)

    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
    for i, (split_value, rows) in enumerate(grouped.items()):
        rows = sorted(rows, key=lambda r: r.x_value)
        x = [r.x_value for r in rows]
        y = [r.mean for r in rows]
        lo = [r.lo for r in rows]
        hi = [r.hi for r in rows]
        label = str(split_value) if split_value is not None else 'all runs'
        color = colors[i % len(colors)]
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.fill_between(x, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel(args.xlabel or args.group_by)
    ax.set_ylabel(args.ylabel or args.metric)
    ax.set_title(args.title)
    ax.grid(alpha=0.25)
    if args.split_by:
        ax.legend(title=args.split_by)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()