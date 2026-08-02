import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_RE = re.compile(r'-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+')
BOOL_RE = re.compile(r'True|False')

REQUIRED_COLUMNS = {
    'step', 'seed',
    'info.returned_episode',
    'info.returned_episode_returns',
    'info.returned_discounted_episode_returns',
    'info.returned_episode_lengths',
}


def parse_float_array(cell: str) -> np.ndarray:
    return np.array([float(x) for x in NUM_RE.findall(cell)], dtype=float)


def parse_bool_array(cell: str) -> np.ndarray:
    return np.array([tok == 'True' for tok in BOOL_RE.findall(cell)], dtype=bool)


def is_truncated(cell: str) -> bool:
    """Detects numpy's own summarization ellipsis ('...'), which indicates
    permanent, unrecoverable data loss for that chunk (array > print threshold)."""
    return '...' in cell


def recover_seed_file(path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(path)
    rows = []
    prev_step = 0.0
    truncated_chunks = 0

    for _, row in df.iterrows():
        this_step = float(row['step'])
        seed = row['seed']

        cells = (
            str(row['info.returned_episode']),
            str(row['info.returned_episode_returns']),
            str(row['info.returned_discounted_episode_returns']),
            str(row['info.returned_episode_lengths']),
        )
        if any(is_truncated(c) for c in cells):
            truncated_chunks += 1

        done = parse_bool_array(cells[0])
        returns = parse_float_array(cells[1])
        disc_returns = parse_float_array(cells[2])
        lengths = parse_float_array(cells[3])

        n = len(done)
        consistent = n > 0 and len(returns) == n and len(disc_returns) == n and len(lengths) == n
        if not consistent:
            prev_step = this_step
            continue

        for i in np.nonzero(done)[0]:
            frac = (int(i) + 1) / n
            interp_step = prev_step + frac * (this_step - prev_step)
            rows.append({
                'step': interp_step,
                'seed': seed,
                'returned_episode_returns': float(returns[i]),
                'returned_discounted_episode_returns': float(disc_returns[i]),
                'returned_episode_lengths': float(lengths[i]),
            })

        prev_step = this_step

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('step').reset_index(drop=True)
    return out, truncated_chunks


def looks_like_broken_schema(path: Path) -> bool:
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(set(header))


def find_target_files(root_dir: Path, pattern: str, out_suffix: str) -> list[Path]:
    candidates = sorted(root_dir.rglob(pattern))
    return [f for f in candidates if out_suffix not in f.stem]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root_dir', type=str, help='Root directory to search recursively for monitor CSVs')
    parser.add_argument('--pattern', type=str, default='monitor_seed_*.csv',
                         help='Glob pattern (relative to each directory) used to find monitor CSVs')
    parser.add_argument('--out-suffix', type=str, default='_recovered')
    parser.add_argument('--dry-run', action='store_true',
                         help='List files that would be processed without writing anything')
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f'{root_dir} does not exist')

    files = find_target_files(root_dir, args.pattern, args.out_suffix)
    if not files:
        raise FileNotFoundError(f"No files matching '{args.pattern}' found under {root_dir}")

    print(f'Found {len(files)} candidate file(s) under {root_dir}.')

    total_recovered = 0
    truncated_files = []
    skipped_files = []

    for f in files:
        if not looks_like_broken_schema(f):
            skipped_files.append(f)
            print(f'SKIP  {f.relative_to(root_dir)}: header does not match expected broken schema.')
            continue

        if args.dry_run:
            print(f'DRY-RUN  {f.relative_to(root_dir)}: would recover -> {f.stem}{args.out_suffix}{f.suffix}')
            continue

        recovered, truncated_chunks = recover_seed_file(f)
        out_path = f.with_name(f.stem + args.out_suffix + f.suffix)
        recovered.to_csv(out_path, index=False)
        total_recovered += len(recovered)
        print(f'OK    {f.relative_to(root_dir)}: recovered {len(recovered)} episodes -> {out_path.name}')

        if truncated_chunks:
            truncated_files.append((f, truncated_chunks))

    if args.dry_run:
        return

    print(f'\nDone. {len(files) - len(skipped_files)} file(s) converted, '
          f'{len(skipped_files)} skipped, {total_recovered} total episodes recovered.')

    if truncated_files:
        print('\nWARNING: the following files had numpy summarization ellipsis ("...") '
              'in at least one chunk -- some episodes in those chunks are permanently '
              'unrecoverable. Consider re-running those variants with the fixed train_chunk():',
              file=sys.stderr)
        for f, n in truncated_files:
            print(f'  {f.relative_to(root_dir)}: {n} truncated chunk(s)', file=sys.stderr)


if __name__ == '__main__':
    main()