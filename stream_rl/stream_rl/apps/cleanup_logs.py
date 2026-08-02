import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

RAW_PATTERN = 'monitor_seed_*.csv'
RECOVERED_SUFFIX = '_recovered'


def find_raw_files(root_dir: Path) -> list[Path]:
    all_matches = sorted(root_dir.rglob(RAW_PATTERN))
    return [f for f in all_matches if RECOVERED_SUFFIX not in f.stem]


def recovered_path_for(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.stem + RECOVERED_SUFFIX + raw_path.suffix)


def recovered_is_valid(recovered_path: Path) -> tuple[bool, str]:
    if not recovered_path.exists():
        return False, 'recovered file does not exist'
    try:
        df = pd.read_csv(recovered_path, nrows=1)
    except Exception as e:
        return False, f'recovered file unreadable ({e})'
    if len(pd.read_csv(recovered_path)) == 0:
        return False, 'recovered file has 0 rows (no completed episodes recovered)'
    return True, 'ok'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root_dir', type=str)
    parser.add_argument('--delete', action='store_true',
                         help='Permanently delete originals instead of archiving them')
    parser.add_argument('--yes', action='store_true',
                         help='Actually perform the action (archive/delete). Without this, only lists candidates.')
    parser.add_argument('--force', action='store_true',
                         help='Also act on raw files whose recovered counterpart is empty/invalid (NOT recommended)')
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f'{root_dir} does not exist')

    archive_root = root_dir / '_archived_raw_monitor_logs'
    raw_files = find_raw_files(root_dir)
    if not raw_files:
        print(f"No '{RAW_PATTERN}' files found under {root_dir}.")
        return

    to_process = []
    skipped = []

    for raw in raw_files:
        recovered = recovered_path_for(raw)
        valid, reason = recovered_is_valid(recovered)
        if valid or args.force:
            to_process.append((raw, recovered, reason if not valid else 'ok'))
        else:
            skipped.append((raw, reason))

    action_word = 'DELETE' if args.delete else 'ARCHIVE'
    print(f'Found {len(raw_files)} raw file(s) under {root_dir}.')
    print(f'{len(to_process)} eligible for {action_word.lower()}, {len(skipped)} skipped.\n')

    for raw, reason in skipped:
        print(f'SKIP    {raw.relative_to(root_dir)}: {reason}')

    for raw, recovered, note in to_process:
        tag = 'FORCED' if note != 'ok' else action_word
        print(f'{tag:8s}{raw.relative_to(root_dir)}  (recovered: {recovered.name}, note: {note})')

    if not args.yes:
        print(f'\nDry-run only. Re-run with --yes to actually {action_word.lower()} these {len(to_process)} file(s).')
        if args.delete:
            print('(--delete is permanent; make sure you have verified the recovered CSVs first.)')
        return

    if args.delete and not args.yes:
        print('Refusing to delete without --yes.', file=sys.stderr)
        sys.exit(1)

    for raw, recovered, note in to_process:
        if args.delete:
            raw.unlink()
        else:
            dest = archive_root / raw.relative_to(root_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(raw), str(dest))

    print(f'\nDone. {action_word.title()}d {len(to_process)} file(s).')
    if not args.delete:
        print(f'Originals moved under: {archive_root}')


if __name__ == '__main__':
    main()