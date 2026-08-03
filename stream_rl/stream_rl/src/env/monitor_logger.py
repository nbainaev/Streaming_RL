import csv
from pathlib import Path


class EpisodeMonitorLogger:
    """Generic per-row CSV logger (despite the name, also reused for the
    step-indexed training-metrics log — see `prefix`)."""

    def __init__(self, run_dir, seed: int, prefix: str = 'monitor'):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.path = self.run_dir / f'{prefix}_seed_{seed}.csv'

        # Resuming from a checkpoint (see base_runner.py's save/load_checkpoint)
        # re-opens the same run_dir/seed -- append to the existing CSV instead
        # of truncating it, so history logged before an interruption survives.
        resume = self.path.exists() and self.path.stat().st_size > 0
        self._fieldnames = None
        if resume:
            with open(self.path, 'r', newline='') as f:
                self._fieldnames = next(csv.reader(f), None)

        self._file = open(self.path, 'a' if resume else 'w', newline='')
        self._writer = (
            csv.DictWriter(self._file, fieldnames=self._fieldnames) if self._fieldnames else None
        )

    def _flatten(self, payload: dict, prefix: str = '') -> dict:
        flat = {}
        for key, value in payload.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(value, dict):
                flat.update(self._flatten(value, name))
            else:
                flat[name] = value
        return flat

    def log(self, payload: dict, step: int) -> None:
        row = {'step': step, 'seed': self.seed}
        row.update(self._flatten(payload))
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            for key in row.keys():
                if key not in self._fieldnames:
                    raise ValueError(
                        f'Logger schema changed for seed {self.seed}. Missing column in header: {key}'
                    )
        self._writer.writerow(row)
        self._file.flush()

    def finish(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()