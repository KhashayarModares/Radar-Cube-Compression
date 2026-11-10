"""Lets the figure scripts read result tables stored as .csv.gz in results/runs (the release compresses large tables).
Imported for its side effect only; the experiment code never imports it."""
from pathlib import Path

import pandas as pd

_read_csv = pd.read_csv


def read_csv(path, *args, **kwargs):
    if isinstance(path, (str, Path)):
        p = Path(path)
        if not p.exists() and p.with_name(p.name + '.gz').exists():
            path = p.with_name(p.name + '.gz')
    return _read_csv(path, *args, **kwargs)


pd.read_csv = read_csv
