"""Large per-flight result files: the plain CSV locally, a .gz copy in Git.

A few per-flight results are over the 5 MB limit for committed files, yet every results
table is rebuilt from them. Each is therefore committed as <name>.csv.gz (written
deterministically, so an unchanged result never changes the file), and readers take
the plain CSV when it exists and the .gz copy otherwise. A fresh clone can then rebuild
every table without running the heavy steps.
"""

import gzip
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "data" / "processed"
LARGE = (
    P / "module05_v2" / "all_models_flight_full.csv",
    P / "module07_v2" / "protocols" / "flights.csv",
    P / "evaluation" / "ablation_flights.csv",
    P / "stealth" / "whitebox_flights.csv",
    P / "explain" / "alarms.csv",
)


def gz_of(path):
    path = Path(path)
    return path.with_name(path.name + ".gz")


def path_of(path):
    """The plain file if present, else its .gz copy, else None."""
    path = Path(path)
    if path.is_file():
        return path
    gz = gz_of(path)
    return gz if gz.is_file() else None


def read_csv(path, **kw):
    p = path_of(path)
    return pd.read_csv(p, **kw) if p is not None else None


def publish(paths=LARGE):
    """Refresh the .gz copy of every large result that exists as a plain CSV.
    Returns the copies that changed."""
    changed = []
    for path in paths:
        if not path.is_file():
            continue
        data = gzip.compress(path.read_bytes(), compresslevel=9, mtime=0)
        gz = gz_of(path)
        if gz.is_file() and gz.read_bytes() == data:
            continue
        tmp = gz.with_name(gz.name + ".partial")
        tmp.write_bytes(data)
        tmp.replace(gz)
        changed.append(gz)
    return changed
