"""Shared helpers for the Module 04 scripts."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}


def resolve(path):
    """Project-relative path (absolute paths are returned unchanged)."""
    return ROOT / path


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_map(path):
    """Read flight_splits.csv and return ({flight_id: split}, id_col, split_col)."""
    df = pd.read_csv(path)
    id_col = (
        "flight_id"
        if "flight_id" in df.columns
        else next(c for c in df.columns if "flight" in c.lower())
    )
    split_col = next(
        (c for c in df.columns if c.lower() in ("split", "set", "partition", "subset")),
        None,
    )
    if split_col is None:
        split_col = next(c for c in df.columns if "split" in c.lower())
    mapping = {
        str(k): SPLIT_ALIASES.get(str(v).strip().lower(), str(v).strip().lower())
        for k, v in zip(df[id_col], df[split_col])
    }
    return mapping, id_col, split_col


def get_columns(att_cfg, pre_cfg):
    """Column mapping (from the Module 03 config) and the time scale."""
    gps = pre_cfg["sensors"]["gps"]
    cols = {
        "flight": pre_cfg["dataset"]["flight_id_column"],
        "time": pre_cfg["dataset"]["timestamp_column"],
        "label": pre_cfg["labels"]["attack_column"],
        "enabled": att_cfg["columns"].get("attack_enabled"),
        "lat": gps["latitude"],
        "lon": gps["longitude"],
        "alt": gps["altitude"],
        "vn": gps["velocity_north"],
        "ve": gps["velocity_east"],
        "vu": gps.get("velocity_up"),
        "speed": att_cfg["columns"].get("gps_speed"),
        "baro": pre_cfg["sensors"]["barometer"]["altitude"],
    }
    scale = UNIT_TO_SECONDS[pre_cfg["dataset"]["timestamp_unit"]]
    return cols, scale
