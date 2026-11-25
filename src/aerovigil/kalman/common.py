"""Shared helpers for the Module 06 scripts."""

# ruff: noqa: F401
import json
from pathlib import Path

import numpy as np

from aerovigil.attacks.common import ROOT, UNIT_TO_SECONDS, load_yaml, resolve
from aerovigil.kalman.calibrate_imu import get_columns


def out_dir(cfg):
    p = resolve(cfg["paths"]["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def load_convention(cfg):
    """IMU convention accepted in Step 3 (None means: run without the IMU)."""
    info = load_json(out_dir(cfg) / "imu_convention.json")
    if info is None:
        raise SystemExit("Run calibrate_imu.py first (imu_convention.json is missing).")
    return info["convention"]


def load_noise_scale(cfg):
    info = load_json(out_dir(cfg) / "filter_noise.json")
    if info is None:
        raise SystemExit("Run tune_noise.py first (filter_noise.json is missing).")
    s = info["scale"]
    return {k: float(v) for k, v in s.items()} if isinstance(s, dict) else float(s)


def truth_arrays(g):
    """Ground-truth columns of a Module 04 attacked flight (never filter inputs)."""
    return {
        "label": g["attack_label"].to_numpy(float),
        "offset_e": g["spoof_offset_east_m"].to_numpy(float),
        "offset_n": g["spoof_offset_north_m"].to_numpy(float),
        "offset_m": g["spoof_offset_m"].to_numpy(float),
        "rel_start": g["time_from_attack_start_s"].to_numpy(float),
    }


def orig_truth(g, label_col):
    """Original-dataset flights: labels only, no spoof-offset ground truth."""
    nan = np.full(len(g), np.nan)
    return {
        "label": g[label_col].to_numpy(float),
        "offset_e": nan,
        "offset_n": nan,
        "offset_m": nan,
        "rel_start": nan,
    }
