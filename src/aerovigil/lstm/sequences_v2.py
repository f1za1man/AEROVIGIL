r"""AEROVIGIL Module 07 v2: binned window sequences on the Module 03 v2 features.

The same windows as Module 05 v2 (5 s long, 2.5 s hop, identical drop rules) and the
same measured phases, but each window becomes a short sequence instead of six summary
statistics per feature: its rows are averaged into BIN_S bins, so an LSTM sees
(20, 36) rather than 625 x 36.

What changes against Module 07 v1:
  * 36 v2 features instead of 22, and they arrive already normalized by Module 03 v2.
    v1 fitted its own scaler on top of them, which is a second normalization;
  * the label is y_event (ramp or snapback against normal or pre), not the Module 05
    whole-window label. That label called the 256 s curved plateau 102 separate attack
    windows, and the plateau is the part of the attack the features cannot see;
  * plateau and post windows are still returned, with y_event -1, so the alarm
    timeline of a flight stays complete. They never enter training or the window AUC.
"""

import numpy as np
import pandas as pd

from aerovigil.baselines.windows import window_bounds
from aerovigil.windowing.make_windows_v2 import SECONDS, STEP_S, window_class

BIN_S = 0.25
T_STEPS = round(SECONDS / BIN_S)
MIN_VALID_FRACTION = 0.5
MIN_SPAN_FRACTION = 0.9
MAX_MISSING_BINS = 0.10
EVENT = ("ramp", "snapback")
CLEAN = ("normal", "pre")
IMU_PREFIX = ("res_accel", "res_course")
META_COLS = [
    "flight_id", "source_flight_id", "origin", "attack_set", "split", "role",
    "mission_type", "attack_type", "velocity_mode", "drift_rate_mps",
    "window_start_s", "window_end_s",
    "n_rows", "missing_fraction", "attack_fraction", "spoof_fraction", "window_class",
    "y_event", "y_window", "max_offset_m", "offset_end_m", "seconds_since_onset_end",
]  # fmt: skip


def bin_means(x, t, bin_s=BIN_S):
    """NaN-aware mean of every feature in consecutive bins, plus rows per bin."""
    n_bins = int(np.floor((t[-1] - t[0]) / bin_s + 1e-9)) + 1
    idx = np.minimum(np.floor((t - t[0]) / bin_s + 1e-9).astype(int), n_bins - 1)
    finite = np.isfinite(x)
    means = np.full((n_bins, x.shape[1]), np.nan)
    for j in range(x.shape[1]):
        s = np.bincount(
            idx, weights=np.where(finite[:, j], x[:, j], 0.0), minlength=n_bins
        )
        c = np.bincount(idx, weights=finite[:, j].astype(float), minlength=n_bins)
        np.divide(s, c, out=means[:, j], where=c > 0)
    return means, np.bincount(idx, minlength=n_bins)


def flight_sequences(g, features, info, overlap_only=False):
    """Sequences and metadata for every kept window of one flight.

    Returns (X (n, T_STEPS, k) float32, meta DataFrame, dropped counts). Missing bins
    are filled with 0.0, which is the median of a normalized feature, and the share
    filled is kept in missing_fraction so a report can check it.
    """
    t = g["t_s"].to_numpy(float)
    x = g[features].to_numpy(float)
    x = np.where(np.isfinite(x), x, np.nan)
    labels = g["attack_label"].to_numpy(float)
    after_gap = g["after_gap"].to_numpy(bool)
    ph = g["phase"].to_numpy()
    truth = g["spoof_truth"].to_numpy()
    offset = g["offset_m"].to_numpy(float)
    since = g["seconds_since_onset"].to_numpy(float)
    starts, a, b = window_bounds(t, SECONDS, STEP_S)
    dropped = {"too_few_rows": 0, "spans_gap": 0, "short_span": 0, "too_many_nan": 0,
               "empty_bin": 0, "no_overlap": 0}  # fmt: skip
    rows, meta = [], []
    for ws, i0, i1 in zip(starts, a, b):
        if i1 - i0 < 2:
            dropped["too_few_rows"] += 1
            continue
        if after_gap[i0 + 1 : i1].any():
            dropped["spans_gap"] += 1
            continue
        if t[i1 - 1] - t[i0] < MIN_SPAN_FRACTION * SECONDS:
            dropped["short_span"] += 1
            continue
        win = x[i0:i1]
        if (np.isfinite(win).mean(axis=0) < MIN_VALID_FRACTION).any():
            dropped["too_many_nan"] += 1
            continue
        frac = float(labels[i0:i1].mean())
        if overlap_only and frac == 0.0:
            dropped["no_overlap"] += 1
            continue
        means, per_bin = bin_means(win, t[i0:i1])
        if len(means) < T_STEPS or (per_bin[:T_STEPS] == 0).any():
            dropped["empty_bin"] += 1
            continue
        seq = means[:T_STEPS]
        missing = float(np.isnan(seq).mean())
        if missing > MAX_MISSING_BINS:
            dropped["too_many_nan"] += 1
            continue
        rows.append(np.nan_to_num(seq, nan=0.0))
        cls = window_class(ph[i0:i1])
        meta.append({
            "window_start_s": float(ws), "window_end_s": float(ws + SECONDS),
            "n_rows": int(i1 - i0), "missing_fraction": missing,
            "attack_fraction": frac,
            "spoof_fraction": float((truth[i0:i1] == 1).mean()),
            "window_class": cls,
            "y_event": 1 if cls in EVENT else (0 if cls in CLEAN else -1),
            "y_window": 1 if frac >= 0.5 else (0 if frac == 0.0 else -1),
            "max_offset_m": float(np.nanmax(offset[i0:i1])),
            "offset_end_m": float(offset[i1 - 1]),
            "seconds_since_onset_end": float(since[i1 - 1]),
            **info,
        })  # fmt: skip
    if not rows:
        empty = np.empty((0, T_STEPS, len(features)), dtype=np.float32)
        return empty, pd.DataFrame(columns=META_COLS), dropped
    frame = pd.DataFrame(meta)
    for c in META_COLS:
        if c not in frame.columns:
            frame[c] = np.nan
    return np.stack(rows).astype(np.float32), frame[META_COLS], dropped
