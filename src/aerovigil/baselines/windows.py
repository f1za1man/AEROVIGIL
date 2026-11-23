"""Time-based window summaries of the Module 03 per-flight features.

A window is `window_s` seconds long and moves by `hop_s`. Windows that span a
log gap, or that have too little valid data, are dropped. Every window becomes
one row of summary statistics, which is what the classical models need.
"""
import warnings

import numpy as np

STATS = ("mean", "std", "min", "max", "last", "delta")


def stat_names(features):
    return [f"{f}__{s}" for f in features for s in STATS]


def window_bounds(t, window_s, hop_s):
    """Return window start times and the [a, b) row ranges they cover."""
    t = np.asarray(t, dtype=float)
    last_start = t[-1] - window_s
    if last_start < t[0]:
        return np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)
    starts = t[0] + hop_s * np.arange(int(np.floor((last_start - t[0]) / hop_s + 1e-9)) + 1)
    a = np.searchsorted(t, starts, side="left")
    b = np.searchsorted(t, starts + window_s, side="left")
    return starts, a, b


def _summarise(x, t_win):
    """Nan-aware statistics of one window x (n_rows x n_features)."""
    valid = np.isfinite(x)
    n, k = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        mn = np.nanmin(x, axis=0)
        mx = np.nanmax(x, axis=0)
    any_valid = valid.any(axis=0)
    i_last = np.where(any_valid, n - 1 - np.argmax(valid[::-1], axis=0), 0)
    i_first = np.where(any_valid, np.argmax(valid, axis=0), 0)
    cols = np.arange(k)
    last = x[i_last, cols]
    first = x[i_first, cols]
    dt = t_win[i_last] - t_win[i_first]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(dt > 0, (last - first) / dt, np.nan)
    return np.stack([mean, std, mn, mx, last, delta], axis=1).reshape(-1)  # feature-major


def summarise_flight(feat, t_s, labels, after_gap, window_s, hop_s,
                     min_valid_fraction=0.5, min_span_fraction=0.9):
    """Summarise one flight.

    feat      (n_rows, n_features) array, NaN allowed
    t_s       seconds since flight start (sorted)
    labels    per-row attack label (0/1)
    after_gap per-row flag, True on the first sample after a log gap

    Returns (stats, meta, dropped) where stats is (n_windows, n_features * 6),
    meta is a list of dicts and dropped counts the rejected windows by reason.
    """
    feat = np.asarray(feat, dtype=float)
    feat = np.where(np.isfinite(feat), feat, np.nan)
    t_s = np.asarray(t_s, dtype=float)
    labels = np.asarray(labels, dtype=float)
    after_gap = np.asarray(after_gap, dtype=bool)

    starts, a, b = window_bounds(t_s, window_s, hop_s)
    rows, meta = [], []
    dropped = {"too_few_rows": 0, "spans_gap": 0, "short_span": 0, "too_many_nan": 0}
    for ws, i0, i1 in zip(starts, a, b):
        if i1 - i0 < 2:
            dropped["too_few_rows"] += 1
            continue
        if after_gap[i0 + 1:i1].any():
            dropped["spans_gap"] += 1
            continue
        if t_s[i1 - 1] - t_s[i0] < min_span_fraction * window_s:
            dropped["short_span"] += 1
            continue
        x = feat[i0:i1]
        if (np.isfinite(x).mean(axis=0) < min_valid_fraction).any():
            dropped["too_many_nan"] += 1
            continue
        rows.append(_summarise(x, t_s[i0:i1]))
        meta.append({
            "window_start_s": float(ws), "window_end_s": float(ws + window_s),
            "last_row": int(i1 - 1), "n_rows": int(i1 - i0),
            "attack_fraction": float(labels[i0:i1].mean()),
        })
    stats = np.vstack(rows) if rows else np.empty((0, feat.shape[1] * len(STATS)))
    return stats, meta, dropped


def label_window(fraction, positive_min_fraction):
    """Return (label, ambiguous). Fraction 0 -> normal; >= threshold -> attack;
    anything in between is ambiguous and excluded from training and scoring."""
    if fraction == 0.0:
        return 0, False
    if fraction >= positive_min_fraction:
        return 1, False
    return -1, True