"""Turn per-flight feature tables into fixed-length window sequences for the LSTM models.

A window is `window_s` long and moves by `hop_s` (the same geometry and the same drop rules as
Module 05). Each window is averaged into bins of `bin_s` seconds, so the LSTM sees a short
sequence of shape (window_s / bin_s, n_features) instead of 625 raw rows.
"""

import numpy as np

from aerovigil.baselines.windows import label_window, window_bounds


def bin_means(feat, t_s, bin_s):
    """NaN-aware mean of every feature in consecutive bins of bin_s seconds.

    Returns (means (n_bins, k), rows_per_bin (n_bins,)). Bins without rows are NaN.
    """
    feat = np.asarray(feat, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    n_bins = int(np.floor((t_s[-1] - t_s[0]) / bin_s + 1e-9)) + 1
    idx = np.minimum(np.floor((t_s - t_s[0]) / bin_s + 1e-9).astype(int), n_bins - 1)
    finite = np.isfinite(feat)
    means = np.full((n_bins, feat.shape[1]), np.nan)
    for j in range(feat.shape[1]):
        s = np.bincount(
            idx, weights=np.where(finite[:, j], feat[:, j], 0.0), minlength=n_bins
        )
        c = np.bincount(idx, weights=finite[:, j].astype(float), minlength=n_bins)
        np.divide(s, c, out=means[:, j], where=c > 0)
    return means, np.bincount(idx, minlength=n_bins)


def flight_windows(
    feat,
    t_s,
    labels,
    after_gap,
    window_s,
    hop_s,
    bin_s,
    max_missing=0.10,
    min_span_fraction=0.9,
    positive_min_fraction=0.5,
):
    """All valid windows of one flight.

    feat (n_rows, k) NaN allowed | t_s seconds since flight start | labels per-row 0/1
    after_gap per-row flag, True on the first sample after a log gap.

    Returns (X (n_windows, T, k) float32 with NaN, meta list of dicts, dropped counts).
    Every window of the flight is returned, including ambiguous ones (y = -1).
    """
    T, step = int(round(window_s / bin_s)), int(round(hop_s / bin_s))
    if abs(T * bin_s - window_s) > 1e-9 or abs(step * bin_s - hop_s) > 1e-9:
        raise ValueError(
            f"bin_s={bin_s} must divide the window ({window_s}) and the hop ({hop_s})"
        )
    feat = np.asarray(feat, dtype=float)
    feat = np.where(np.isfinite(feat), feat, np.nan)
    t = np.asarray(t_s, dtype=float)
    labels = np.asarray(labels, dtype=float)
    gap = np.asarray(after_gap, dtype=bool)
    starts, a, b = window_bounds(t, window_s, hop_s)
    dropped = {
        "too_few_rows": 0,
        "spans_gap": 0,
        "short_span": 0,
        "empty_bin": 0,
        "too_many_nan": 0,
    }
    if len(starts) == 0:
        return np.empty((0, T, feat.shape[1]), dtype=np.float32), [], dropped
    means, rows = bin_means(feat, t, bin_s)
    cum_gap = np.concatenate([[0], np.cumsum(gap)])
    cum_lab = np.concatenate([[0.0], np.cumsum(labels)])
    xs, meta = [], []
    for k, (ws, i0, i1) in enumerate(zip(starts, a, b)):
        if i1 - i0 < 2:
            dropped["too_few_rows"] += 1
            continue
        if cum_gap[i1] - cum_gap[i0 + 1] > 0:
            dropped["spans_gap"] += 1
            continue
        if t[i1 - 1] - t[i0] < min_span_fraction * window_s:
            dropped["short_span"] += 1
            continue
        first = int(round((ws - t[0]) / bin_s))
        xw = means[first : first + T]
        if len(xw) < T or (rows[first : first + T] == 0).any():
            dropped["empty_bin"] += 1
            continue
        if np.isnan(xw).mean() > max_missing:
            dropped["too_many_nan"] += 1
            continue
        frac = float((cum_lab[i1] - cum_lab[i0]) / (i1 - i0))
        y, amb = label_window(frac, positive_min_fraction)
        xs.append(xw.astype(np.float32))
        meta.append(
            {
                "window_idx": int(k),
                "window_start_s": float(ws),
                "window_end_s": float(ws + window_s),
                "last_row": int(i1 - 1),
                "attack_fraction": frac,
                "y": int(y),
                "ambiguous": bool(amb),
            }
        )
    X = np.stack(xs) if xs else np.empty((0, T, feat.shape[1]), dtype=np.float32)
    return X, meta, dropped


# ----------------------------------------------------------------------------- scaling
def fit_scaler(X, clip=8.0):
    """Centre and scale from NORMAL TRAINING windows: median and winsorised (1-99%) std."""
    v = np.asarray(X, dtype=float).reshape(-1, X.shape[-1])
    center = np.nanmedian(v, axis=0)
    lo, hi = np.nanpercentile(v, [1, 99], axis=0)
    scale = np.array(
        [np.nanstd(np.clip(v[:, j], lo[j], hi[j])) for j in range(v.shape[1])]
    )
    return {
        "center": center.tolist(),
        "scale": np.maximum(scale, 1e-6).tolist(),
        "clip": float(clip),
    }


def apply_scaler(X, sc):
    """Standardise, clip to +-clip and replace missing values by 0 (the normal level)."""
    c = np.asarray(sc["center"], dtype=np.float32)
    s = np.asarray(sc["scale"], dtype=np.float32)
    out = (np.asarray(X, dtype=np.float32) - c) / s
    out = np.clip(out, -sc["clip"], sc["clip"])
    return np.nan_to_num(out, nan=0.0).astype(np.float32)
