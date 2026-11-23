"""CUSUM and chi-square (NIS) detectors with thresholds set by a false-alarm budget."""

import numpy as np

W_COLS = ["w_pos_e", "w_vel_e", "w_pos_n", "w_vel_n"]


# ------------------------------------------------------------ standardise --
def fit_standardiser(w):
    """Per-component centre and scale from NORMAL calibration data (median / MAD)."""
    mu = np.median(w, axis=0)
    sd = 1.4826 * np.median(np.abs(w - mu), axis=0)
    return mu, np.maximum(sd, 1e-6)


# ------------------------------------------------------------------ CUSUM --
def cusum_alarms(z, valid, t, k, h, refractory_s):
    """Indices of CUSUM alarms. Two-sided, per component, reset after each alarm."""
    sp = np.zeros(z.shape[1])
    sn = np.zeros(z.shape[1])
    alarms, skip_until = [], -np.inf
    for i in range(len(z)):
        if not valid[i] or t[i] < skip_until:
            sp[:] = 0.0
            sn[:] = 0.0
            continue
        sp = np.maximum(0.0, sp + z[i] - k)
        sn = np.maximum(0.0, sn - z[i] - k)
        if max(sp.max(), sn.max()) > h:
            alarms.append(i)
            sp[:] = 0.0
            sn[:] = 0.0
            skip_until = t[i] + refractory_s
    return np.array(alarms, dtype=int)


def cusum_alarm_counts(z, valid, t, k, hs, refractory_s):
    """Alarm counts for many thresholds in one pass (for calibration)."""
    hs = np.asarray(hs, dtype=float)
    m = z.shape[1]
    sp = np.zeros((len(hs), m))
    sn = np.zeros((len(hs), m))
    skip = np.full(len(hs), -np.inf)
    counts = np.zeros(len(hs), dtype=int)
    for i in range(len(z)):
        if not valid[i]:
            sp[:] = 0.0
            sn[:] = 0.0
            continue
        blocked = t[i] < skip
        sp = np.where(blocked[:, None], 0.0, np.maximum(0.0, sp + z[i] - k))
        sn = np.where(blocked[:, None], 0.0, np.maximum(0.0, sn - z[i] - k))
        fire = (np.maximum(sp.max(axis=1), sn.max(axis=1)) > hs) & ~blocked
        if fire.any():
            counts += fire
            sp[fire] = 0.0
            sn[fire] = 0.0
            skip[fire] = t[i] + refractory_s
    return counts


def calibrate_threshold(counts, grid, allowed):
    """Smallest grid value from which the alarm count stays within the budget."""
    ok = counts <= allowed
    for i in range(len(grid)):
        if ok[i:].all():
            return float(grid[i])
    return float(grid[-1])


# --------------------------------------------------------- chi-square test --
def nis_alarms(nis, valid, t, c, refractory_s):
    """Indices where the single-step NIS exceeds c (one alarm per refractory period)."""
    alarms, skip_until = [], -np.inf
    for i in np.where(valid & (nis > c))[0]:
        if t[i] >= skip_until:
            alarms.append(i)
            skip_until = t[i] + refractory_s
    return np.array(alarms, dtype=int)


def nis_alarm_counts(nis, valid, t, grid, refractory_s):
    return np.array([len(nis_alarms(nis, valid, t, c, refractory_s)) for c in grid])


# --------------------------------------------------------- flight metrics --
def attack_window(df):
    """(start, end) times of the labelled attack, or None."""
    on = df["label"].to_numpy() == 1
    if not on.any():
        return None
    t = df["t_s"].to_numpy()
    return float(t[on][0]), float(t[on][-1])


def flight_metrics(df, alarm_idx, rate_hz, grace_s=0.0):
    """Alarm time, delay, displacement and false alarms of one flight."""
    t = df["t_s"].to_numpy()
    valid = df["valid"].to_numpy()
    win = attack_window(df)
    alarm_t = t[alarm_idx]
    out = {"n_alarms": int(len(alarm_idx))}
    if win is None:
        out.update(
            attacked=False,
            detected=False,
            first_alarm_s=np.nan,
            delay_s=np.nan,
            displacement_m=np.nan,
            false_alarms=int(len(alarm_idx)),
            normal_hours=float(valid.sum() / rate_hz / 3600.0),
        )
        return out
    start, end = win
    pre = alarm_t < start
    inside = alarm_t[(alarm_t >= start) & (alarm_t <= end + grace_s)]
    detected = len(inside) > 0
    first = float(inside[0]) if detected else np.nan
    offset = df["offset_m"].to_numpy(float)
    on = (t >= start) & (t <= end)
    if detected:
        disp = (
            float(offset[np.searchsorted(t, first)])
            if np.isfinite(offset).any()
            else np.nan
        )
    else:
        disp = float(np.nanmax(offset[on])) if np.isfinite(offset[on]).any() else np.nan
    out.update(
        attacked=True,
        detected=detected,
        first_alarm_s=first,
        delay_s=(first - start) if detected else np.nan,
        displacement_m=disp,
        attack_start_s=start,
        attack_end_s=end,
        false_alarms=int(pre.sum()),
        normal_hours=float((valid & (t < start)).sum() / rate_hz / 3600.0),
    )
    return out
