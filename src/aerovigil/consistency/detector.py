r"""AEROVIGIL Module 09: the consistency detector, built on the channels the filter ignores.

Module 08 showed that the Kalman chi-square test cannot see a velocity-consistent drift.
The reason is not tuning. Once the attack has eased in, the spoofed position and the
spoofed velocity agree with each other and imply no extra acceleration, and no extra
acceleration is exactly what the IMU reports. The innovations are not small, they are
zero, so nothing accumulates and no amount of integrating them helps.

Write the spoofed offset as d(t). Then:

    d      is unobservable. The vehicle cannot know where it really is.
    d''    is what the filter watches, and the attacker drives it to zero by easing in
           over twenty seconds instead of one and a half.
    d'     is the rate the vehicle is being displaced, which is the attack itself. It
           cannot be made small without giving up the attack.

So the detector has to watch d', and two channels of the Module 03 v2 features do:

    res_course_heading       GPS course over ground against the attitude yaw. A spoofer
                             dragging the vehicle sideways must claim a direction of
                             travel that its own nose is not pointing along. Measured on
                             a nose-forward flight, this channel moves from 0.003 clean
                             to 0.710 under a 5 m/s drift with a 20 s onset: the slow
                             onset hides nothing, because the channel sees d' and not d''.
    res_horizontal_velocity  reported velocity against the velocity implied by
                             successive positions. A position-only spoofer leaves these
                             disagreeing; a velocity-consistent one does not. This is the
                             complement of the course channel, so between them the two
                             cover both kinds of attacker.

One escape remains and it is worth stating plainly: a drift aimed along the current
course changes the direction of travel very little (0.110 instead of 0.710 in the same
measurement), so it evades the course channel. But an attacker restricted to its
victim's own heading can only push it further along the path it was already flying, and
cannot steer it to a chosen place. Module 10 puts that to the test.

Scores are standardised per channel by the median and MAD of the calibration flights,
and a row alarms when the largest standardised channel passes the threshold, so either
check can fire on its own. The threshold comes from the same 1 alarm per flight-hour
budget Modules 06, 07 and 08 used, on the same calibration flights, so every number in
this project remains comparable.
"""

import numpy as np

COURSE = "res_course_heading_roll_mean"
VELOCITY = "res_horizontal_velocity_roll_mean"
CHANNELS = (COURSE, VELOCITY)
REFRACTORY_S = 10.0
BUDGET_PER_HOUR = 1.0
HZ = 125.0


def channel_values(frame, channels=CHANNELS):
    """The detector's input columns as a (rows, channels) array."""
    missing = [c for c in channels if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"Missing feature columns {missing}. Build the Module 03 v2 features first."
        )
    return frame[list(channels)].to_numpy(float)


def fit_standardiser(values):
    """Per-channel centre and scale from clean flights, by median and MAD."""
    mu = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - mu), axis=0)
    return mu, np.maximum(1.4826 * mad, 1e-9)


def score_rows(values, mu, sd):
    """One score per row: the largest standardised channel, so either check can fire.

    A NaN channel (an IMU dropout, or a stretch too slow for the course check to mean
    anything) contributes nothing rather than blocking the other channel.
    """
    z = np.abs((values - mu) / sd)
    live = np.isfinite(z).any(axis=1)
    best = np.zeros(len(z))
    if live.any():
        # only rows with something to compare, so nanmax never sees an all-NaN slice
        best[live] = np.nanmax(z[live], axis=1)
    return best, live


def alarm_indices(score, t, valid, threshold, refractory_s=REFRACTORY_S):
    """Rows that alarm, counted once per refractory period."""
    out, skip = [], -np.inf
    for i in np.flatnonzero(valid & (score > threshold)):
        if t[i] >= skip:
            out.append(int(i))
            skip = t[i] + refractory_s
    return np.asarray(out, dtype=int)


def monitored_hours(valid, hz=HZ):
    return float(np.count_nonzero(valid)) / hz / 3600.0


def calibrate(clean, budget=BUDGET_PER_HOUR, refractory_s=REFRACTORY_S, grid=None):
    """Lowest threshold whose alarm count on the clean flights fits the budget.

    clean: list of (score, t, valid) tuples, one per calibration flight. Alarms are
    counted per flight, because the statistic resets between flights.
    """
    hours = sum(monitored_hours(v) for _, _, v in clean)
    allowed = max(1, int(round(budget * hours)))
    pool = np.concatenate([s[v] for s, _, v in clean]) if clean else np.zeros(1)
    if grid is None:
        qs = np.linspace(0.90, 1.0, 400)
        grid = np.unique(np.append(np.quantile(pool, qs), pool.max() * 1.01))
    for thr in grid:
        used = sum(len(alarm_indices(s, t, v, thr, refractory_s)) for s, t, v in clean)
        if used <= allowed:
            return float(thr), hours, allowed, used
    return float(grid[-1]), hours, allowed, used


def flight_verdict(score, t, valid, truth, threshold, refractory_s=REFRACTORY_S):
    """Detection, delay and displacement of one flight, the Module 06 way.

    truth: dict with per-row arrays 'label' (the injected window) and 'offset_m'.
    """
    idx = alarm_indices(score, t, valid, threshold, refractory_s)
    on = truth["label"] == 1
    out = {"n_alarms": int(len(idx)), "monitored_hours": monitored_hours(valid)}
    if not on.any():
        return {**out, "attacked": False, "detected": False, "delay_s": np.nan,
                "displacement_m": np.nan, "false_alarms": int(len(idx))}  # fmt: skip
    start, end = float(t[on][0]), float(t[on][-1])
    offset = truth["offset_m"]
    inside = idx[(t[idx] >= start) & (t[idx] <= end)]
    detected = len(inside) > 0
    final = float(np.nanmax(offset[on])) if np.isfinite(offset[on]).any() else np.nan
    return {
        **out, "attacked": True, "detected": bool(detected),
        "delay_s": float(t[inside[0]] - start) if detected else np.nan,
        "displacement_m": float(offset[inside[0]]) if detected else final,
        "final_offset_m": final,
        "false_alarms": int((t[idx] < start).sum()),
    }  # fmt: skip


def per_channel_scores(values, mu, sd):
    """Standardised magnitude of each channel separately, for the report."""
    z = np.abs((values - mu) / sd)
    return {name: z[:, i] for i, name in enumerate(CHANNELS)}
