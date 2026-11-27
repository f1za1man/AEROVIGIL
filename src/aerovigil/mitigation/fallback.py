r"""Plan M12 (Step G): dead reckoning after a spoofing alarm, on the logged flights.

The fallback is Module 06's own Kalman filter with GPS taken away. Until the alarm it
runs as usual (GPS position and velocity, IMU acceleration as the input, barometer
for height), and it saves its state after every 0.1 s tick. When the alarm fires:

    1. roll back: restore the state saved `rollback_s` seconds before the alarm, so
       the last seconds of GPS, which may already be spoofed, are forgotten;
    2. coast horizontally: predict only, driven by the IMU acceleration minus the
       accelerometer bias the filter had learned while GPS was still trusted;
    3. keep the barometer: the vertical filter goes on updating with the barometer
       alone (it is not spoofed), so height stays aided.

Heading: the IMU acceleration is turned into East-North-Up with the logged attitude
(PX4's estimate, which may itself lean on GPS). The "mag" variant protects the heading
with the magnetometer: the Earth's field, turned into the world frame by the logged
attitude, should keep a fixed direction; after the alarm, any turn of that direction
away from its value over the 10 s before the alarm is taken as a heading error and
the acceleration is turned back by it.

Ground truth is the unspoofed GPS (`raw_gps`, in the base tables as raw_latitude_deg
...) for the dataset's own flights, and the source flight's GPS for Module 04's
synthetic attacks (same rows; the attack only moved the GPS). Errors are horizontal
distances between an estimate and that truth, in the filter's local frame.

    without fallback   the filter as it runs, still trusting the (spoofed) GPS
    with fallback      the dead-reckoned estimate from the alarm on
"""

import math

import numpy as np

from aerovigil.kalman.imu import rotate
from aerovigil.kalman.kalman import (
    H_HORIZONTAL,
    H_VERTICAL,
    KalmanFilter,
    horizontal_matrices,
    noise_matrices,
    vertical_matrices,
)
from aerovigil.kalman.pipeline import EARTH_R, prepare_ticks

MAG_COLS = ["magnetic_field_x", "magnetic_field_y", "magnetic_field_z"]
# The magnetometer logs in PX4's body axes (forward, right, down); the attitude
# quaternion turns forward-left-up into ENU. Measured on 12 normal training flights:
# with y and z flipped the world-frame field keeps one direction to 1.7 deg (circular
# sd, every flight) and dips 63 deg below the horizon, as the Earth's field does at
# the simulator's location (47.4 N, 8.5 E); every other axis choice spreads >= 8.9 deg.
MAG_TO_FLU = np.array([1.0, -1.0, -1.0])
MAG_REF_S = 10.0  # the field direction before the alarm is averaged over this long
H_BARO = H_VERTICAL[[1]]  # barometer = height + barometer bias


# --------------------------------------------------------------- the flight --


class Ticks:
    """One flight cut into Module 06's filter ticks, with truth and field direction."""

    def __init__(self, g, ctx, truth=None, source=None):
        """truth: (lat, lon, alt) arrays on g's rows; default g's raw_gps columns.
        source: the clean flight a synthetic attack was made from (same rows); its
        magnetometer is used where g has none (Module 04 only moved the GPS)."""
        if source is not None and not set(MAG_COLS) <= set(g.columns):
            g = g.assign(**{c: source[c].to_numpy() for c in MAG_COLS})
        keep = g["aligned_all"].to_numpy() == 1
        self.g = g[keep].reset_index(drop=True)
        if truth is None:
            truth = tuple(g[c].to_numpy(float) for c in
                          ("raw_latitude_deg", "raw_longitude_deg", "raw_altitude"))  # fmt: skip
        lat, lon, alt = (np.asarray(a, float)[keep] for a in truth)
        gk = self.g
        lat0 = gk["latitude_deg"].iloc[0]
        lon0 = gk["longitude_deg"].iloc[0]
        alt0 = gk["altitude"].iloc[0]
        self.truth = np.column_stack([
            EARTH_R * np.radians(lon - lon0) * np.cos(np.radians(lat0)),
            EARTH_R * np.radians(lat - lat0),
            alt - alt0,
        ])  # fmt: skip
        self.ctx = ctx
        self.segs = prepare_ticks(gk, ctx.cols, ctx.fcfg, ctx.conv, ctx.scale)
        t_s = gk["t_s"].to_numpy(float)
        psi = field_direction(gk, ctx.cols["quat"])
        cs = np.vstack([[0.0, 0.0], np.cumsum(np.column_stack(
            [np.cos(psi), np.sin(psi)]), axis=0)])  # fmt: skip
        for seg in self.segs:
            b = seg["rows"]
            prev = np.concatenate([[seg["s0"]], b[:-1]])
            m = (cs[b + 1] - cs[prev + 1]) / (b - prev)[:, None]
            seg["psi"] = np.arctan2(m[:, 1], m[:, 0])
            seg["t_s"] = t_s[b]
            seg["truth"] = self.truth[b]


def field_direction(g, quat_cols):
    """Direction (rad, from East) of the magnetic field turned into the world frame
    by the logged attitude, on every row; NaN where the magnetometer is missing."""
    m = g[MAG_COLS].to_numpy(float) * MAG_TO_FLU
    q = g[quat_cols].to_numpy(float)
    ok = np.isfinite(m).all(axis=1) & np.isfinite(q).all(axis=1)
    out = np.full(len(g), np.nan)
    if ok.any():
        w = rotate(m[ok], q[ok], "body_to_world")
        out[ok] = np.arctan2(w[:, 1], w[:, 0])
    return out


# ---------------------------------------------------------------- filtering --


def _filters(seg, fcfg):
    i = seg["init"]
    kf_h = KalmanFilter(
        [[i["e"], i["n"]], [i["ve"], i["vn"]], [0.0, 0.0]],
        np.diag([fcfg["p0_pos"] ** 2, fcfg["p0_vel"] ** 2, fcfg["p0_bias"] ** 2]),
    )
    kf_v = KalmanFilter(
        [[i["u"]], [i["vu"]], [0.0], [i["baro"] - i["u"]]],
        np.diag([fcfg["p0_pos"] ** 2, fcfg["p0_vel"] ** 2, fcfg["p0_bias"] ** 2,
                 fcfg["p0_baro_bias"] ** 2]),
    )  # fmt: skip
    return kf_h, kf_v


def run_normal(seg, fcfg, noise):
    """Module 06's filter_segment loop, saving the state after every tick.

    Returns est (n, 3) East, North, Up estimates and states: a list of n + 1 tuples
    (x_h, p_h, x_v, p_v), states[0] the initial state and states[k + 1] the one
    after tick k."""
    r_h, r_v = noise_matrices(fcfg, noise)
    kf_h, kf_v = _filters(seg, fcfg)
    n = len(seg["t"])
    est = np.empty((n, 3))
    states = [(kf_h.x.copy(), kf_h.p.copy(), kf_v.x.copy(), kf_v.p.copy())]
    for k in range(n):
        dt = seg["dt"][k]
        f, b, q = horizontal_matrices(dt, fcfg["sigma_acc"], fcfg["sigma_bias_walk"])
        kf_h.predict(f, q, b, seg["acc"][k, [0, 1]][None, :])
        kf_h.update([[seg["e"][k], seg["n"][k]], [seg["ve"][k], seg["vn"][k]]],
                    H_HORIZONTAL, r_h)  # fmt: skip
        fv, bv, qv = vertical_matrices(dt, fcfg["sigma_acc"], fcfg["sigma_bias_walk"],
                                       fcfg["sigma_baro_bias_walk"])  # fmt: skip
        kf_v.predict(fv, qv, bv, seg["acc"][k, [2]][None, :])
        kf_v.update([[seg["u"][k]], [seg["baro"][k]], [seg["vu"][k]]], H_VERTICAL, r_v)
        est[k] = (kf_h.x[0, 0], kf_h.x[0, 1], kf_v.x[0, 0])
        states.append((kf_h.x.copy(), kf_h.p.copy(), kf_v.x.copy(), kf_v.p.copy()))
    return est, states


def coast(seg, states, fcfg, noise, start, stop, heading=None):
    """Dead reckoning on ticks start .. stop - 1, from the state saved before tick
    `start` (states[start]). heading: optional per-tick angle (rad) to turn the
    horizontal acceleration by, or "hold" to ignore the IMU and hold the last
    velocity (a baseline). Returns (stop - start, 3) estimates."""
    _, r_v = noise_matrices(fcfg, noise)
    r_baro = r_v[[1]][:, [1]]
    x_h, p_h, x_v, p_v = states[start]
    kf_h = KalmanFilter(x_h.copy(), p_h.copy())
    kf_v = KalmanFilter(x_v.copy(), p_v.copy())
    out = np.empty((stop - start, 3))
    for j, k in enumerate(range(start, stop)):
        dt = seg["dt"][k]
        a = seg["acc"][k]
        ae, an = a[0], a[1]
        if isinstance(heading, str):  # "hold": input = the bias, velocity held
            ae, an = kf_h.x[2, 0], kf_h.x[2, 1]
        elif heading is not None and np.isfinite(heading[k]):
            c, s = math.cos(heading[k]), math.sin(heading[k])
            ae, an = c * ae - s * an, s * ae + c * an
        f, b, q = horizontal_matrices(dt, fcfg["sigma_acc"], fcfg["sigma_bias_walk"])
        kf_h.predict(f, q, b, np.array([[ae, an]]))
        fv, bv, qv = vertical_matrices(dt, fcfg["sigma_acc"], fcfg["sigma_bias_walk"],
                                       fcfg["sigma_baro_bias_walk"])  # fmt: skip
        kf_v.predict(fv, qv, bv, a[[2]][None, :])
        kf_v.update([[seg["baro"][k]]], H_BARO, r_baro)
        out[j] = (kf_h.x[0, 0], kf_h.x[0, 1], kf_v.x[0, 0])
    return out


def heading_fix(seg, switch_k):
    """Per-tick heading correction from the field direction: zero before the switch,
    then (direction over the MAG_REF_S before the switch) - (direction now)."""
    psi = seg["psi"]
    t = seg["t_s"]
    ref_mask = (t > t[switch_k] - MAG_REF_S) & (t <= t[switch_k]) & np.isfinite(psi)
    fix = np.zeros(len(psi))
    if not ref_mask.any():
        return fix
    ref = math.atan2(np.sin(psi[ref_mask]).mean(), np.cos(psi[ref_mask]).mean())
    d = (ref - psi + np.pi) % (2 * np.pi) - np.pi
    fix[switch_k:] = np.where(np.isfinite(d[switch_k:]), d[switch_k:], 0.0)
    return fix


# -------------------------------------------------------------- one switch --


def fallback_run(ticks, switch_t, rollback_s, variant="imu", horizon_s=None):
    """Errors with and without the fallback after a switch at base time switch_t.

    Returns a dict of arrays on the ticks from the switch on (or None when the switch
    falls outside every filter segment): since_s, err_with, err_without (horizontal,
    m), verr_with, verr_without (vertical, m), plus the segment's end time."""
    ctx = ticks.ctx
    for seg in ticks.segs:
        t = seg["t_s"]
        if not len(t) or switch_t < t[0] or switch_t > t[-1]:
            continue
        s = int(np.searchsorted(t, switch_t, side="left"))  # first tick at/after
        r = int(np.searchsorted(t, switch_t - rollback_s, side="right"))  # restart
        r = min(r, s)
        stop = len(t)
        if horizon_s is not None:
            stop = int(np.searchsorted(t, switch_t + horizon_s, side="right"))
        if s >= stop:
            return None
        if "normal" not in seg:
            seg["normal"] = run_normal(seg, ctx.fcfg, ctx.noise)
        est, states = seg["normal"]
        heading = {"mag": lambda: heading_fix(seg, s), "hold": lambda: "hold",
                   "imu": lambda: None}[variant]()  # fmt: skip
        dr = coast(seg, states, ctx.fcfg, ctx.noise, r, stop, heading)[s - r :]
        truth = seg["truth"][s:stop]
        return {
            "since_s": t[s:stop] - switch_t,
            "err_with": np.hypot(*(dr[:, :2] - truth[:, :2]).T),
            "err_without": np.hypot(*(est[s:stop, :2] - truth[:, :2]).T),
            "verr_with": np.abs(dr[:, 2] - truth[:, 2]),
            "verr_without": np.abs(est[s:stop, 2] - truth[:, 2]),
            "segment_end_s": float(t[-1] - switch_t),
        }
    return None


def first_exceed(since, err, limit):
    """(time the error first goes above limit, True) or (last time seen, False)."""
    over = np.nonzero(err > limit)[0]
    if len(over):
        return float(since[over[0]]), True
    return float(since[-1]) if len(since) else 0.0, False


def kaplan_meier(times, events, grid):
    """Fraction still within the limit at each grid time (right-censored data)."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    out = np.ones(len(grid))
    surv = 1.0
    order = np.argsort(times, kind="stable")
    times, events = times[order], events[order]
    uniq = np.unique(times[events])
    steps = []
    for u in uniq:
        at_risk = int((times >= u).sum())
        d = int(((times == u) & events).sum())
        if at_risk:
            surv *= 1.0 - d / at_risk
        steps.append((u, surv))
    for i, x in enumerate(grid):
        out[i] = next((s for u, s in reversed(steps) if u <= x), 1.0)
    if len(times):  # beyond the longest observed time nothing is known
        out[np.asarray(grid) > times.max()] = np.nan
    return out


def crossover(since, err_with, err_without, hold_s=5.0):
    """(first time the fallback's error is above the no-fallback error and stays above
    for hold_s, True), or (last time seen, False) when that never happens."""
    worse = err_with > err_without
    i = 0
    n = len(since)
    while i < n:
        if worse[i]:
            j = i
            while j < n and worse[j]:
                j += 1
            if since[j - 1] - since[i] >= hold_s:
                return float(since[i]), True
            i = j
        else:
            i += 1
    return float(since[-1]) if n else 0.0, False


def km_quantile(times, events, q):
    """Earliest time by which a fraction q has gone above the limit (NaN if never
    reached within the data)."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    grid = np.unique(times[events])
    if not len(grid):
        return np.nan
    s = kaplan_meier(times, events, grid)
    hit = np.nonzero(s <= 1.0 - q)[0]
    return float(grid[hit[0]]) if len(hit) else np.nan
