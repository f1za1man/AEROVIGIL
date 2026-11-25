"""Turn one flight into filter ticks, run the Kalman filters, return innovations."""

import numpy as np
import pandas as pd

from aerovigil.kalman.imu import world_accel
from aerovigil.kalman.kalman import (
    H_HORIZONTAL,
    H_VERTICAL,
    KalmanFilter,
    horizontal_matrices,
    noise_matrices,
    vertical_matrices,
)

EARTH_R = 6371000.0


def flight_arrays(g, cols, scale):
    """Time in seconds and local ENU measurements for a whole flight."""
    t_us = g[cols["time"]].to_numpy(dtype="int64")
    t = (t_us - t_us[0]) * scale
    lat = g[cols["lat"]].to_numpy(float)
    lon = g[cols["lon"]].to_numpy(float)
    alt = g[cols["alt"]].to_numpy(float)
    return {
        "t": t,
        "e": EARTH_R * np.radians(lon - lon[0]) * np.cos(np.radians(lat[0])),
        "n": EARTH_R * np.radians(lat - lat[0]),
        "u": alt - alt[0],
        "baro": g[cols["baro"]].to_numpy(float) - alt[0],
        "ve": g[cols["ve"]].to_numpy(float),
        "vn": g[cols["vn"]].to_numpy(float),
        "vu": g[cols["vu"]].to_numpy(float),
    }


def segments(t, gap_factor):
    """Row ranges separated by log gaps (same large-gap rule as Module 03)."""
    step = np.diff(t)
    med = np.median(step[step > 0])
    cut = np.where(step > gap_factor * med)[0] + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(t)]])
    return list(zip(starts, ends))


def prepare_ticks(g, cols, fcfg, conv, scale):
    """Resample a flight to filter ticks, one list entry per gap-free segment.

    Each tick carries the mean world-frame acceleration of the IMU rows since the
    previous tick (the filter input) and the latest GPS and barometer values."""
    d = flight_arrays(g, cols, scale)
    t = d["t"]
    if conv is not None:
        a = world_accel(
            g[cols["accel"]].to_numpy(float),
            g[cols["quat"]].to_numpy(float),
            conv,
            fcfg["gravity"],
        )
    else:
        a = np.zeros((len(t), 3))
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(a, axis=0)])
    dt_f = 1.0 / fcfg["rate_hz"]
    out = []
    for seg_id, (s0, s1) in enumerate(segments(t, fcfg["gap_factor"])):
        span = t[s1 - 1] - t[s0]
        if span < fcfg["min_segment_s"]:
            continue
        tick_t = t[s0] + dt_f * np.arange(1, int(np.floor(span / dt_f)) + 1)
        b = np.searchsorted(t[s0:s1], tick_t, side="right") - 1 + s0
        prev = np.concatenate([[s0], b[:-1]])
        keep = b > prev  # ticks that contain new rows
        b, prev = b[keep], prev[keep]
        if len(b) == 0:
            continue
        count = (b - prev)[:, None]
        out.append(
            {
                "seg": seg_id,
                "s0": s0,
                "rows": b,
                "t0": t[s0],
                "t": t[b],
                "dt": t[b] - t[prev],
                "acc": (cs[b + 1] - cs[prev + 1]) / count,
                "e": d["e"][b],
                "n": d["n"][b],
                "ve": d["ve"][b],
                "vn": d["vn"][b],
                "u": d["u"][b],
                "vu": d["vu"][b],
                "baro": d["baro"][b],
                "init": {
                    k: d[k][s0] for k in ("e", "n", "ve", "vn", "u", "vu", "baro")
                },
            }
        )
    return out


def filter_segment(seg, fcfg, noise_scale=1.0, use_imu=True):
    """Run the horizontal (E, N) and vertical Kalman filters over one segment."""
    r_h, r_v = noise_matrices(fcfg, noise_scale)
    sig_acc = fcfg["sigma_acc"] if use_imu else fcfg["sigma_acc_no_imu"]
    i = seg["init"]
    kf_h = KalmanFilter(
        [[i["e"], i["n"]], [i["ve"], i["vn"]], [0.0, 0.0]],
        np.diag([fcfg["p0_pos"] ** 2, fcfg["p0_vel"] ** 2, fcfg["p0_bias"] ** 2]),
    )
    kf_v = KalmanFilter(
        [[i["u"]], [i["vu"]], [0.0], [i["baro"] - i["u"]]],
        np.diag(
            [
                fcfg["p0_pos"] ** 2,
                fcfg["p0_vel"] ** 2,
                fcfg["p0_bias"] ** 2,
                fcfg["p0_baro_bias"] ** 2,
            ]
        ),
    )
    k_n = len(seg["t"])
    white = np.empty((k_n, 4))
    nis_h = np.empty(k_n)
    nis_v = np.empty(k_n)
    nu = np.empty((k_n, 4))
    est = np.empty((k_n, 2))
    for k in range(k_n):
        dt = seg["dt"][k]
        f, b, q = horizontal_matrices(dt, sig_acc, fcfg["sigma_bias_walk"])
        kf_h.predict(f, q, b, seg["acc"][k, [0, 1]][None, :])
        nu_h, _, w_h, nis = kf_h.update(
            [[seg["e"][k], seg["n"][k]], [seg["ve"][k], seg["vn"][k]]],
            H_HORIZONTAL,
            r_h,
        )
        white[k] = [
            w_h[0, 0],
            w_h[1, 0],
            w_h[0, 1],
            w_h[1, 1],
        ]  # pos_e, vel_e, pos_n, vel_n
        nu[k] = [nu_h[0, 0], nu_h[1, 0], nu_h[0, 1], nu_h[1, 1]]
        nis_h[k] = nis.sum()
        est[k] = kf_h.x[0]

        fv, bv, qv = vertical_matrices(
            dt, sig_acc, fcfg["sigma_bias_walk"], fcfg["sigma_baro_bias_walk"]
        )
        kf_v.predict(fv, qv, bv, seg["acc"][k, [2]][None, :])
        _, _, _, nis = kf_v.update(
            [[seg["u"][k]], [seg["baro"][k]], [seg["vu"][k]]], H_VERTICAL, r_v
        )
        nis_v[k] = nis[0]
    return {"white": white, "nu": nu, "nis_h": nis_h, "nis_v": nis_v, "est": est}


def run_flight(g, cols, fcfg, conv, scale, noise_scale=1.0, truth=None):
    """Return one row per filter tick with innovations, NIS, estimates and labels.
    truth: optional dict with per-row arrays 'label', 'offset_e', 'offset_n', 'offset_m',
    'rel_start' taken from the attacked flight (ground truth, never a filter input)."""
    if "aligned_all" in g.columns and not g["aligned_all"].all():
        # Module 03 v2 keeps rows where the IMU or barometer dropped out, as NaN.
        # The filter cannot carry a NaN, so those rows come out before anything else.
        keep = g["aligned_all"].to_numpy() == 1
        g = g[keep].reset_index(drop=True)
        if truth is not None:
            truth = {k: v[keep] for k, v in truth.items()}
    use_imu = conv is not None
    frames = []
    for seg in prepare_ticks(g, cols, fcfg, conv, scale):
        r = filter_segment(seg, fcfg, noise_scale, use_imu)
        df = pd.DataFrame(
            {
                "t_s": seg["t"],
                "segment": seg["seg"],
                "valid": (seg["t"] - seg["t0"]) >= fcfg["warmup_s"],
                "w_pos_e": r["white"][:, 0],
                "w_vel_e": r["white"][:, 1],
                "w_pos_n": r["white"][:, 2],
                "w_vel_n": r["white"][:, 3],
                "nu_pos_e": r["nu"][:, 0],
                "nu_vel_e": r["nu"][:, 1],
                "nu_pos_n": r["nu"][:, 2],
                "nu_vel_n": r["nu"][:, 3],
                "nis_h": r["nis_h"],
                "nis_v": r["nis_v"],
                "est_e": r["est"][:, 0],
                "est_n": r["est"][:, 1],
                "gps_e": seg["e"],
                "gps_n": seg["n"],
            }
        )
        if truth is not None:
            rows = seg["rows"]
            df["label"] = truth["label"][rows]
            df["offset_e"] = truth["offset_e"][rows]
            df["offset_n"] = truth["offset_n"][rows]
            df["offset_m"] = truth["offset_m"][rows]
            df["rel_start_s"] = truth["rel_start"][rows]
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32") if c not in ("t_s",) else out[c]
    return out
