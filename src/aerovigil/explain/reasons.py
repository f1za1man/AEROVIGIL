r"""Plan M10 (Step E): the vocabulary of reasons, and what "normal" looks like.

Every explanation in the project is phrased in terms of a small set of physical checks,
so that an operator reads the same words whichever detector raised the alarm. Each
check has one measured quantity, in physical units, and a "normal up to" value: the
95th percentile of that quantity on the clean rows (or filter ticks) of the normal
training flights. A reason then reads "<what disagrees>: <value> (normal up to <ref>)".

Row checks, built from Module 03's 36 residual features (each feature belongs to one):

    course              GPS direction of travel vs where the drone points (degrees)
    horizontal_velocity GPS position change vs GPS-reported horizontal speed (m/s)
    vertical_velocity   GPS altitude change vs GPS-reported climb rate (m/s)
    altitude            GPS altitude vs barometer: how fast they drift apart (m/s)
    accel_horizontal    GPS velocity change vs the accelerometer, horizontal (m/s^2)
    accel_vertical      GPS climb-rate change vs the accelerometer, vertical (m/s^2)
    gps_motion          GPS-derived speed itself (m/s)

Filter checks, from Module 06's Kalman filter at the alarm tick:

    kf_position         GPS position vs the filter's prediction (m)
    kf_velocity         GPS velocity vs the filter's IMU-driven prediction (m/s)
    kf_altitude         vertical filter: GPS altitude and climb vs barometer (NIS)

The normals are cached in data/processed/explain/normals.json (rebuild with --force).
"""

import json
import time

import numpy as np
import pandas as pd

from aerovigil.adaptive.search import normal_flights
from aerovigil.stealth import detectors as D

OUT = D.ROOT / "data" / "processed" / "explain"
NORMALS = OUT / "normals.json"
NORMAL_Q = 95

ROW_CHECKS = {
    "course": ("res_course_heading", "res_course_heading_abs", "deg",
               "GPS direction of travel disagrees with where the drone is pointing"),
    "horizontal_velocity": ("res_horizontal_velocity", "res_horizontal_velocity", "m/s",
                            "GPS position change disagrees with the GPS-reported speed"),
    "vertical_velocity": ("res_vertical_velocity", "res_vertical_velocity_abs", "m/s",
                          "GPS altitude change disagrees with the reported climb rate"),
    "altitude": ("res_altitude", "res_altitude_rate", "m/s",
                 "GPS altitude is drifting away from the barometer"),
    "accel_horizontal": ("res_accel_horizontal", "res_accel_horizontal", "m/s^2",
                         "GPS speed changes do not match the accelerometer "
                         "(horizontal)"),
    "accel_vertical": ("res_accel_vertical", "res_accel_vertical_abs", "m/s^2",
                       "GPS climb-rate changes do not match the accelerometer "
                       "(vertical)"),
    "gps_motion": ("gps_", "gps_speed_from_position", "m/s",
                   "GPS-derived speed is unusual"),
}  # fmt: skip
FILTER_CHECKS = {
    "kf_position": ("m", "GPS position jumped away from where the navigation filter "
                    "predicted"),
    "kf_velocity": ("m/s", "GPS velocity disagrees with the filter's IMU-based "
                    "prediction"),
    "kf_altitude": ("NIS", "GPS altitude and climb disagree with the barometer "
                    "(vertical filter)"),
}  # fmt: skip
SHORT = {"course": "heading vs course", "horizontal_velocity": "horizontal speed",
         "vertical_velocity": "climb rate", "altitude": "barometer altitude",
         "accel_horizontal": "horizontal accelerometer",
         "accel_vertical": "vertical accelerometer", "gps_motion": "GPS speed",
         "kf_position": "filter position", "kf_velocity": "filter velocity",
         "kf_altitude": "vertical filter"}  # fmt: skip
CHECKS = tuple(ROW_CHECKS) + tuple(FILTER_CHECKS)


def family_of(feature):
    """The row check a feature (or a window column 'feature__stat') belongs to."""
    base = feature.split("__")[0]
    for name, (prefix, *_rest) in ROW_CHECKS.items():
        if base.startswith(prefix):
            return name
    raise KeyError(f"feature {feature!r} belongs to no check")


def row_measure(name, features):
    """The check's measured quantity on every row of a feature frame."""
    _, col, _, _ = ROW_CHECKS[name]
    v = np.abs(features[col].to_numpy(float))
    if name == "course":  # sin(course - heading), scaled down at low speed
        v = np.degrees(np.arcsin(np.clip(v, 0.0, 1.0)))
    return v


def filter_measures(df):
    """kf_position, kf_velocity, kf_altitude on every filter tick."""
    return {
        "kf_position": np.hypot(df["nu_pos_e"], df["nu_pos_n"]).to_numpy(float),
        "kf_velocity": np.hypot(df["nu_vel_e"], df["nu_vel_n"]).to_numpy(float),
        "kf_altitude": df["nis_v"].to_numpy(float),
    }


def build_normals(force=False):
    """95th percentile of every check on the clean normal training flights."""
    if NORMALS.is_file() and not force:
        return json.loads(NORMALS.read_text())
    started = time.time()
    det = D.Detector()
    rows = {k: [] for k in CHECKS}
    for fid, _, g in normal_flights("train"):
        fl = det.flight(g)
        for k in ROW_CHECKS:
            rows[k].append(row_measure(k, fl.features))
        df = fl.filter
        if df is not None:
            ok = df["valid"].to_numpy(bool)
            for k, v in filter_measures(df).items():
                rows[k].append(v[ok])
    out = {}
    for k, parts in rows.items():
        v = np.concatenate(parts)
        v = v[np.isfinite(v)]
        out[k] = float(np.percentile(v, NORMAL_Q))
    OUT.mkdir(parents=True, exist_ok=True)
    D.atomic_write(NORMALS, lambda p: p.write_text(json.dumps(out, indent=2)))
    print(f"normals from {len(rows['course'])} normal training flights "
          f"[{time.time() - started:.0f} s]: "
          + ", ".join(f"{k} {v:.3g}" for k, v in out.items()))  # fmt: skip
    return out


def unit_of(check):
    return ROW_CHECKS[check][2] if check in ROW_CHECKS else FILTER_CHECKS[check][0]


def text_of(check):
    return ROW_CHECKS[check][3] if check in ROW_CHECKS else FILTER_CHECKS[check][1]


def fmt(value, unit):
    if not np.isfinite(value):
        return "n/a"
    if unit == "deg":
        return f"{value:.0f} deg"
    if unit == "NIS":
        return f"{value:.1f}"
    return f"{value:.2f} {unit}" if value < 10 else f"{value:.0f} {unit}"


def sentence(check, value, normal):
    """'<what disagrees>: <value> (normal up to <ref>)'."""
    u = unit_of(check)
    ratio = value / normal if normal > 0 and np.isfinite(value) else np.nan
    if np.isfinite(ratio) and ratio <= 1.0:
        return (
            f"{text_of(check)}: {fmt(value, u)} (within normal, up to {fmt(normal, u)})"
        )
    times = f", {ratio:.0f}x normal" if np.isfinite(ratio) and ratio >= 2 else ""
    return f"{text_of(check)}: {fmt(value, u)} (normal up to {fmt(normal, u)}{times})"


def message(detector, alarm_t, reasons, consistent, title=None):
    """The operator alert: what fired, when, the top three reasons, what still agrees.

    reasons: list of (check, value, normal) in order of importance.
    """
    head = f"GPS SPOOFING ALARM at {alarm_t:.1f} s ({title or detector})."
    body = " ".join(f"{i}) {sentence(c, v, n)}." for i, (c, v, n) in
                    enumerate(reasons, start=1))  # fmt: skip
    ok = ", ".join(SHORT[c] for c in consistent)
    tail = f" Still consistent: {ok}." if ok else " No sensor check is still normal."
    return f"{head} {body}{tail}"


def as_frame(normals):
    return pd.DataFrame(
        [
            {"check": k, "unit": unit_of(k), "normal_up_to": v, "meaning": text_of(k)}
            for k, v in normals.items()
        ]
    )
