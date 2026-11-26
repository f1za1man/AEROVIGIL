r"""AEROVIGIL diagnostic: is the attitude quaternion being read in the right order?

Module 09's tilt diagnostic returned a residual of about +5 m/s at every drift rate and
in every direction, which is impossible if it were measuring the drift. The cause is
that the apparent tilt exceeds 90 degrees on most rows, so tan(tilt) goes negative, the
clamp drives the implied speed to zero and the residual becomes the raw flight speed.

A multirotor does not fly inverted. Either the quaternion columns are not in the order
(x, y, z, w) that the project assumes, or the frame convention is different. Module 06's
calibrate_imu.py searched eight conventions (rotation, gravity, frame) but never the
component order, so a wrong order would have been absorbed by whichever of the eight
least-badly compensated for it, and the filter was then tuned around the error.

That matters well beyond the tilt idea. If the order is wrong then every attitude-derived
feature is wrong: res_accel_horizontal, res_accel_vertical and res_course_heading, which
are 14 of the 36 v2 features, plus the IMU input to the Kalman filter. It would explain
the course channel's clean spread of 0.19 on nose-forward flights, the acceleration
channel never responding to any attack, and the noise tuner driving the velocity scale
to its 1e-6 floor, which is the tuner saying the IMU is worthless.

This script decides it. For each candidate component order it reports the tilt
distribution, and for each order crossed with each of Module 06's conventions it rotates
the measured accelerometer into the world frame and compares it with the acceleration
implied by GPS velocity on clean, steady rows. The right combination gives small errors
and physically sensible tilts. Nothing is written and nothing is changed.

Usage:
    python src\aerovigil\consistency\diagnose_quaternion.py
    python src\aerovigil\consistency\diagnose_quaternion.py --flights 6
"""

import argparse

import numpy as np
import pandas as pd

from aerovigil.consistency.diagnose_tilt import MAX_ACCEL, MIN_SPEED, load, smooth
from aerovigil.kalman.imu import G, all_conventions, conv_name, world_accel

# name: which of the four stored columns holds (x, y, z, w)
ORDERS = {
    "xyzw (assumed)": (0, 1, 2, 3),
    "wxyz": (1, 2, 3, 0),
}
COLUMNS = ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]
PLAUSIBLE_TILT_DEG = 35.0


def quaternion(g, order):
    q = g[COLUMNS].to_numpy(float)[:, list(order)]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def tilt_deg(q):
    """Angle between the body z axis and world up, from a normalised (x,y,z,w) array."""
    cos_t = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def steady_rows(g):
    """Clean, level-ish cruise: fast enough to matter, not accelerating hard."""
    speed = np.hypot(
        g["velocity_north"].to_numpy(float), g["velocity_east"].to_numpy(float)
    )
    t = g["t_s"].to_numpy(float)
    accel = np.gradient(smooth(speed), t)
    return (speed > MIN_SPEED) & (np.abs(accel) < MAX_ACCEL) & np.isfinite(speed)


def gps_accel(g):
    """Acceleration implied by the reported velocity, in ENU, smoothed."""
    t = g["t_s"].to_numpy(float)
    out = []
    for col in ("velocity_east", "velocity_north", "velocity_up"):
        out.append(np.gradient(smooth(g[col].to_numpy(float)), t))
    return np.column_stack(out)


def tilt_table(flights):
    rows = []
    for name, order in ORDERS.items():
        pool = np.concatenate([tilt_deg(quaternion(g, order)) for g in flights])
        rows.append({
            "order": name, "median_tilt_deg": float(np.median(pool)),
            "p95_tilt_deg": float(np.percentile(pool, 95)),
            "share_over_90_deg": float((pool > 90).mean()),
            "plausible": bool(np.median(pool) < PLAUSIBLE_TILT_DEG),
        })  # fmt: skip
    return pd.DataFrame(rows)


def accel_table(flights):
    """Median horizontal error between IMU-derived and GPS-derived acceleration."""
    rows = []
    for name, order in ORDERS.items():
        for conv in all_conventions():
            errs, verts = [], []
            for g in flights:
                ok = steady_rows(g)
                if not ok.any():
                    continue
                q = quaternion(g, order)[ok]
                acc = g[
                    [
                        "linear_acceleration_x",
                        "linear_acceleration_y",
                        "linear_acceleration_z",
                    ]
                ].to_numpy(float)[ok]
                world = world_accel(acc, q, conv)
                ref = gps_accel(g)[ok]
                errs.append(np.hypot(world[:, 0] - ref[:, 0], world[:, 1] - ref[:, 1]))
                verts.append(world[:, 2] - ref[:, 2])
            if not errs:
                continue
            e = np.concatenate(errs)
            v = np.concatenate(verts)
            rows.append({
                "order": name, "convention": conv_name(conv),
                "median_horiz_error_mps2": float(np.nanmedian(e)),
                "median_vert_bias_mps2": float(np.nanmedian(v)),
            })  # fmt: skip
    out = pd.DataFrame(rows)
    return out.sort_values("median_horiz_error_mps2").reset_index(drop=True)


def run(limit=6):
    flights = load(limit)
    print("=" * 78)
    print("AEROVIGIL DIAGNOSTIC: quaternion component order and IMU convention")
    print("=" * 78)
    print(f"Clean test flights {len(flights)}   gravity constant {G:g} m/s^2\n")
    tilts = tilt_table(flights)
    print("TILT DISTRIBUTION per component order")
    print("  a multirotor cruises at roughly 5 to 25 degrees and is never inverted")
    print(tilts.round(3).to_string(index=False))
    good = tilts[tilts["plausible"]]["order"].tolist()
    print(
        "\n  plausible order(s): "
        + (", ".join(good) if good else "NONE - neither order gives sensible tilts")
    )
    acc = accel_table(flights)
    print("\nIMU AGAINST GPS ACCELERATION on steady rows, best first")
    print("  the right combination makes the horizontal error small")
    print(acc.head(8).round(3).to_string(index=False))
    best = acc.iloc[0]
    print(
        f"\nBest: order {best['order']}, convention {best['convention']}, "
        f"horizontal error {best['median_horiz_error_mps2']:.3f} m/s^2"
    )
    assumed = acc[acc["order"] == "xyzw (assumed)"].iloc[0]
    if best["order"] != "xyzw (assumed)":
        ratio = assumed["median_horiz_error_mps2"] / max(
            best["median_horiz_error_mps2"], 1e-9
        )
        print(
            f"The order this project assumes is {ratio:.1f} times worse. Every "
            "attitude-derived\nfeature and the filter's IMU input are affected."
        )
    else:
        print("The order this project assumes is already the best one.")
    return tilts, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=6)
    a = ap.parse_args()
    run(a.flights)


if __name__ == "__main__":
    main()
