r"""AEROVIGIL Module 09 diagnostic: does tilt angle predict airspeed in this dataset?

Module 09 established that a velocity-consistent drift is invisible to position,
velocity and accelerometer checks, and that the course-against-yaw check fails on real
flights because a multirotor's nose does not point along its track (clean MAD 0.19 here
against 0.002 on nose-forward synthetic flights).

One physical relation is left that a spoofer does not control. A multirotor accelerates
by tilting, and in steady flight the tilt is what balances aerodynamic drag:

    T cos(theta) = m g           (vertical)
    T sin(theta) = drag ~ v^2    (horizontal)
    =>  tan(theta) = v^2 / k     for some airframe constant k

So tilt implies a speed, sqrt(k tan(theta)), that the attacker cannot alter: it owns the
GPS message, not the airframe. If the spoofed speed departs from the tilt-implied speed
by much more than the clean scatter, this is a detector. If it does not, there is no
detector in this dataset's sensors and Module 09's honest result is the velocity channel
plus a stated impossibility.

This script only measures. It fits k on clean flights, reports the scatter, and then
reports how far a spoofed flight moves away from the relation, in units of that scatter.
A separation ratio above about 5 means a usable detector; below 2 means none.

Usage:
    python src\aerovigil\consistency\diagnose_tilt.py
    python src\aerovigil\consistency\diagnose_tilt.py --flights 12
"""

import argparse

import numpy as np
import pandas as pd

from aerovigil.adaptive.profile import DEFAULT, profile_fn
from aerovigil.attacks.inject_v2 import inject
from aerovigil.lstm.build_sequences_v2 import BASE_REPORT, ROOT, SPLITS

M03_BASE = ROOT / "data" / "processed" / "module03_v2" / "base"
MIN_SPEED = 2.0
MIN_TILT_DEG = 2.0
MAX_ACCEL = 0.5
SMOOTH_S = 2.0
HZ = 125.0
RATES = (0.25, 1.0, 5.0)
START_FRACTION = 0.25


def tilt_rad(g):
    """Angle between the body z axis and world up, from the attitude quaternion."""
    x = g["orientation_x"].to_numpy(float)
    y = g["orientation_y"].to_numpy(float)
    z = g["orientation_z"].to_numpy(float)
    w = g["orientation_w"].to_numpy(float)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n = np.where(n > 0, n, 1.0)
    cos_t = 1.0 - 2.0 * ((x / n) ** 2 + (y / n) ** 2)
    return np.arccos(np.clip(cos_t, -1.0, 1.0))


def smooth(v, seconds=SMOOTH_S, hz=HZ):
    win = max(int(seconds * hz), 1)
    return pd.Series(v).rolling(win, min_periods=1, center=True).mean().to_numpy()


def steady(g):
    """Rows in steady horizontal flight, where tan(tilt) reflects drag and not haste."""
    speed = np.hypot(
        g["velocity_north"].to_numpy(float), g["velocity_east"].to_numpy(float)
    )
    t = g["t_s"].to_numpy(float)
    theta = tilt_rad(g)
    accel = np.gradient(smooth(speed), t)
    ok = (
        (speed > MIN_SPEED)
        & (theta > np.radians(MIN_TILT_DEG))
        & (np.abs(accel) < MAX_ACCEL)
        & np.isfinite(speed)
        & np.isfinite(theta)
    )
    return smooth(speed), np.tan(theta), ok


def fit_k(flights):
    """Airframe constant from clean flights: the median of v^2 / tan(theta)."""
    ratios = []
    for g in flights:
        speed, tan_t, ok = steady(g)
        if ok.any():
            ratios.append((speed[ok] ** 2) / tan_t[ok])
    pool = np.concatenate(ratios)
    return float(np.median(pool)), len(pool)


def residuals(g, k):
    """Spoofed speed minus the speed the tilt implies, over the steady rows."""
    speed, tan_t, ok = steady(g)
    implied = np.sqrt(k * np.maximum(tan_t, 1e-12))
    return (speed - implied)[ok], ok


def mean_course_deg(g):
    """The flight's average direction of travel, in the profile's bearing convention."""
    vn = float(np.nanmean(g["velocity_north"].to_numpy(float)))
    ve = float(np.nanmean(g["velocity_east"].to_numpy(float)))
    return float(np.degrees(np.arctan2(ve, vn)))


def attacked(g, rate, onset=20.0, bearing=135.0):
    duration = float(g["t_s"].iloc[-1])
    start = duration * START_FRACTION
    spec = {"attack_type": "adaptive", "start_s": start,
            "duration_s": duration - start, "params": {}}  # fmt: skip
    params = {**DEFAULT, "rate_mps": rate, "onset_s": onset, "duty": 1.0,
              "jitter_m": 0.0, "bearing_deg": bearing}  # fmt: skip
    out, _ = inject(g, spec, "consistent", "probe", profile_fn(params))
    return out


def load(limit):
    rep = pd.read_csv(BASE_REPORT).sort_values("flight_id")
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    rep = rep[(rep["attacked"] == 0) & (rep["flight_id"].map(splits) == "test")]
    per = -(-limit // max(rep["mission"].nunique(), 1))
    rep = rep.groupby("mission", group_keys=False).head(per).head(limit)
    out = []
    for _, r in rep.iterrows():
        path = M03_BASE / r["file"]
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Build the Module 03 v2 base first.")
        out.append(pd.read_parquet(path))
    if not out:
        raise SystemExit("No clean test flights found.")
    return out


def run(limit=12):
    flights = load(limit)
    print("=" * 74)
    print("AEROVIGIL - MODULE 09 DIAGNOSTIC: does tilt predict airspeed?")
    print("=" * 74)
    print(f"Clean test flights {len(flights)}   model  v = sqrt(k tan(tilt))")
    k, n_rows = fit_k(flights)
    print(f"Fitted k = {k:.3f} from {n_rows:,} steady rows "
          f"(speed > {MIN_SPEED:g} m/s, tilt > {MIN_TILT_DEG:g} deg, "
          f"|dv/dt| < {MAX_ACCEL:g} m/s^2)")  # fmt: skip
    clean = np.concatenate([residuals(g, k)[0] for g in flights])
    spread = float(1.4826 * np.median(np.abs(clean - np.median(clean))))
    print(f"\nCLEAN FLIGHTS: residual median {np.median(clean):+.3f} m/s, "
          f"robust spread {spread:.3f} m/s")  # fmt: skip
    print(f"  the relation explains clean flight to within about {spread:.2f} m/s\n")
    rows = []
    for rate in RATES:
        for aim in ("across", "along"):
            shifts = []
            for g in flights:
                course = mean_course_deg(g)
                bearing = course if aim == "along" else course + 90.0
                r, _ = residuals(attacked(g, rate, bearing=bearing), k)
                if len(r):
                    shifts.append(float(np.median(r)))
            shift = float(np.median(shifts)) if shifts else np.nan
            rows.append({
                "drift_rate_mps": rate, "aim": aim,
                "median_residual_shift_mps": shift,
                "separation_ratio": abs(shift) / spread if spread > 0 else np.nan,
            })  # fmt: skip
    table = pd.DataFrame(rows)
    print("SPOOFED FLIGHTS: how far the attack moves the residual")
    print("  along  = drift aimed along the flight path: changes the speed by the")
    print("           full drift rate, so the tilt relation should see it")
    print("  across = drift aimed sideways: barely changes the speed, so it should not")
    print(table.round(3).to_string(index=False))

    def verdict(v):
        return "usable" if v >= 5 else "borderline" if v >= 2 else "nothing"

    print("")
    for aim in ("along", "across"):
        sub = table[table["aim"] == aim]
        worst = float(sub["separation_ratio"].min())
        best = float(sub["separation_ratio"].max())
        print(
            f"  {aim:6s} separation {worst:.1f} to {best:.1f} x the clean spread"
            f"  -> {verdict(best)} at the fastest drift, {verdict(worst)} at the slowest"
        )
    slow = table[table["drift_rate_mps"] == min(RATES)]["separation_ratio"].max()
    print(
        f"\nAt {min(RATES):g} m/s the best any aim achieves is {slow:.1f} x the spread. "
        "A detector\nneeds that to be large, because a slow drift is the attack that "
        "matters."
    )
    print("\nFor reference, the channels already measured on this data:")
    print("  res_horizontal_velocity  catches 97% of the dataset's own attacks")
    print(
        "  res_course_heading       clean spread 0.19, attack peak 5.2, threshold 15.4"
    )
    print("  res_accel_horizontal     unchanged by a velocity-consistent drift")
    return table, k, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=12)
    a = ap.parse_args()
    run(a.flights)


if __name__ == "__main__":
    main()
