r"""AEROVIGIL Module 03 v2: consistency features on the Raw base table.

Keeps the 22 GPS/barometer features of Module 03 (same code, imported from
build_features.py) and adds 14 IMU features with the convention that Module 06
validated (body_to_world / specific_force / enu):

    res_accel_horizontal : |IMU acceleration (East, North) - acceleration implied by
                           the change of the reported GPS velocity|
    res_accel_vertical   : the same for the vertical axis
    res_course_heading   : sin(GPS course - IMU yaw), faded out below 2 m/s

Each residual gets its value, |value| (signed ones), and trailing 2 s mean, std and
max. The labels from build_raw_base.py (offset_m, spoof_truth, phase, ...) are carried
next to the features and never enter the feature matrix.

Needs build_raw_base.py to have run.

Writes data/processed/module03_v2/features/<flight>.parquet and
       data/processed/module03_v2/feature_metadata.csv

Usage:
    python -m aerovigil.features.build_features_v2
"""

from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.features.build_features import (
    LAG_SECONDS,
    ROLL_SECONDS,
    build_flight,
    feature_metadata,
    trailing,
)
from aerovigil.features.imu_consistency import CONVENTION, imu_features

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "module03_v2"
BASE_DIR = OUT_DIR / "base"
FEAT_DIR = OUT_DIR / "features"
META_FILE = OUT_DIR / "feature_metadata.csv"
REPORT_DIR = ROOT / "data" / "interim" / "module03_v2"
V1_CONFIG = {
    "flight": "flight_id",
    "time": "time_usec",
    "label": "attack_label",
    "lat": "latitude_deg",
    "lon": "longitude_deg",
    "alt": "altitude",
    "vn": "velocity_north",
    "ve": "velocity_east",
    "vu": "velocity_up",
    "baro": "pressure_altitude",
}
CARRIED = [
    "t_s",
    "offset_m",
    "spoof_truth",
    "phase",
    "seconds_since_onset",
    "aligned_all",
]
KEY_FEATURES = [
    "res_horizontal_velocity",
    "res_accel_horizontal",
    "res_accel_vertical",
    "res_course_heading",
    "res_altitude_abs",
    "gps_speed_from_position",
]
SAMPLE_EVERY = 25


def build_flight_v2(g):
    out = build_flight(g, V1_CONFIG, 1e-6)
    t = g["t_s"].to_numpy(float)
    step = np.diff(t)
    gap = step > 10 * np.median(step[step > 0])
    seg = np.concatenate([[0], np.cumsum(gap)])
    j = np.searchsorted(t, t - LAG_SECONDS, side="right") - 1
    ok = j >= 0
    j = np.where(ok, j, 0)
    ok &= seg[j] == seg
    dt = np.where(ok, t - t[j], np.nan)
    idx = pd.to_timedelta(t, unit="s")
    for name, (x, signed) in imu_features(g, t, j, ok, dt).items():
        out[name] = np.asarray(x, dtype="float32")
        if signed:
            out[name + "_abs"] = np.asarray(np.abs(x), dtype="float32")
        for stat in ("mean", "std", "max"):
            out[f"{name}_roll_{stat}"] = np.asarray(
                trailing(x, idx, ROLL_SECONDS, stat), dtype="float32"
            )
    for col in CARRIED:
        out[col] = g[col].to_numpy()
    return out


def metadata(feature_names):
    meta = feature_metadata(
        [
            f
            for f in feature_names
            if not f.startswith("res_acc") and not f.startswith("res_course")
        ]
    )
    meta = meta[meta["available_for_all_flights"] == "Yes"]
    gap_note = "NaN in the first 1 s of a flight, after a GPS gap, and in IMU dropouts"
    rows = []
    base = [
        (
            "res_accel_horizontal",
            "|| mean IMU accel (E,N) over L - d(smoothed GPS velocity)/dt ||",
            "orientation_*, linear_acceleration_*, velocity_east/north",
            "m/s^2",
        ),
        (
            "res_accel_vertical",
            "mean IMU accel (U) over L - d(smoothed GPS velocity_up)/dt",
            "orientation_*, linear_acceleration_*, velocity_up",
            "m/s^2",
        ),
        (
            "res_course_heading",
            "sin(atan2(vn,ve) - IMU yaw) * min(speed/2, 1)",
            "orientation_*, velocity_east/north",
            "unitless",
        ),
    ]
    for name, formula, sources, units in base:
        extra = ["", "_roll_mean", "_roll_std", "_roll_max"]
        if name != "res_accel_horizontal":
            extra.insert(1, "_abs")
        for suffix in extra:
            rows.append(
                {
                    "feature": name + suffix,
                    "formula": (
                        formula if not suffix else f"{suffix.strip('_')} of {name}"
                    ),
                    "source_columns": sources,
                    "units": units,
                    "assumptions": "Module 06 IMU convention body_to_world/"
                    "specific_force/enu; L = 1 s; causal windows",
                    "missing_value_behaviour": gap_note,
                    "available_for_all_flights": "Yes",
                }
            )
    return pd.concat([meta, pd.DataFrame(rows)], ignore_index=True)


def main():
    files = sorted(BASE_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No base files found. Run build_raw_base.py first.")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("AEROVIGIL - FEATURES v2 (GPS/barometer + IMU consistency)")
    print("=" * 65)
    print(f"Base flights: {len(files)} | convention {CONVENTION}\n")
    rows_total, feature_names, nan_counts, samples = 0, None, None, []
    skip = {"flight_id", "mission_type", "time_usec", "attack_label", "after_gap"}
    for n, path in enumerate(files, start=1):
        g = pd.read_parquet(path)
        out = build_flight_v2(g)
        out.to_parquet(FEAT_DIR / path.name, index=False)
        if feature_names is None:
            feature_names = [c for c in out.columns if c not in skip | set(CARRIED)]
            nan_counts = pd.Series(0, index=feature_names, dtype="int64")
        nan_counts += out[feature_names].isna().sum()
        rows_total += len(out)
        mark = out["flight_id"].iloc[0].split("__")[1]
        sub = out.iloc[::SAMPLE_EVERY][["phase", *KEY_FEATURES]].copy()
        sub["flight_class"] = mark
        samples.append(sub)
        if n % 40 == 0 or n == len(files):
            print(f"Processed {n}/{len(files)} flights")
    metadata(feature_names).to_csv(META_FILE, index=False)

    pd.set_option("display.width", 200)
    sample = pd.concat(samples, ignore_index=True)
    names = {0: "normal flight", 1: "pre", 2: "ramp", 3: "plateau", 4: "post"}
    sample["phase_name"] = sample["phase"].map(names)
    print("\n" + "-" * 65)
    print(f"Feature columns: {len(feature_names)}   rows: {rows_total:,}")
    print("\nMISSING (NaN) % PER FEATURE")
    print((nan_counts / rows_total * 100).round(2).to_string())
    print("\nKEY FEATURES BY PHASE (median / 95th percentile, every 25th row)")
    order = ["normal flight", "pre", "ramp", "plateau", "post"]
    for col in KEY_FEATURES:
        table = sample.groupby("phase_name")[col].agg(
            median="median", p95=lambda x: x.quantile(0.95)
        )
        print(f"\n{col}")
        print(table.reindex(order).round(4).to_string())
    forbidden = [
        f
        for f in feature_names
        if any(k in f.lower() for k in ("attack", "label", "flight", "phase", "spoof"))
    ]
    checks = {
        "one feature file per base flight": len(list(FEAT_DIR.glob("*.parquet")))
        == len(files),
        "row count equals the base table": rows_total
        == sum(len(pd.read_parquet(p, columns=["t_s"])) for p in files),
        "no label/ID/phase column used as a feature": not forbidden,
        "feature metadata written": META_FILE.exists(),
        "metadata covers every feature": set(feature_names)
        <= set(pd.read_csv(META_FILE)["feature"]),
    }
    print("\nACCEPTANCE CHECKS")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\nSaved:", FEAT_DIR)
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Feature checks FAILED")


if __name__ == "__main__":
    main()
