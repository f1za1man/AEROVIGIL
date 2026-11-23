from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "preprocessing" / "preprocessing.yaml"
OUT_DIR = ROOT / "data" / "processed" / "module03"
CLEAN_DIR = OUT_DIR / "cleaned"
FEAT_DIR = OUT_DIR / "features"
META_FILE = OUT_DIR / "feature_metadata.csv"

LAG_SECONDS = 1.0  # horizon for position-derived velocity and rates
ROLL_SECONDS = 2.0  # trailing window for rolling statistics
GAP_FACTOR = 10  # same large-gap rule as the timestamp report
EARTH_R = 6371000.0  # metres


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def trailing(values, index, seconds, stat):
    r = pd.Series(values, index=index).rolling(
        f"{int(seconds * 1000)}ms", min_periods=1
    )
    return getattr(r, stat)().to_numpy()


def build_flight(g, c, scale):
    t_us = g[c["time"]].to_numpy(dtype="int64")
    t = (t_us - t_us[0]) * scale  # seconds since flight start

    # ---- gap segmentation (gaps are masked, never repaired) -----------
    step = np.diff(t)
    median_step = np.median(step[step > 0])
    gap_flag = step > GAP_FACTOR * median_step
    seg = np.concatenate([[0], np.cumsum(gap_flag)])
    after_gap = np.concatenate([[False], gap_flag])

    # ---- lagged sample: latest sample at least LAG_SECONDS earlier ----
    idx = np.searchsorted(t, t - LAG_SECONDS, side="right") - 1
    ok = idx >= 0
    j = np.where(ok, idx, 0)
    ok &= seg[j] == seg  # lag must not cross a gap
    dt = np.where(ok, t - t[j], np.nan)

    lat = g[c["lat"]].to_numpy(float)
    lon = g[c["lon"]].to_numpy(float)
    alt = g[c["alt"]].to_numpy(float)
    baro = g[c["baro"]].to_numpy(float)
    vn = g[c["vn"]].to_numpy(float)
    ve = g[c["ve"]].to_numpy(float)
    vu = g[c["vu"]].to_numpy(float)

    # ---- 9.1 GPS movement (haversine, longitude wrapped) ---------------
    phi1, phi2 = np.radians(lat[j]), np.radians(lat)
    dphi = phi2 - phi1
    dlmb = wrap_pi(np.radians(lon - lon[j]))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    dist = 2 * EARTH_R * np.arcsin(np.sqrt(a))
    speed_pos = dist / dt
    vn_pos = dphi * EARTH_R / dt
    ve_pos = dlmb * EARTH_R * np.cos((phi1 + phi2) / 2) / dt
    vu_pos = (alt - alt[j]) / dt

    # ---- reported velocity averaged over the same trailing window ------
    idx_td = pd.to_timedelta(t, unit="s")
    vn_ref = trailing(vn, idx_td, LAG_SECONDS, "mean")
    ve_ref = trailing(ve, idx_td, LAG_SECONDS, "mean")
    vu_ref = trailing(vu, idx_td, LAG_SECONDS, "mean")

    # ---- residuals ------------------------------------------------------
    residuals = {
        # name: (values, signed)
        "res_altitude": (alt - baro, True),
        "res_horizontal_velocity": (np.hypot(vn_pos - vn_ref, ve_pos - ve_ref), False),
        "res_vertical_velocity": (vu_pos - vu_ref, True),
    }

    feats = {
        "gps_speed_from_position": speed_pos,
        "gps_vertical_rate_from_altitude": vu_pos,
    }
    for name, (x, signed) in residuals.items():
        feats[name] = x
        if signed:
            feats[name + "_abs"] = np.abs(x)
        feats[name + "_rate"] = (x - x[j]) / dt
        feats[name + "_roll_mean"] = trailing(x, idx_td, ROLL_SECONDS, "mean")
        feats[name + "_roll_std"] = trailing(x, idx_td, ROLL_SECONDS, "std")
        feats[name + "_roll_max"] = trailing(x, idx_td, ROLL_SECONDS, "max")
        feats[name + "_roll_min"] = trailing(x, idx_td, ROLL_SECONDS, "min")

    out = pd.DataFrame(
        {
            "flight_id": g[c["flight"]].to_numpy(),
            "mission_type": g["mission_type"].to_numpy(),
            "time_usec": t_us,
            "attack_label": g[c["label"]].to_numpy(),
            "after_gap": after_gap,
        }
    )
    for k, v in feats.items():
        out[k] = np.asarray(v, dtype="float32")
    return out


def feature_metadata(feature_names):
    rows = []

    def add(name, formula, sources, units, assumptions, missing, available="Yes"):
        rows.append(
            {
                "feature": name,
                "formula": formula,
                "source_columns": sources,
                "units": units,
                "assumptions": assumptions,
                "missing_value_behaviour": missing,
                "available_for_all_flights": available,
            }
        )

    gap_note = (
        f"NaN in the first {LAG_SECONDS:g} s of a flight and for "
        f"{LAG_SECONDS:g} s after each gap"
    )
    add(
        "gps_speed_from_position",
        "haversine(GPS_t, GPS_{t-L}) / dt",
        "latitude_deg, longitude_deg, time_usec",
        "m/s",
        "Time sorted within flight; L = 1 s; lag does not cross a gap",
        gap_note,
    )
    add(
        "gps_vertical_rate_from_altitude",
        "(altitude_t - altitude_{t-L}) / dt",
        "altitude, time_usec",
        "m/s",
        "GPS altitude in metres",
        gap_note,
    )
    add(
        "res_altitude",
        "altitude - pressure_altitude",
        "altitude, pressure_altitude",
        "m",
        "Both in metres with compatible reference (checked in summary)",
        "None",
    )
    add(
        "res_horizontal_velocity",
        "|| [dN/dt, dE/dt] - mean(velocity_north, velocity_east) over L ||",
        "latitude_deg, longitude_deg, velocity_north, velocity_east",
        "m/s",
        "Reported velocity in m/s, north/east frame, same GPS solution",
        gap_note,
    )
    add(
        "res_vertical_velocity",
        "(dAlt/dt) - mean(velocity_up) over L",
        "altitude, velocity_up",
        "m/s",
        "velocity_up in m/s, positive upward",
        gap_note,
    )
    for base in ("res_altitude", "res_horizontal_velocity", "res_vertical_velocity"):
        if base + "_abs" in feature_names:
            add(
                base + "_abs",
                f"|{base}|",
                base,
                "same as base",
                "None beyond base",
                "Same as base",
            )
        add(
            base + "_rate",
            f"({base}_t - {base}_{{t-L}}) / dt",
            base,
            "base unit / s",
            "Lag does not cross a gap",
            gap_note,
        )
        for stat in ("mean", "std", "max", "min"):
            add(
                f"{base}_roll_{stat}",
                f"trailing {stat} of {base} over {ROLL_SECONDS:g} s",
                base,
                "same as base",
                "Trailing (causal) window; no future samples used",
                "Ignores NaN; std needs >= 2 valid samples",
            )
    add(
        "GPS/IMU integrated velocity residual",
        "not generated",
        "linear_acceleration_*, orientation_*",
        "m/s",
        "Needs documented attitude/gravity conventions and drift handling",
        "Not generated",
        "No - unavailable",
    )
    add(
        "GPS course vs magnetometer heading residual",
        "not generated",
        "velocity_north/east, magnetic_field_*",
        "deg",
        "Needs documented magnetometer axes, units and tilt compensation",
        "Not generated",
        "No - unavailable",
    )
    return pd.DataFrame(rows)


def main():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gps = cfg["sensors"]["gps"]
    if not gps.get("velocity_up"):
        raise SystemExit(
            'Add  velocity_up: "velocity_up"  under sensors.gps '
            "in preprocessing.yaml first."
        )

    c = {
        "flight": cfg["dataset"]["flight_id_column"],
        "time": cfg["dataset"]["timestamp_column"],
        "label": cfg["labels"]["attack_column"],
        "lat": gps["latitude"],
        "lon": gps["longitude"],
        "alt": gps["altitude"],
        "vn": gps["velocity_north"],
        "ve": gps["velocity_east"],
        "vu": gps["velocity_up"],
        "baro": cfg["sensors"]["barometer"]["altitude"],
    }
    scale = {
        "seconds": 1.0,
        "milliseconds": 1e-3,
        "microseconds": 1e-6,
        "nanoseconds": 1e-9,
    }[cfg["dataset"]["timestamp_unit"]]

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CLEAN_DIR.glob("*.parquet"))

    print("=" * 65)
    print("AEROVIGIL - SENSOR CONSISTENCY FEATURES")
    print("=" * 65)
    print(f"Cleaned flights: {len(files)}")
    print(f"Lag {LAG_SECONDS:g} s | rolling window {ROLL_SECONDS:g} s\n")

    rows_total = attack_total = 0
    nan_counts, key_parts, feature_names = None, [], None
    key_cols = [
        "res_altitude_abs",
        "res_horizontal_velocity",
        "res_vertical_velocity_abs",
        "gps_speed_from_position",
    ]

    for n, path in enumerate(files, start=1):
        g = pd.read_parquet(path)
        out = build_flight(g, c, scale)
        out.to_parquet(FEAT_DIR / path.name, index=False)

        if feature_names is None:
            feature_names = [
                x
                for x in out.columns
                if x
                not in (
                    "flight_id",
                    "mission_type",
                    "time_usec",
                    "attack_label",
                    "after_gap",
                )
            ]
            nan_counts = pd.Series(0, index=feature_names, dtype="int64")
        nan_counts += out[feature_names].isna().sum()
        rows_total += len(out)
        attack_total += int((out["attack_label"] == 1).sum())
        key_parts.append(out[["attack_label"] + key_cols])

        if n % 40 == 0 or n == len(files):
            print(f"Processed {n}/{len(files)} flights")

    meta = feature_metadata(feature_names)
    meta.to_csv(META_FILE, index=False)

    # ---- summary ---------------------------------------------------------
    key = pd.concat(key_parts, ignore_index=True)
    print("\n" + "-" * 65)
    print(f"Feature columns generated: {len(feature_names)}")
    print("\nMISSING (NaN) % PER FEATURE")
    print((nan_counts / rows_total * 100).round(2).to_string())

    print("\nVALIDITY CHECK - key features by label " "(median / 95th percentile)")
    for col in key_cols:
        s = key.groupby("attack_label")[col].agg(
            median="median", p95=lambda x: x.quantile(0.95)
        )
        print(f"\n{col}")
        print(s.round(4).to_string())

    forbidden = [
        f
        for f in feature_names
        if "attack" in f.lower() or "label" in f.lower() or "flight" in f.lower()
    ]
    checks = {
        "One feature file per cleaned flight": len(list(FEAT_DIR.glob("*.parquet")))
        == len(files),
        "Row count preserved (7,893,049)": rows_total == 7_893_049,
        "Attack rows preserved (4,302,145)": attack_total == 4_302_145,
        "No label/ID column used as a feature": len(forbidden) == 0,
        "Feature metadata written": META_FILE.exists(),
    }
    print("\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nMetadata: {META_FILE}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Feature checks FAILED")


if __name__ == "__main__":
    main()
