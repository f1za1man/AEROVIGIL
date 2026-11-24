r"""AEROVIGIL Module 03 v2: build the base flight table from the Raw files.

Why: Merged keeps only the timestamps shared by every sensor file (about 70 % of the
rows) and cuts the end of each flight. Here every stream is aligned to the gps.csv
timeline by nearest time (tolerance 10 ms), so no gps row is lost.

Per flight (the same 360 flights and flight ids as Merged) it writes one parquet file
with the sensor columns, the unspoofed raw_gps position, the groundtruth, and labels
measured from the data:
    attack_enabled / attack_label : the original label (kept unchanged)
    offset_m                      : horizontal distance between gps and raw_gps
    spoof_truth                   : 1 from the first labelled row to the last row with
                                    offset_m >= 1 m (the real spoofed interval)
    phase                         : 0 normal flight, 1 pre-attack, 2 ramp,
                                    3 plateau, 4 post-attack
    seconds_since_onset           : seconds from the first labelled row (NaN if none)

Needs inspect_raw_files.py and inspect_dataset.py to have run.

Writes data/processed/module03_v2/base/<flight>.parquet and
       data/interim/module03_v2/base_report.csv

Usage:
    python src\aerovigil\data\build_raw_base.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INTERIM = PROJECT_ROOT / "data" / "interim"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "module03_v2" / "base"
REPORT_DIR = INTERIM / "module03_v2"
METRES_PER_DEGREE = 111320.0
TOLERANCE_US = 10_000
SPOOF_MIN_OFFSET_M = 1.0
RAMP_END_FRACTION = 0.9
PHASE_NORMAL, PHASE_PRE, PHASE_RAMP, PHASE_PLATEAU, PHASE_POST = 0, 1, 2, 3, 4

GPS_COLS = [
    "latitude_deg",
    "longitude_deg",
    "altitude",
    "eph",
    "epv",
    "velocity",
    "velocity_east",
    "velocity_north",
    "velocity_up",
    "attack_enabled",
]
RAW_GPS_COLS = [c for c in GPS_COLS if c not in ("eph", "epv", "attack_enabled")]
RAW_GPS_RENAME = {
    "latitude_deg": "raw_latitude_deg",
    "longitude_deg": "raw_longitude_deg",
    "altitude": "raw_altitude",
    "velocity": "raw_velocity",
    "velocity_east": "raw_velocity_east",
    "velocity_north": "raw_velocity_north",
    "velocity_up": "raw_velocity_up",
}
TRUTH_RENAME = {
    "altitude": "gt_altitude",
    "velocity_east": "gt_velocity_east",
    "velocity_north": "gt_velocity_north",
    "velocity_up": "gt_velocity_up",
    "attitude_q_w": "gt_q_w",
    "attitude_q_x": "gt_q_x",
    "attitude_q_y": "gt_q_y",
    "attitude_q_z": "gt_q_z",
}
BARO_COLS = ["temperature", "absolute_pressure", "pressure_altitude"]
IMU_COLS = [
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
]
MAG_COLS = ["magnetic_field_x", "magnetic_field_y", "magnetic_field_z"]
TRUTH_COLS = [
    "latitude_rad",
    "longitude_rad",
    "altitude",
    "velocity_east",
    "velocity_north",
    "velocity_up",
    "attitude_q_w",
    "attitude_q_x",
    "attitude_q_y",
    "attitude_q_z",
]
FLOAT64 = {
    "latitude_deg",
    "longitude_deg",
    "raw_latitude_deg",
    "raw_longitude_deg",
    "gt_latitude_deg",
    "gt_longitude_deg",
    "offset_m",
    "t_s",
    "seconds_since_onset",
}


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def load_stream(folder, name, columns):
    """Read one Raw file, sort by time and drop repeated timestamps."""
    df = pd.read_csv(folder / f"{name}.csv", usecols=["time_usec", *columns])
    df = df.sort_values("time_usec", kind="stable")
    duplicates = int(df["time_usec"].duplicated().sum())
    return (
        df.drop_duplicates("time_usec", keep="first").reset_index(drop=True),
        duplicates,
    )


def attach(base, other, rename=None):
    """Nearest-time join of one stream onto the gps timeline; returns table, misses."""
    other = other.rename(columns=rename or {})
    value_cols = [c for c in other.columns if c != "time_usec"]
    key = "_stream_time"
    other = other.assign(**{key: other["time_usec"]})
    joined = pd.merge_asof(
        base[["time_usec"]],
        other,
        on="time_usec",
        direction="nearest",
        tolerance=TOLERANCE_US,
    )
    misses = int(joined[key].isna().sum())
    return joined[value_cols], misses


def horizontal_offset_m(lat, lon, raw_lat, raw_lon):
    north = (lat - raw_lat) * METRES_PER_DEGREE
    east = (lon - raw_lon) * METRES_PER_DEGREE * np.cos(np.radians(raw_lat))
    return np.hypot(north, east)


def label_phases(offset, attack_label, t_s, attacked_flight):
    """Return phase, spoof_truth, seconds_since_onset and a dict of timing facts."""
    n = len(offset)
    phase = np.full(n, PHASE_NORMAL, dtype=np.int8)
    truth = np.zeros(n, dtype=np.int8)
    since = np.full(n, np.nan)
    facts = dict.fromkeys(
        ["onset_s", "ramp_end_s", "spoof_end_s", "label_end_s", "plateau_m"], np.nan
    )
    labelled = np.flatnonzero(attack_label == 1)
    if not attacked_flight or len(labelled) == 0:
        return phase, truth, since, facts
    onset = int(labelled[0])
    over = np.flatnonzero(np.nan_to_num(offset) >= SPOOF_MIN_OFFSET_M)
    end = int(over[-1]) if len(over) else onset
    end = max(end, onset)
    inside = np.nan_to_num(offset[onset : end + 1])
    plateau = float(np.median(inside[len(inside) // 2 :])) if len(inside) else 0.0
    reached = np.flatnonzero(inside >= RAMP_END_FRACTION * plateau)
    ramp_end = onset + (int(reached[0]) if len(reached) else 0)
    phase[:onset] = PHASE_PRE
    phase[onset:ramp_end] = PHASE_RAMP
    phase[ramp_end : end + 1] = PHASE_PLATEAU
    phase[end + 1 :] = PHASE_POST
    truth[onset : end + 1] = 1
    since = t_s - t_s[onset]
    facts.update(
        onset_s=float(t_s[onset]),
        ramp_end_s=float(t_s[ramp_end]),
        spoof_end_s=float(t_s[end]),
        label_end_s=float(t_s[labelled[-1]]),
        plateau_m=plateau,
    )
    return phase, truth, since, facts


def build_flight(folder, flight_id, mission, attacked_flight):
    gps, dup_gps = load_stream(folder, "gps", GPS_COLS + ["time_utc_usec"])
    base = gps.drop(columns=["time_utc_usec"]).copy()
    misses, dups = {}, {"gps": dup_gps}
    parts = []
    streams = [
        ("raw_gps", RAW_GPS_COLS, RAW_GPS_RENAME),
        ("barometer", BARO_COLS, None),
        ("imu", IMU_COLS, None),
        ("magnetometer", MAG_COLS, None),
        ("groundtruth", TRUTH_COLS, TRUTH_RENAME),
    ]
    for name, cols, rename in streams:
        table, dup = load_stream(folder, name, cols)
        joined, miss = attach(base, table, rename)
        parts.append(joined)
        misses[name], dups[name] = miss, dup
    df = pd.concat([base, *parts], axis=1)
    df["gt_latitude_deg"] = np.degrees(df.pop("latitude_rad"))
    df["gt_longitude_deg"] = np.degrees(df.pop("longitude_rad"))
    df["attack_label"] = df["attack_enabled"]
    df["t_s"] = (df["time_usec"] - df["time_usec"].iloc[0]) / 1e6
    df["offset_m"] = horizontal_offset_m(
        df["latitude_deg"],
        df["longitude_deg"],
        df["raw_latitude_deg"],
        df["raw_longitude_deg"],
    )
    phase, truth, since, facts = label_phases(
        df["offset_m"].to_numpy(),
        df["attack_label"].to_numpy(),
        df["t_s"].to_numpy(),
        attacked_flight,
    )
    df["phase"], df["spoof_truth"], df["seconds_since_onset"] = phase, truth, since
    checked = ["raw_latitude_deg", "pressure_altitude", "orientation_w"]
    checked += ["magnetic_field_x", "gt_q_w"]
    df["aligned_all"] = df[checked].notna().all(axis=1).astype(np.int8)
    df.insert(0, "mission_type", mission)
    df.insert(0, "flight_id", flight_id)
    for col in df.columns:
        if col in FLOAT64 or col in ("flight_id", "mission_type"):
            continue
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    label = df["attack_label"].to_numpy() == 1
    spoof = truth == 1
    report = {
        "flight_id": flight_id,
        "mission": mission,
        "attacked": int(attacked_flight),
        "rows": len(df),
        "duplicates_dropped": sum(dups.values()),
        "rows_all_streams_aligned": int(df["aligned_all"].sum()),
        **{f"missed_{k}": v for k, v in misses.items()},
        "label_rows": int(label.sum()),
        "spoof_rows": int(spoof.sum()),
        "label_not_spoof_rows": int((label & ~spoof).sum()),
        "spoof_not_label_rows": int((spoof & ~label).sum()),
        "duration_s": float(df["t_s"].iloc[-1]),
        **facts,
    }
    return df, report


def main():
    inventory = pd.read_csv(INTERIM / "raw_file_inventory.csv")
    manifest = pd.read_csv(INTERIM / "flight_manifest.csv")
    inventory["log"] = (
        inventory["folder"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    )
    inventory["key"] = (
        inventory["scenario"] + "|" + inventory["condition"] + "|" + inventory["log"]
    )
    folders = dict(zip(inventory["key"], inventory["folder"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AEROVIGIL - RAW BASE TABLE (Module 03 v2)")
    print("=" * 65)
    reports, used = [], set()
    for n, (_, m) in enumerate(manifest.iterrows(), start=1):
        attacked = int(m["folder_label"]) == 1
        condition = "attacked" if attacked else "normal"
        log = str(m["flight_id"]).split("__")[-1]
        key = f"{m['mission_type']}|{condition}|{log}"
        if key not in folders:
            raise SystemExit(f"No Raw folder for {m['flight_id']} ({key})")
        df, report = build_flight(
            RAW_ROOT / folders[key], m["flight_id"], m["mission_type"], attacked
        )
        name = safe_name(m["flight_id"])
        if name in used:
            name = f"{name}_{n}"
        used.add(name)
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        report["file"] = f"{name}.parquet"
        reports.append(report)
        if n % 40 == 0:
            print(f"Built flight {n}/{len(manifest)}")

    table = pd.DataFrame(reports)
    table.to_csv(REPORT_DIR / "base_report.csv", index=False)
    merged_rows = pd.read_csv(INTERIM / "module03" / "raw_coverage.csv")["rows_merged"]
    pd.set_option("display.width", 200)
    print("\n" + "-" * 65)
    print(f"flights built            : {len(table)}")
    print(f"rows (Raw base)          : {int(table['rows'].sum()):,}")
    print(f"rows (Merged)            : {int(merged_rows.sum()):,}")
    print(f"duplicate timestamps cut : {int(table['duplicates_dropped'].sum()):,}")
    print(
        f"rows with every stream   : {int(table['rows_all_streams_aligned'].sum()):,}"
    )
    miss_cols = [c for c in table.columns if c.startswith("missed_")]
    print("rows with no stream match within 10 ms:")
    print(table[miss_cols].sum().astype(int).to_string())
    att = table[table["attacked"] == 1]
    print("\nattacked flights (median):")
    cols = ["onset_s", "ramp_end_s", "spoof_end_s", "plateau_m", "duration_s"]
    print(att.groupby("mission")[cols].median().round(2).to_string())
    print(f"label rows           : {int(att['label_rows'].sum()):,}")
    print(f"spoofed rows (truth) : {int(att['spoof_rows'].sum()):,}")
    print(f"labelled, not spoofed: {int(att['label_not_spoof_rows'].sum()):,}")
    print(f"spoofed, not labelled: {int(att['spoof_not_label_rows'].sum()):,}")
    print(
        "flights with spoofed-not-labelled rows:",
        int((att["spoof_not_label_rows"] > 0).sum()),
    )

    checks = {
        "360 flights built": len(table) == 360,
        "one file per flight": len(list(OUT_DIR.glob("*.parquet"))) == len(table),
        "every attacked flight has an onset": bool(att["onset_s"].notna().all()),
        "onset before ramp end before spoof end": bool(
            (
                (att["onset_s"] <= att["ramp_end_s"])
                & (att["ramp_end_s"] <= att["spoof_end_s"])
            ).all()
        ),
        "normal flights have no spoofed rows": bool(
            (table.loc[table["attacked"] == 0, "spoof_rows"] == 0).all()
        ),
        "more rows than Merged": int(table["rows"].sum()) > int(merged_rows.sum()),
    }
    print("\nACCEPTANCE CHECKS")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\nSaved:", OUT_DIR)
    print("Report:", REPORT_DIR / "base_report.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
