r"""AEROVIGIL Module 03 check: how a spoofing attack ends, and why Merged lost rows.

Part 1: for every attacked Raw flight whose attack_enabled returns to 0 before the
flight ends, measure when the offset really drops back below 1 m compared with when
the label ends, and how abrupt the drop is.
Part 2: for a sample of flights, count rows in each Raw stream and how many
timestamps all streams share (the Merged join keeps only those).

Needs check_raw_coverage.py to have run.

Writes data/interim/module03/attack_end.csv and stream_rates.csv

Usage:
    python src\aerovigil\data\check_attack_end.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INTERIM = PROJECT_ROOT / "data" / "interim"
CLEAN_DIR = PROJECT_ROOT / "data" / "processed" / "module03" / "cleaned"
OUT_DIR = INTERIM / "module03"
METRES_PER_DEGREE = 111320.0
STREAMS = ["gps", "raw_gps", "imu", "barometer", "magnetometer", "groundtruth"]
PER_GROUP = 4


def offset_series(gps, raw):
    both = pd.merge_asof(
        gps[["time_usec", "latitude_deg", "longitude_deg", "attack_enabled"]],
        raw[["time_usec", "latitude_deg", "longitude_deg"]],
        on="time_usec",
        direction="nearest",
        suffixes=("_gps", "_raw"),
    )
    lat = np.radians(both["latitude_deg_raw"])
    north = (both["latitude_deg_gps"] - both["latitude_deg_raw"]) * METRES_PER_DEGREE
    east = (
        (both["longitude_deg_gps"] - both["longitude_deg_raw"])
        * METRES_PER_DEGREE
        * np.cos(lat)
    )
    both["offset_m"] = np.hypot(north, east)
    both["t_s"] = (both["time_usec"] - both["time_usec"].iloc[0]) / 1e6
    return both


def label_runs(on):
    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    return int((edges == 1).sum())


def attack_end_row(both):
    on = both["attack_enabled"].to_numpy() == 1
    t = both["t_s"].to_numpy()
    off = both["offset_m"].to_numpy()
    last_label = int(np.flatnonzero(on)[-1])
    over = np.flatnonzero(off >= 1.0)
    last_over = int(over[-1]) if len(over) else 0
    window = slice(max(last_over - 250, 0), last_over + 1)
    steps = np.abs(np.diff(off[window])) if last_over > 0 else np.array([0.0])
    return {
        "label_runs": label_runs(on),
        "t_label_end_s": float(t[last_label]),
        "t_offset_end_s": float(t[last_over]),
        "label_minus_offset_s": float(t[last_label] - t[last_over]),
        "offset_peak_m": float(off.max()),
        "offset_at_last_over_m": float(off[last_over]),
        "largest_step_before_end_m": float(steps.max()),
        "offset_back_above_1m_after_label": bool(last_over > last_label),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(OUT_DIR / "raw_coverage.csv")
    inventory = pd.read_csv(INTERIM / "raw_file_inventory.csv")
    inventory["log"] = (
        inventory["folder"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    )
    inventory["key"] = (
        inventory["scenario"] + "|" + inventory["condition"] + "|" + inventory["log"]
    )
    coverage["log"] = coverage["flight_id"].str.split("__").str[-1]
    coverage["key"] = (
        coverage["mission"] + "|" + coverage["condition"] + "|" + coverage["log"]
    )
    folders = dict(zip(inventory["key"], inventory["folder"]))

    print("=" * 65)
    print("AEROVIGIL - ATTACK END AND STREAM RATES (read only)")
    print("=" * 65)

    ended = coverage[coverage["attack_runs_to_end"].eq(False)]
    rows = []
    for _, c in ended.iterrows():
        folder = RAW_ROOT / folders[c["key"]]
        gps = pd.read_csv(folder / "gps.csv").sort_values("time_usec")
        raw = pd.read_csv(folder / "raw_gps.csv").sort_values("time_usec")
        rows.append(
            {
                "flight_id": c["flight_id"],
                "mission": c["mission"],
                **attack_end_row(offset_series(gps, raw)),
            }
        )
    end_table = pd.DataFrame(rows)
    end_table.to_csv(OUT_DIR / "attack_end.csv", index=False)
    print(f"\nPart 1: attacked flights whose label ends early: {len(end_table)}")
    if len(end_table):
        cols = [
            "label_runs",
            "label_minus_offset_s",
            "offset_peak_m",
            "offset_at_last_over_m",
            "largest_step_before_end_m",
        ]
        print(end_table[cols].describe().round(3).to_string())
        print(
            "flights with more than one label run:",
            int((end_table["label_runs"] > 1).sum()),
        )
        print(
            "flights where offset is back above 1 m after the label ends:",
            int(end_table["offset_back_above_1m_after_label"].sum()),
        )
        print(
            "per mission (median label_minus_offset_s):",
            end_table.groupby("mission")["label_minus_offset_s"]
            .median()
            .round(3)
            .to_dict(),
        )

    sample = (
        coverage[coverage["status"] == "ok"]
        .groupby(["mission", "condition"], group_keys=False)
        .head(PER_GROUP)
    )
    rows = []
    for _, c in sample.iterrows():
        folder = RAW_ROOT / folders[c["key"]]
        times = {
            s: pd.read_csv(folder / f"{s}.csv", usecols=["time_usec"])["time_usec"]
            for s in STREAMS
        }
        common_sensors = set(times["gps"])
        for s in ["imu", "barometer", "magnetometer"]:
            common_sensors &= set(times[s])
        row = {"flight_id": c["flight_id"], "rows_merged": int(c["rows_merged"])}
        for s in STREAMS:
            row[f"rows_{s}"] = len(times[s])
        row["common_gps_imu_baro_mag"] = len(common_sensors)
        for s in ["raw_gps", "imu", "barometer", "magnetometer", "groundtruth"]:
            row[f"gps_in_{s}"] = int(times["gps"].isin(times[s]).sum())
        rows.append(row)
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT_DIR / "stream_rates.csv", index=False)
    print(f"\nPart 2: stream row counts in {len(rates)} sampled flights (median)")
    print(rates.drop(columns="flight_id").median().round(0).astype(int).to_string())
    print("Saved:", OUT_DIR / "attack_end.csv", "and", OUT_DIR / "stream_rates.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
