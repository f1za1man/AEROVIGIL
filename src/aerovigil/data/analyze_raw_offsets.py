r"""AEROVIGIL Module 02: true spoofing offset of every raw attacked flight (read only).

In the Raw data, raw_gps holds the position before spoofing and gps the spoofed one
(checked with check_raw_streams.py). Their horizontal difference is the real offset the
attack added. This script measures, per attacked raw flight, when the attack starts, how
fast the offset grows, how large it gets, and whether the flight is also in the Merged
data. Needs inspect_raw_files.py and inspect_dataset.py to have run.

Writes data/interim/raw_offset_summary.csv

Usage:
    python src\aerovigil\data\analyze_raw_offsets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INTERIM = PROJECT_ROOT / "data" / "interim"
METRES_PER_DEGREE = 111320.0


def offset_table(gps, raw):
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


def summarize(both):
    on = both["attack_enabled"] == 1
    out = {"rows": len(both), "attack_rows": int(on.sum())}
    keys = [
        "onset_s",
        "max_offset_m",
        "plateau_offset_m",
        "t90_s",
        "offset_1s",
        "offset_5s",
        "offset_10s",
        "share_attack_rows_below_1m",
    ]
    if not on.any():
        return {**out, **dict.fromkeys(keys, np.nan)}
    onset = float(both.loc[on, "t_s"].iloc[0])
    rel = both["t_s"] - onset
    offset = both["offset_m"]
    attack = offset[on]
    peak = float(attack.max())
    reach = rel[on][attack >= 0.9 * peak]

    def after(seconds):
        later = offset[rel >= seconds]
        return float(later.iloc[0]) if len(later) else np.nan

    out.update(
        onset_s=onset,
        max_offset_m=peak,
        plateau_offset_m=float(attack.iloc[len(attack) // 2 :].median()),
        t90_s=float(reach.iloc[0]) if len(reach) else np.nan,
        offset_1s=after(1.0),
        offset_5s=after(5.0),
        offset_10s=after(10.0),
        share_attack_rows_below_1m=float((attack < 1.0).mean()),
    )
    return out


def main():
    inventory = pd.read_csv(INTERIM / "raw_file_inventory.csv")
    manifest = pd.read_csv(INTERIM / "flight_manifest.csv")
    merged = set(
        zip(
            manifest["mission_type"],
            manifest["folder_label"],
            manifest["flight_id"].str.split("__").str[-1],
        )
    )
    attacked = inventory[inventory["condition"] == "attacked"]

    rows = []
    for _, entry in attacked.iterrows():
        folder = RAW_ROOT / entry["folder"]
        gps = pd.read_csv(folder / "gps.csv").sort_values("time_usec")
        raw = pd.read_csv(folder / "raw_gps.csv").sort_values("time_usec")
        log = Path(entry["folder"]).name
        row = {
            "scenario": entry["scenario"],
            "log": log,
            "in_merged": (entry["scenario"], 1, log) in merged,
            **summarize(offset_table(gps, raw)),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(INTERIM / "raw_offset_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 65)
    print("AEROVIGIL - RAW SPOOFING OFFSET PER ATTACKED FLIGHT (read only)")
    print("=" * 65)
    print(f"attacked raw flights: {len(table)}")
    print("\nflights by scenario and presence in the Merged data:")
    print(table.groupby(["scenario", "in_merged"]).size().to_string())
    cols = [
        "onset_s",
        "max_offset_m",
        "plateau_offset_m",
        "t90_s",
        "offset_1s",
        "offset_5s",
        "offset_10s",
        "share_attack_rows_below_1m",
    ]
    print("\nmedian per group (metres and seconds):")
    print(table.groupby(["scenario", "in_merged"])[cols].median().round(2).to_string())
    missing = table[~table["in_merged"]]
    if len(missing):
        print("\nraw attacked flights that are NOT in Merged (first 5 and last 3):")
        print(missing[["scenario", "log"]].head(5).to_string(index=False))
        print(missing[["scenario", "log"]].tail(3).to_string(index=False))
    print("\nSaved:", INTERIM / "raw_offset_summary.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
