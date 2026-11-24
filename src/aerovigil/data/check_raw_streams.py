r"""AEROVIGIL Module 02: compare gps, raw_gps and groundtruth in one attacked raw flight.

Read-only. Needs data/interim/raw_file_inventory.csv (from inspect_raw_files.py).

Usage:
    python src\aerovigil\data\check_raw_streams.py
    python src\aerovigil\data\check_raw_streams.py --scenario curved
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INVENTORY = PROJECT_ROOT / "data" / "interim" / "raw_file_inventory.csv"
METRES_PER_DEGREE = 111320.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="straight")
    args = ap.parse_args()

    inventory = pd.read_csv(INVENTORY)
    chosen = inventory[
        (inventory["scenario"] == args.scenario)
        & (inventory["condition"] == "attacked")
    ]
    if chosen.empty:
        raise SystemExit(f"No attacked {args.scenario} flight in the inventory.")
    folder = RAW_ROOT / chosen.iloc[0]["folder"]

    gps = pd.read_csv(folder / "gps.csv").sort_values("time_usec")
    raw = pd.read_csv(folder / "raw_gps.csv").sort_values("time_usec")
    truth = pd.read_csv(folder / "groundtruth.csv").sort_values("time_usec")

    print("=" * 65)
    print("AEROVIGIL - RAW STREAM CHECK (read only)")
    print("=" * 65)
    print(folder)
    print("rows gps / raw_gps / groundtruth:", len(gps), len(raw), len(truth))
    print("attack_enabled in gps    :", gps["attack_enabled"].value_counts().to_dict())
    print("attack_enabled in raw_gps:", raw["attack_enabled"].value_counts().to_dict())
    for name, table in [("gps", gps), ("raw_gps", raw), ("groundtruth", truth)]:
        start, end = int(table["time_usec"].min()), int(table["time_usec"].max())
        print(f"{name:12s} time_usec {start} .. {end}")

    both = pd.merge_asof(
        gps[["time_usec", "latitude_deg", "attack_enabled"]],
        raw[["time_usec", "latitude_deg"]],
        on="time_usec",
        direction="nearest",
        suffixes=("_gps", "_raw"),
    )
    diff_m = (both["latitude_deg_gps"] - both["latitude_deg_raw"]) * METRES_PER_DEGREE
    attack = both["attack_enabled"] == 1
    print("\ngps minus raw_gps, latitude (m):")
    print("  max |difference|            :", round(float(diff_m.abs().max()), 3))
    print("  rows with attack_enabled = 1:", int(attack.sum()), "of", len(both))
    print(
        "  mean |difference| attack = 0:", round(float(diff_m[~attack].abs().mean()), 3)
    )
    print(
        "  mean |difference| attack = 1:", round(float(diff_m[attack].abs().mean()), 3)
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
