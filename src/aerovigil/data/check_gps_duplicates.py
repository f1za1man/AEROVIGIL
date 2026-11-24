r"""AEROVIGIL Module 03 v2 check: the two copies of a repeated gps.csv row (read only).

gps.csv logs many timestamps twice, and in about 5 % of those pairs the two copies
differ. The base table keeps the first copy. This shows which columns differ, where
in the attack the differing pairs lie, and which copy is closer to the unspoofed
raw_gps position, so we know whether keeping the first copy is right.

Usage:
    python src\aerovigil\data\check_gps_duplicates.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INTERIM = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = INTERIM / "module03_v2"
METRES_PER_DEGREE = 111320.0
PER_GROUP = 4


def offset_m(lat, lon, raw_lat, raw_lon):
    north = (lat - raw_lat) * METRES_PER_DEGREE
    east = (lon - raw_lon) * METRES_PER_DEGREE * np.cos(np.radians(raw_lat))
    return np.hypot(north, east)


def pair_table(gps, raw):
    gps = gps.copy()
    gps["copy"] = gps.groupby("time_usec").cumcount()
    first = gps[gps["copy"] == 0].set_index("time_usec")
    second = gps[gps["copy"] == 1].set_index("time_usec")
    common = first.index.intersection(second.index)
    first, second = first.loc[common], second.loc[common]
    cols = [c for c in first.columns if c not in ("copy", "time_utc_usec")]
    differs = pd.DataFrame(index=common)
    for col in cols:
        same = (first[col] == second[col]) | (first[col].isna() & second[col].isna())
        differs[col] = ~same
    raw = raw.drop_duplicates("time_usec").set_index("time_usec")
    ref = raw.reindex(common)
    out = pd.DataFrame(index=common)
    out["any_diff"] = differs.any(axis=1)
    out["offset_first"] = offset_m(
        first["latitude_deg"],
        first["longitude_deg"],
        ref["latitude_deg"],
        ref["longitude_deg"],
    )
    out["offset_second"] = offset_m(
        second["latitude_deg"],
        second["longitude_deg"],
        ref["latitude_deg"],
        ref["longitude_deg"],
    )
    out["label_first"] = first["attack_enabled"]
    out["label_second"] = second["attack_enabled"]
    onset = gps.loc[gps["attack_enabled"] == 1, "time_usec"]
    start = float(onset.iloc[0]) if len(onset) else np.nan
    out["since_onset_s"] = (out.index.to_numpy() - start) / 1e6
    return out, differs


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
    manifest["condition"] = np.where(
        manifest["folder_label"] == 1, "attacked", "normal"
    )
    manifest["key"] = (
        manifest["mission_type"]
        + "|"
        + manifest["condition"]
        + "|"
        + manifest["flight_id"].str.split("__").str[-1]
    )
    sample = manifest.groupby(["mission_type", "condition"], group_keys=False).head(
        PER_GROUP
    )

    print("=" * 65)
    print("AEROVIGIL - GPS DUPLICATE PAIRS (read only)")
    print("=" * 65)
    pairs, column_counts, per_flight = [], pd.Series(dtype="int64"), []
    for _, m in sample.iterrows():
        folder = RAW_ROOT / folders[m["key"]]
        gps = pd.read_csv(folder / "gps.csv")
        raw = pd.read_csv(folder / "raw_gps.csv")
        table, differs = pair_table(gps, raw)
        table["flight_id"] = m["flight_id"]
        table["condition"] = m["condition"]
        pairs.append(table)
        column_counts = column_counts.add(differs.sum(), fill_value=0)
        per_flight.append(
            {
                "flight_id": m["flight_id"],
                "pairs": len(table),
                "differing_pairs": int(table["any_diff"].sum()),
            }
        )
    allp = pd.concat(pairs)
    pd.set_option("display.width", 200)
    print("\npairs per sampled flight:")
    print(pd.DataFrame(per_flight).to_string(index=False))
    print("\ncolumns that differ between the two copies (count of pairs):")
    print(column_counts[column_counts > 0].astype(int).to_string())
    diff = allp[allp["any_diff"]]
    print(f"\ndiffering pairs: {len(diff):,} of {len(allp):,}")
    if len(diff):
        print("\nmedian offset to raw_gps in differing pairs (m):")
        print(f"  first copy  : {diff['offset_first'].median():.2f}")
        print(f"  second copy : {diff['offset_second'].median():.2f}")
        print(
            "  second copy more spoofed than first:",
            f"{(diff['offset_second'] > diff['offset_first']).mean():.1%}",
        )
        print(
            "  label differs between copies:",
            int((diff["label_first"] != diff["label_second"]).sum()),
        )
        att = diff[diff["condition"] == "attacked"]
        print("\nseconds since the first labelled row, differing pairs (attacked):")
        print(att["since_onset_s"].describe().round(2).to_string())
        print(
            "  share within 0-6 s of onset:",
            f"{att['since_onset_s'].between(0, 6).mean():.1%}",
        )
        print(
            "  differing pairs in normal flights:",
            int((diff["condition"] == "normal").sum()),
        )
    allp.reset_index().to_csv(REPORT_DIR / "gps_duplicate_pairs.csv", index=False)
    print("\nSaved:", REPORT_DIR / "gps_duplicate_pairs.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
