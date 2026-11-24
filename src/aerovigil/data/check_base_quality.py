r"""AEROVIGIL Module 03 v2 check: duplicate timestamps and missing-stream gaps (read only).

Part 1: in a sample of Raw flights, how many rows repeat a time_usec in each stream,
and whether the repeated rows hold different values (then "keep the first" would
throw information away).
Part 2: in the Raw base table, whether rows with a missing stream come in long gaps
or as scattered single rows.

Needs build_raw_base.py to have run.

Usage:
    python src\aerovigil\data\check_base_quality.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INTERIM = PROJECT_ROOT / "data" / "interim"
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "module03_v2" / "base"
REPORT_DIR = INTERIM / "module03_v2"
STREAMS = ["gps", "raw_gps", "imu", "barometer", "magnetometer", "groundtruth"]
PER_GROUP = 4
CHECKED = ["raw_latitude_deg", "pressure_altitude", "orientation_w"]
CHECKED += ["magnetic_field_x", "gt_q_w"]
GAP_SECONDS = 0.1


def duplicate_stats(df):
    dup_mask = df["time_usec"].duplicated(keep=False)
    extra = int(df["time_usec"].duplicated().sum())
    if not dup_mask.any():
        return {"extra_rows": 0, "groups": 0, "groups_with_different_values": 0}
    groups = df[dup_mask].groupby("time_usec")
    differing = int((groups.nunique(dropna=False).max(axis=1) > 1).sum())
    return {
        "extra_rows": extra,
        "groups": int(groups.ngroups),
        "groups_with_different_values": differing,
    }


def missing_runs(t_s, missing):
    """Return (number of runs, longest run in seconds, rows in runs > GAP_SECONDS)."""
    if not missing.any():
        return 0, 0.0, 0
    edges = np.diff(np.concatenate([[0], missing.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    lengths = t_s[ends] - t_s[starts]
    long_rows = sum(
        int(e - s + 1) for s, e, d in zip(starts, ends, lengths) if d > GAP_SECONDS
    )
    return len(starts), float(lengths.max()), long_rows


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

    print("=" * 65)
    print("AEROVIGIL - BASE TABLE QUALITY (read only)")
    print("=" * 65)

    sample = manifest.groupby(["mission_type", "condition"], group_keys=False).head(
        PER_GROUP
    )
    rows = []
    for _, m in sample.iterrows():
        folder = RAW_ROOT / folders[m["key"]]
        for name in STREAMS:
            stats = duplicate_stats(pd.read_csv(folder / f"{name}.csv"))
            rows.append({"stream": name, "rows_total": 0, **stats})
    dup = pd.DataFrame(rows).drop(columns="rows_total")
    summary = dup.groupby("stream").sum()
    print(f"\nPart 1: repeated time_usec in {len(sample)} sampled flights")
    print(summary.reindex(STREAMS).to_string())

    gap_rows = []
    for path in sorted(BASE_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["t_s", *CHECKED])
        t_s = df["t_s"].to_numpy()
        for col in CHECKED:
            runs, longest, long_rows = missing_runs(t_s, df[col].isna().to_numpy())
            gap_rows.append(
                {
                    "flight": path.stem,
                    "column": col,
                    "missing_rows": int(df[col].isna().sum()),
                    "runs": runs,
                    "longest_run_s": longest,
                    "rows_in_runs_over_0.1s": long_rows,
                }
            )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(REPORT_DIR / "base_gaps.csv", index=False)
    by_col = gaps.groupby("column").agg(
        missing_rows=("missing_rows", "sum"),
        runs=("runs", "sum"),
        longest_run_s=("longest_run_s", "max"),
        rows_in_long_runs=("rows_in_runs_over_0.1s", "sum"),
    )
    print("\nPart 2: missing rows per column, over all 360 flights")
    print(by_col.to_string())
    print("\nSaved:", REPORT_DIR / "base_gaps.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
