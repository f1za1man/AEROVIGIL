r"""AEROVIGIL Module 03 check: does Merged cover the whole Raw flight? (read only)

For every Merged flight that has a Raw folder this compares row counts, time ranges
and the attack window, and measures what the spoofing offset does after the attack
label returns to 0. Needs: inspect_raw_files.py, inspect_dataset.py and
clean_flights.py to have run.

Writes data/interim/module03/raw_coverage.csv

Usage:
    python src\aerovigil\data\check_raw_coverage.py
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
CLEAN_DIR = PROJECT_ROOT / "data" / "processed" / "module03" / "cleaned"
OUT_DIR = INTERIM / "module03"
METRES_PER_DEGREE = 111320.0


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


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


def after_end(both):
    """Offset 1, 5 and 10 s after the last attack_enabled==1 row, if the flight goes on."""
    on = both["attack_enabled"].to_numpy() == 1
    if not on.any():
        return {}
    last = int(np.flatnonzero(on)[-1])
    if last >= len(both) - 1:
        return {"attack_runs_to_end": True}
    end_t = float(both["t_s"].iloc[last])
    out = {
        "attack_runs_to_end": False,
        "seconds_after_end": float(both["t_s"].iloc[-1]) - end_t,
    }
    out["offset_before_end_m"] = float(both["offset_m"].iloc[last])
    for s in (1, 5, 10):
        later = both.loc[both["t_s"] >= end_t + s, "offset_m"]
        out[f"offset_end_plus_{s}s_m"] = float(later.iloc[0]) if len(later) else np.nan
    return out


def main():
    inventory = pd.read_csv(INTERIM / "raw_file_inventory.csv")
    manifest = pd.read_csv(INTERIM / "flight_manifest.csv")
    inventory["log"] = (
        inventory["folder"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    )
    inventory["key"] = (
        inventory["scenario"] + "|" + inventory["condition"] + "|" + inventory["log"]
    )

    rows = []
    for _, m in manifest.iterrows():
        condition = "attacked" if int(m["folder_label"]) == 1 else "normal"
        log = str(m["flight_id"]).split("__")[-1]
        key = f"{m['mission_type']}|{condition}|{log}"
        hit = inventory[inventory["key"] == key]
        row = {
            "flight_id": m["flight_id"],
            "mission": m["mission_type"],
            "condition": condition,
        }
        clean = CLEAN_DIR / f"{safe_name(m['flight_id'])}.parquet"
        if hit.empty or not clean.exists():
            row["status"] = "no_raw_folder" if hit.empty else "no_cleaned_file"
            rows.append(row)
            continue
        folder = RAW_ROOT / hit.iloc[0]["folder"]
        merged = pd.read_parquet(clean, columns=["time_usec", "attack_label"])
        merged = merged.sort_values("time_usec")
        gps = pd.read_csv(folder / "gps.csv").sort_values("time_usec")
        raw_gps = pd.read_csv(folder / "raw_gps.csv").sort_values("time_usec")
        mt, rt = merged["time_usec"].to_numpy(), gps["time_usec"].to_numpy()
        step_m = float(np.median(np.diff(mt))) / 1e6 if len(mt) > 1 else np.nan
        step_r = float(np.median(np.diff(rt))) / 1e6 if len(rt) > 1 else np.nan
        row.update(
            status="ok",
            rows_merged=len(merged),
            rows_raw_gps=len(gps),
            row_ratio=len(merged) / max(len(gps), 1),
            step_merged_s=step_m,
            step_raw_s=step_r,
            exact_time_match=float(np.isin(mt, rt).mean()),
            start_gap_s=(mt[0] - rt[0]) / 1e6,
            end_gap_s=(rt[-1] - mt[-1]) / 1e6,
            raw_duration_s=(rt[-1] - rt[0]) / 1e6,
            merged_duration_s=(mt[-1] - mt[0]) / 1e6,
            attack_rows_merged=int((merged["attack_label"] == 1).sum()),
            attack_rows_raw=int((gps["attack_enabled"] == 1).sum()),
        )
        if condition == "attacked":
            row.update(after_end(offset_series(gps, raw_gps)))
        rows.append(row)

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "raw_coverage.csv", index=False)
    ok = table[table["status"] == "ok"]

    pd.set_option("display.width", 200)
    print("=" * 65)
    print("AEROVIGIL - MERGED VS RAW COVERAGE (read only)")
    print("=" * 65)
    print("status counts:", table["status"].value_counts().to_dict())
    if ok.empty:
        print("No flight could be compared.")
        return
    cols = [
        "row_ratio",
        "step_merged_s",
        "step_raw_s",
        "exact_time_match",
        "start_gap_s",
        "end_gap_s",
        "raw_duration_s",
        "merged_duration_s",
    ]
    print("\nmedian per mission and condition:")
    print(ok.groupby(["mission", "condition"])[cols].median().round(3).to_string())
    print(
        "\nflights with row_ratio < 0.95:",
        int((ok["row_ratio"] < 0.95).sum()),
        "of",
        len(ok),
    )
    print(
        "flights with exact_time_match < 0.99:",
        int((ok["exact_time_match"] < 0.99).sum()),
    )
    print("flights with |start_gap| > 1 s:", int((ok["start_gap_s"].abs() > 1).sum()))
    print("flights with |end_gap| > 1 s:", int((ok["end_gap_s"].abs() > 1).sum()))
    print(
        "attack rows merged / raw (sum):",
        int(ok["attack_rows_merged"].sum()),
        "/",
        int(ok["attack_rows_raw"].sum()),
    )
    if "offset_end_plus_5s_m" in ok:
        ended = ok[ok["attack_runs_to_end"].eq(False)]
        print(
            "\nattacked flights whose attack label ends before the flight does:",
            len(ended),
        )
        if len(ended):
            print(
                ended[
                    [
                        "offset_before_end_m",
                        "offset_end_plus_1s_m",
                        "offset_end_plus_5s_m",
                        "offset_end_plus_10s_m",
                        "seconds_after_end",
                    ]
                ]
                .describe()
                .round(2)
                .to_string()
            )
    print("\nSaved:", OUT_DIR / "raw_coverage.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
