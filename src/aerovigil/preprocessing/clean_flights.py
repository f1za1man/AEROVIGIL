from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "preprocessing" / "preprocessing.yaml"
OUT_DIR = ROOT / "data" / "processed" / "module03"
CLEAN_DIR = OUT_DIR / "cleaned"
REPORT_FILE = OUT_DIR / "cleaning_report.csv"

TEXT_COLUMNS = {"flight_id", "mission_type"}
GAP_FACTOR = 10  # same large-gap rule used in timestamp_report.csv


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def main():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    flight_col = cfg["dataset"]["flight_id_column"]
    time_col = cfg["dataset"]["timestamp_column"]
    label_col = cfg["labels"]["attack_column"]
    attack_value = cfg["labels"]["attack_value"]
    scale = {"seconds": 1.0, "milliseconds": 1e-3,
             "microseconds": 1e-6, "nanoseconds": 1e-9}[
        cfg["dataset"]["timestamp_unit"]]

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AEROVIGIL - FLIGHT-LEVEL CLEANING")
    print("=" * 65)

    df = pd.read_csv(
        ROOT / cfg["dataset"]["input_file"],
        dtype={flight_col: "category", "mission_type": "category"},
        low_memory=False,
    )
    total_in = len(df)
    attack_in = int((df[label_col] == attack_value).sum())
    print(f"Rows loaded: {total_in:,}   Attack rows: {attack_in:,}")

    numeric_cols = [c for c in df.columns if c not in TEXT_COLUMNS]
    groups = df.groupby(flight_col, observed=True, sort=True).indices
    print(f"Flights found: {len(groups)}\n")

    report, used_names = [], set()

    for n, (fid, idx) in enumerate(groups.items(), start=1):
        g = df.iloc[idx].copy()
        g[flight_col] = g[flight_col].astype(str)
        g["mission_type"] = g["mission_type"].astype(str)
        rows_in = len(g)

        # Convert numeric columns; count anything that fails to parse
        coerced = 0
        for c in numeric_cols:
            before = int(g[c].isna().sum())
            g[c] = pd.to_numeric(g[c], errors="coerce")
            coerced += int(g[c].isna().sum()) - before

        # Order, duplicates, sorting (no rows are removed)
        t_raw = g[time_col].to_numpy()
        out_of_order = int((np.diff(t_raw) < 0).sum())
        duplicates = int(g[time_col].duplicated().sum())
        g = g.sort_values(time_col, kind="stable").reset_index(drop=True)
        t = g[time_col].to_numpy()
        is_sorted = bool((np.diff(t) >= 0).all())

        # Gaps (recorded, not repaired)
        dt = np.diff(t) * scale
        dt_pos = dt[dt > 0]
        if len(dt_pos):
            med = float(np.median(dt_pos))
            gaps = int((dt_pos > GAP_FACTOR * med).sum())
            max_gap = float(dt_pos.max())
        else:
            med, gaps, max_gap = np.nan, 0, np.nan

        missing_total = int(g.isna().sum().sum())
        attack_rows = int((g[label_col] == attack_value).sum())

        name = safe_name(fid)
        if name in used_names:
            name = f"{name}_{n}"
        used_names.add(name)
        out_file = CLEAN_DIR / f"{name}.parquet"
        g.to_parquet(out_file, index=False)

        report.append({
            "flight_id": fid,
            "file": out_file.name,
            "mission_type": g["mission_type"].iloc[0],
            "rows_in": rows_in,
            "rows_out": len(g),
            "out_of_order_rows": out_of_order,
            "duplicate_timestamps": duplicates,
            "sorted_after_cleaning": is_sorted,
            "missing_values": missing_total,
            "non_numeric_coerced_to_nan": coerced,
            "median_interval_s": med,
            "large_gaps": gaps,
            "max_gap_s": max_gap,
            "attack_rows": attack_rows,
        })

        if n % 40 == 0 or n == len(groups):
            print(f"Cleaned {n}/{len(groups)} flights")

    rep = pd.DataFrame(report)
    rep.to_csv(REPORT_FILE, index=False)

    # ---- Acceptance criteria -------------------------------------
    total_out = int(rep["rows_out"].sum())
    attack_out = int(rep["attack_rows"].sum())
    checks = {
        "Every flight has an ID": bool(rep["flight_id"].notna().all()
                                       and (rep["flight_id"] != "").all()),
        "Timestamps ordered in every flight": bool(rep["sorted_after_cleaning"].all()),
        "No rows lost": total_in == total_out,
        "Attack labels preserved": attack_in == attack_out,
        "One file per flight": len(list(CLEAN_DIR.glob("*.parquet"))) == len(rep),
    }

    print("\n" + "-" * 65)
    print(f"Rows in / out        : {total_in:,} / {total_out:,}")
    print(f"Attack rows in / out : {attack_in:,} / {attack_out:,}")
    print(f"Rows out of order    : {int(rep['out_of_order_rows'].sum()):,} (sorted)")
    print(f"Duplicate timestamps : {int(rep['duplicate_timestamps'].sum()):,}")
    print(f"Missing values       : {int(rep['missing_values'].sum()):,}")
    print(f"Non-numeric coerced  : {int(rep['non_numeric_coerced_to_nan'].sum()):,}")
    print(f"Large gaps recorded  : {int(rep['large_gaps'].sum()):,}")
    print(f"\nReport: {REPORT_FILE}")
    print("\nACCEPTANCE CRITERIA")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Cleaning acceptance criteria FAILED")


if __name__ == "__main__":
    main()