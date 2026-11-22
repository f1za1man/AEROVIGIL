from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "preprocessing" / "preprocessing.yaml"
OUTPUT_DIR = ROOT / "data" / "processed" / "module03"
REPORT_FILE = OUTPUT_DIR / "timestamp_report.csv"

UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}
LARGE_GAP_FACTOR = 10  # a gap is "large" if > 10 x the flight's median interval


def main():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_file = ROOT / cfg["dataset"]["input_file"]
    flight_col = cfg["dataset"]["flight_id_column"]
    time_col = cfg["dataset"]["timestamp_column"]
    unit = cfg["dataset"]["timestamp_unit"]
    scale = UNIT_TO_SECONDS[unit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AEROVIGIL - TIMESTAMP INSPECTION")
    print("=" * 65)
    print(f"Flight column : {flight_col}")
    print(f"Time column   : {time_col} ({unit})")

    # Only three columns are needed, so this fits comfortably in memory.
    df = pd.read_csv(
        input_file,
        usecols=[flight_col, time_col, "timestamp"],
        dtype={flight_col: "category", time_col: "int64", "timestamp": "int64"},
    )
    print(f"Rows loaded   : {len(df):,}")

    # Check 1: are time_usec and timestamp identical?
    identical = bool((df[time_col] == df["timestamp"]).all())
    print(f"\n{time_col} == timestamp for all rows: {identical}")

    rows = []
    for flight_id, g in df.groupby(flight_col, observed=True, sort=True):
        t_file_order = g[time_col].to_numpy()

        # Order as stored in the file (before any sorting)
        file_diff = np.diff(t_file_order)
        out_of_order = int((file_diff < 0).sum())

        # Statistics after sorting
        t = np.sort(t_file_order)
        diff = np.diff(t)
        duplicates = int((diff == 0).sum())
        positive = diff[diff > 0] * scale

        if len(positive) > 0:
            median_dt = float(np.median(positive))
            mean_dt = float(positive.mean())
            p05, p95 = np.percentile(positive, [5, 95])
            large_gaps = int((positive > LARGE_GAP_FACTOR * median_dt).sum())
            max_gap = float(positive.max())
            est_hz = 1.0 / median_dt if median_dt > 0 else np.nan
        else:
            median_dt = mean_dt = p05 = p95 = max_gap = est_hz = np.nan
            large_gaps = 0

        rows.append(
            {
                "flight_id": flight_id,
                "records": len(t),
                "first_timestamp": int(t[0]),
                "last_timestamp": int(t[-1]),
                "duration_seconds": float((t[-1] - t[0]) * scale),
                "median_interval_s": median_dt,
                "mean_interval_s": mean_dt,
                "p05_interval_s": float(p05),
                "p95_interval_s": float(p95),
                "max_gap_s": max_gap,
                "estimated_hz": est_hz,
                "duplicate_timestamps": duplicates,
                "out_of_order_in_file": out_of_order,
                "large_gaps": large_gaps,
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_FILE, index=False)

    print("\n" + "-" * 65)
    print(f"Flights analysed: {len(report)}")
    print(f"Report saved to : {REPORT_FILE}")

    print("\nESTIMATED SAMPLING RATE (Hz)")
    print(report["estimated_hz"].describe().round(3).to_string())

    print("\nDURATION (seconds)")
    print(report["duration_seconds"].describe().round(2).to_string())

    print("\nDATA QUALITY")
    print(f"Flights with duplicate timestamps : "
          f"{int((report['duplicate_timestamps'] > 0).sum())} "
          f"(total {int(report['duplicate_timestamps'].sum()):,})")
    print(f"Flights with out-of-order rows    : "
          f"{int((report['out_of_order_in_file'] > 0).sum())} "
          f"(total {int(report['out_of_order_in_file'].sum()):,})")
    print(f"Flights with large gaps           : "
          f"{int((report['large_gaps'] > 0).sum())} "
          f"(total {int(report['large_gaps'].sum()):,})")
    print(f"Largest single gap (seconds)      : "
          f"{report['max_gap_s'].max():.4f}")

    print("\nUNIQUE ROUNDED RATES (Hz) AND FLIGHT COUNTS")
    print(report["estimated_hz"].round(0).value_counts().sort_index().to_string())
    print("=" * 65)


if __name__ == "__main__":
    main()