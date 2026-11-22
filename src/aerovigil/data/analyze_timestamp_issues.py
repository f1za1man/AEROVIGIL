from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "preprocessing" / "preprocessing.yaml"
OUT_DIR = ROOT / "data" / "processed" / "module03"
GAP_FACTOR = 10


def main():
    cfg = yaml.safe_load(CONFIG_FILE.open(encoding="utf-8"))
    flight_col = cfg["dataset"]["flight_id_column"]
    time_col = cfg["dataset"]["timestamp_column"]
    label_col = cfg["labels"]["attack_column"]
    scale = 1e-6  # microseconds -> seconds

    df = pd.read_csv(
        ROOT / cfg["dataset"]["input_file"],
        usecols=[flight_col, time_col, label_col, "mission_type"],
        dtype={flight_col: "category", "mission_type": "category"},
    )

    flight_rows, gap_rows = [], []

    for fid, g in df.groupby(flight_col, observed=True, sort=True):
        t = g[time_col].to_numpy()
        y = g[label_col].to_numpy()

        # ---- 1. Out-of-order structure -------------------------------
        order = np.argsort(t, kind="stable")
        displacement = np.abs(order - np.arange(len(t)))
        backward = np.diff(t)
        backward = -backward[backward < 0] * scale
        # Do labels change when sorted? (sanity: rows move together)
        t_s, y_s = t[order], y[order]

        # ---- 2. Gaps (after sorting) ---------------------------------
        dt = np.diff(t_s) * scale
        med = np.median(dt)
        big = np.where(dt > GAP_FACTOR * med)[0]
        for i in big:
            gap_rows.append(
                {
                    "flight_id": fid,
                    "gap_seconds": dt[i],
                    "position_fraction": t_s[i] / 1 - t_s[0],
                    "label_before": y_s[i],
                    "label_after": y_s[i + 1],
                    "median_interval_s": med,
                }
            )
            gap_rows[-1]["position_fraction"] = (
                (t_s[i] - t_s[0]) / (t_s[-1] - t_s[0])
            )

        # ---- 3. Per-flight context ----------------------------------
        label_changes = int((np.diff(y_s) != 0).sum())
        flight_rows.append(
            {
                "flight_id": fid,
                "mission_type": g["mission_type"].iloc[0],
                "hz": round(1.0 / med),
                "attack_fraction": float(y_s.mean()),
                "label_transitions": label_changes,
                "n_out_of_order": int(len(backward)),
                "max_backward_jump_s": float(backward.max()) if len(backward) else 0.0,
                "max_row_displacement": int(displacement.max()),
            }
        )

    flights = pd.DataFrame(flight_rows)
    gaps = pd.DataFrame(gap_rows)
    flights.to_csv(OUT_DIR / "flight_context.csv", index=False)
    gaps.to_csv(OUT_DIR / "gap_report.csv", index=False)

    print("=" * 65)
    print("1. OUT-OF-ORDER ROWS")
    print("=" * 65)
    print(flights[["n_out_of_order", "max_backward_jump_s",
                   "max_row_displacement"]].describe().round(4).to_string())

    print("\n" + "=" * 65)
    print("2. GAPS")
    print("=" * 65)
    print(f"Total gaps: {len(gaps)}")
    if len(gaps):
        print(gaps["gap_seconds"].describe().round(4).to_string())
        print("\nLabel just before -> after gap:")
        print(pd.crosstab(gaps["label_before"], gaps["label_after"]).to_string())
        print("\nGap position within flight (0=start, 1=end):")
        print(gaps["position_fraction"].describe().round(3).to_string())
        print("\nTen longest gaps:")
        print(gaps.nlargest(10, "gap_seconds")[
            ["flight_id", "gap_seconds", "position_fraction",
             "label_before", "label_after"]].round(3).to_string(index=False))

    print("\n" + "=" * 65)
    print("3. SAMPLING RATE vs MISSION TYPE AND ATTACK FRACTION")
    print("=" * 65)
    print(pd.crosstab(flights["hz"], flights["mission_type"]).to_string())
    print("\nMean attack fraction by rate:")
    print(flights.groupby("hz")["attack_fraction"]
          .agg(["count", "mean", "min", "max"]).round(3).to_string())

    print("\n" + "=" * 65)
    print("4. ATTACK STRUCTURE PER FLIGHT")
    print("=" * 65)
    print(flights["attack_fraction"].describe().round(3).to_string())
    print("\nLabel transitions per flight:")
    print(flights["label_transitions"].value_counts().sort_index().to_string())
    print("\nFlights fully normal :", int((flights["attack_fraction"] == 0).sum()))
    print("Flights fully attack :", int((flights["attack_fraction"] == 1).sum()))
    print("Flights mixed        :",
          int(((flights["attack_fraction"] > 0) &
               (flights["attack_fraction"] < 1)).sum()))
    print("=" * 65)


if __name__ == "__main__":
    main()