"""Step 3: check the Module 03 outputs that the attack generator depends on."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    get_columns,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)

EARTH_R = 6371000.0  # metres


def main():
    cfg = load_yaml(ROOT / "configs" / "attacks" / "attacks.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols, scale = get_columns(cfg, pre)
    clean_dir = resolve(cfg["paths"]["clean_flights_dir"])

    print("=" * 65)
    print("AEROVIGIL - MODULE 04 INPUT CHECK")
    print("=" * 65)

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    print(f"Cleaned flights in report : {len(report)}")
    print(f"Fully normal flights      : {int((report['attack_rows'] == 0).sum())}")

    raw = pd.read_csv(resolve(cfg["paths"]["flight_splits"]))
    print(f"\nflight_splits.csv columns : {list(raw.columns)}")
    print(raw.head(3).to_string(index=False))
    split_map, id_col, split_col = load_split_map(
        resolve(cfg["paths"]["flight_splits"])
    )
    print(f"Detected ID column        : {id_col}")
    print(f"Detected split column     : {split_col}")

    clean = report[report["attack_rows"] == 0].copy()
    clean["split"] = clean["flight_id"].astype(str).map(split_map)
    print(f"Clean flights without a split: {int(clean['split'].isna().sum())}")
    print("\nNORMAL SOURCE FLIGHTS: split x mission type")
    print(pd.crosstab(clean["split"], clean["mission_type"], margins=True).to_string())

    first = pd.read_parquet(clean_dir / report["file"].iloc[0])
    need = [
        cols[k]
        for k in (
            "flight",
            "time",
            "label",
            "lat",
            "lon",
            "alt",
            "vn",
            "ve",
            "vu",
            "baro",
        )
    ] + ["mission_type"]
    print(f"\nCleaned parquet columns ({len(first.columns)}): {list(first.columns)}")
    print(f"Required columns missing  : {[c for c in need if c not in first.columns]}")
    print(f"attack_enabled present    : {cols['enabled'] in first.columns}")
    print(f"Label dtype               : {first[cols['label']].dtype}")

    # Velocity frame check: which axis/sign convention matches the GPS motion?
    print(
        "\nVELOCITY FRAME CHECK (mean abs error in m/s, position-derived vs reported)"
    )
    rows = []
    for f in clean["file"].iloc[:5]:
        g = pd.read_parquet(clean_dir / f)
        t = (g[cols["time"]].to_numpy("int64") - g[cols["time"]].iloc[0]) * scale
        j = np.maximum(np.searchsorted(t, t - 1.0, side="right") - 1, 0)
        dt = t - t[j]
        ok = dt > 0.5
        lat = np.radians(g[cols["lat"]].to_numpy(float))
        lon = np.radians(g[cols["lon"]].to_numpy(float))
        pn = (lat - lat[j]) * EARTH_R / np.where(ok, dt, np.nan)
        pe = (lon - lon[j]) * EARTH_R * np.cos(lat) / np.where(ok, dt, np.nan)
        vn, ve = g[cols["vn"]].to_numpy(float), g[cols["ve"]].to_numpy(float)
        hyp = {
            "N=north,E=east (assumed)": (vn, ve),
            "N=-north": (-vn, ve),
            "E=-east": (vn, -ve),
            "axes swapped": (ve, vn),
        }
        row = {"flight": f[:34]}
        for name, (a_, b_) in hyp.items():
            row[name] = float(np.nanmean(np.hypot(pn - a_, pe - b_)[ok]))
        rows.append(row)
    vf = pd.DataFrame(rows)
    print(vf.round(3).to_string(index=False))
    best = vf.drop(columns="flight").idxmin(axis=1)
    good = bool((best == "N=north,E=east (assumed)").all())
    print(
        "Assumed north/east convention is the best match in every flight:",
        "YES" if good else "NO - STOP and send me this output",
    )

    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"\nFree disk space: {free:.1f} GB  (the full attack set needs about 4-5 GB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
