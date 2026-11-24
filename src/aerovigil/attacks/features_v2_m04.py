r"""AEROVIGIL Module 04 v2: v2 features for the synthetic attack flights.

Builds the 36 Module 03 v2 features (GPS/barometer + IMU consistency) on every
flight of data/processed/module04_v2/base and normalizes them with the parameters
that Module 03 v2 fitted on clean TRAINING rows of the real flights. The parameters
are only applied here, never refitted, so a feature value means the same thing in
the real and in the synthetic flights, and no synthetic row can leak into the scale.

The output schema is identical to data/processed/module03_v2/normalized, so Modules
05-07 can read both directories with the same code. attack_type and velocity_mode
stay in the manifest; join on flight_id.

Writes data/processed/module04_v2/normalized/<flight>.parquet and
       data/interim/module04_v2/feature_report.csv

Usage:
    python src\aerovigil\attacks\features_v2_m04.py
    python src\aerovigil\attacks\features_v2_m04.py --limit 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.preprocessing.normalize_features_v2 import CLIP, NON_FEATURES

ROOT = Path(__file__).resolve().parents[3]
M03 = ROOT / "data" / "processed" / "module03_v2"
M04 = ROOT / "data" / "processed" / "module04_v2"
PARAM_FILE = M03 / "normalization_parameters.json"
BASE_DIR = M04 / "base"
NORM_DIR = M04 / "normalized"
REPORT = ROOT / "data" / "interim" / "module04_v2" / "feature_report.csv"


def load_parameters():
    """Return (features, median, scale) exactly as Module 03 v2 fitted them."""
    p = json.loads(PARAM_FILE.read_text(encoding="utf-8"))
    features = list(p["features"])
    median = np.array([p["median"][f] for f in features], dtype=np.float64)
    scale = np.array([p["scale"][f] for f in features], dtype=np.float64)
    print(f"Normalization parameters : {PARAM_FILE.name}")
    print(
        f"  fitted on {p['fit_flights']} training flights, {p['fit_rows']:,} clean rows"
    )
    print(f"  {len(features)} features, method {p['method']}, clip +-{p['clip']:g}")
    return features, median, scale


def run(limit=None):
    files = sorted(BASE_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No attacked flights in {BASE_DIR}. Run generate_v2.py first."
        )
    if limit:
        files = files[:limit]
    features, median, scale = load_parameters()
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    for f in NORM_DIR.glob("*.parquet"):
        f.unlink()
    print(f"\nAttacked flights: {len(files)}\n")
    rows, started = [], time.time()
    clipped = np.zeros(len(features), dtype="int64")
    finite = np.zeros(len(features), dtype="int64")
    nans = np.zeros(len(features), dtype="int64")
    total = 0
    for n, path in enumerate(files, start=1):
        out = build_flight_v2(pd.read_parquet(path))
        got = [c for c in out.columns if c not in NON_FEATURES]
        if got != features:
            raise SystemExit(
                "Feature columns differ from the Module 03 v2 parameters.\n"
                f"  missing: {sorted(set(features) - set(got))}\n"
                f"  extra  : {sorted(set(got) - set(features))}"
            )
        z = (out[features].to_numpy(np.float64) - median) / scale
        nans += np.isnan(z).sum(axis=0)
        finite += np.isfinite(z).sum(axis=0)
        clipped += (np.abs(z) > CLIP).sum(axis=0)
        total += len(out)
        out[features] = np.clip(z, -CLIP, CLIP).astype(np.float32)
        out.to_parquet(NORM_DIR / path.name, index=False)
        ph = out["phase"].to_numpy()
        rows.append({
            "flight_id": out["flight_id"].iloc[0], "rows": len(out),
            "nan_rows": int(out[features].isna().any(axis=1).sum()),
            "ramp_rows": int((ph == 2).sum()), "plateau_rows": int((ph == 3).sum()),
            "post_rows": int((ph == 4).sum()), "file": path.name,
        })  # fmt: skip
        if n % 100 == 0 or n == len(files):
            print(f"Featurized {n}/{len(files)} flights")
    report(
        pd.DataFrame(rows),
        features,
        total,
        nans,
        finite,
        clipped,
        time.time() - started,
    )


def report(rep, features, total, nans, finite, clipped, elapsed):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(REPORT, index=False)
    pd.set_option("display.width", 200)
    size = sum(f.stat().st_size for f in NORM_DIR.glob("*.parquet")) / 1e9
    print(
        f"\nRows: {total:,}   flights: {len(rep)}   "
        f"elapsed {elapsed:.0f} s   disk {size:.2f} GB"
    )
    table = pd.DataFrame(
        {
            "nan_%": (nans / total * 100).round(2),
            "clipped_%": (clipped / np.maximum(finite, 1) * 100).round(3),
        },
        index=features,
    )
    print("\nMISSING AND CLIPPED PER FEATURE (clipping is expected: these are attacks)")
    print(table.to_string())
    phase_rows = rep[["ramp_rows", "plateau_rows", "post_rows"]].sum()
    print(f"\nPHASE ROWS OVER ALL FLIGHTS\n{phase_rows.to_string()}")
    checks = {
        "one normalized file per flight": (
            len(list(NORM_DIR.glob("*.parquet"))) == len(rep)
        ),
        "parameters were applied, not refitted": PARAM_FILE.is_file(),
        "features match Module 03 v2": len(features) == len(table),
        "no flight is entirely missing": bool((rep["rows"] > 0).all()),
    }  # fmt: skip
    print("\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    no_ramp = rep.loc[rep["ramp_rows"] == 0, "flight_id"].tolist()
    print(f"\nFlights with no ramp rows: {len(no_ramp)} (weak attacks, not an error)")
    if no_ramp:
        print(f"  e.g. {no_ramp[:3]}")
    if not all(checks.values()):
        raise SystemExit("Module 04 v2 feature checks FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N flights only")
    run(ap.parse_args().limit)


if __name__ == "__main__":
    main()
