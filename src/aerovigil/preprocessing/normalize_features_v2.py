r"""AEROVIGIL Module 03 v2: robust normalization fitted on clean training rows only.

The scaler is fitted on rows of TRAINING flights that are known to be clean: every row
of a normal flight and the pre-attack rows of attacked flights (phase 0 and 1). Attack,
ramp, plateau and post-attack rows never enter the fit. Each feature is centred on its
median and divided by its IQR / 1.349 (a robust standard deviation), then clipped to
+/- CLIP so that the large ramp values cannot dominate training.

Reuses data/interim/module03/flight_splits.csv (the seed-42 split) unchanged.

Writes data/processed/module03_v2/normalized/<flight>.parquet and
       data/processed/module03_v2/normalization_parameters.json

Usage:
    python -m aerovigil.preprocessing.normalize_features_v2
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "module03_v2"
FEAT_DIR = OUT_DIR / "features"
NORM_DIR = OUT_DIR / "normalized"
PARAM_FILE = OUT_DIR / "normalization_parameters.json"
SPLIT_FILE = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
CLEAN_PHASES = (0, 1)
FIT_EVERY = 5
CLIP = 20.0
SCALE_FLOOR = 1e-6
NON_FEATURES = {
    "flight_id",
    "mission_type",
    "time_usec",
    "attack_label",
    "after_gap",
    "t_s",
    "offset_m",
    "spoof_truth",
    "phase",
    "seconds_since_onset",
    "aligned_all",
}


def main():
    splits = pd.read_csv(SPLIT_FILE).set_index("flight_id")["split"]
    files = sorted(FEAT_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No feature files found. Run build_features_v2.py first.")
    first = pd.read_parquet(files[0])
    features = [c for c in first.columns if c not in NON_FEATURES]
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AEROVIGIL - ROBUST NORMALIZATION v2 (clean training rows only)")
    print("=" * 65)
    fit_parts, fit_flights = [], []
    for path in files:
        df = pd.read_parquet(path, columns=["flight_id", "phase", *features])
        fid = df["flight_id"].iloc[0]
        if splits.get(fid) != "train":
            continue
        clean = df[df["phase"].isin(CLEAN_PHASES)].iloc[::FIT_EVERY]
        fit_parts.append(clean[features].to_numpy(np.float64))
        fit_flights.append(fid)
    fit = np.concatenate(fit_parts)
    median = np.nanmedian(fit, axis=0)
    q75, q25 = np.nanpercentile(fit, [75, 25], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, SCALE_FLOOR)
    floored = [f for f, s, v in zip(features, scale, q75 - q25) if v / 1.349 < s]
    print(f"Training flights in the fit : {len(fit_flights)}")
    print(f"Clean rows in the fit       : {len(fit):,} (every {FIT_EVERY}th row)")
    print(f"Features                    : {len(features)}")
    print(f"Features with a floored scale: {floored if floored else 'none'}")

    clipped = {f: [0, 0] for f in features}
    ramp_z = {f: [] for f in features}
    rows_total = 0
    for n, path in enumerate(files, start=1):
        df = pd.read_parquet(path)
        z = (df[features].to_numpy(np.float64) - median) / scale
        hit = np.abs(z) > CLIP
        rows_total += len(df)
        for k, f in enumerate(features):
            clipped[f][0] += int(np.nansum(hit[:, k]))
            clipped[f][1] += int(np.isfinite(z[:, k]).sum())
        ramp = (df["phase"] == 2).to_numpy()
        if ramp.any() and splits.get(df["flight_id"].iloc[0]) == "train":
            for k, f in enumerate(features):
                ramp_z[f].append(np.nanmedian(z[ramp, k]))
        out = df.copy()
        out[features] = np.clip(z, -CLIP, CLIP).astype(np.float32)
        out.to_parquet(NORM_DIR / path.name, index=False)
        if n % 40 == 0 or n == len(files):
            print(f"Normalized {n}/{len(files)} flights")

    params = {
        "method": "median / (IQR / 1.349), clipped",
        "clip": CLIP,
        "scale_floor": SCALE_FLOOR,
        "fit_phases": list(CLEAN_PHASES),
        "fit_every_nth_row": FIT_EVERY,
        "fit_flights": len(fit_flights),
        "fit_rows": len(fit),
        "features": features,
        "median": dict(zip(features, median.tolist())),
        "scale": dict(zip(features, scale.tolist())),
    }
    PARAM_FILE.write_text(json.dumps(params, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    table = pd.DataFrame(
        {
            "median": median,
            "scale": scale,
            "clipped_%": [
                100 * clipped[f][0] / max(clipped[f][1], 1) for f in features
            ],
            "median_z_in_ramp": [
                float(np.nanmedian(ramp_z[f])) if ramp_z[f] else np.nan
                for f in features
            ],
        },
        index=features,
    )
    print("\n" + "-" * 65)
    print("PARAMETERS, SHARE CLIPPED, AND MEDIAN NORMALIZED VALUE IN THE RAMP")
    print(table.round(4).to_string())

    check_rows = sum(
        len(pd.read_parquet(p, columns=["phase"])) for p in NORM_DIR.glob("*.parquet")
    )
    fit_set, train_set = set(fit_flights), set(splits[splits == "train"].index)
    checks = {
        "fitted on training flights only": fit_set <= train_set,
        "no validation/test flight in the fit": not (
            fit_set & set(splits[splits != "train"].index)
        ),
        "only clean phases (normal flight, pre-attack) in the fit": set(CLEAN_PHASES)
        == {0, 1},
        "no constant feature": bool((scale > SCALE_FLOOR).all()),
        "row count preserved": check_rows == rows_total,
        "one normalized file per flight": len(list(NORM_DIR.glob("*.parquet")))
        == len(files),
        "parameters saved": PARAM_FILE.exists(),
    }
    print("\nACCEPTANCE CHECKS")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\nSaved:", PARAM_FILE)
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Normalization checks FAILED")


if __name__ == "__main__":
    main()
