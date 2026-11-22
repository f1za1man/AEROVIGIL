import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "module03"
FEAT_DIR = OUT_DIR / "features"
NORM_DIR = OUT_DIR / "normalized"
PARAM_FILE = OUT_DIR / "normalization_parameters.json"
SPLIT_FILE = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"

ID_COLUMNS = ["flight_id", "mission_type", "time_usec",
              "attack_label", "after_gap"]


class RunningStats:
    """Exact running mean/variance per column, NaN ignored (Chan's method)."""

    def __init__(self, k):
        self.n = np.zeros(k)
        self.mean = np.zeros(k)
        self.m2 = np.zeros(k)

    def update(self, x):
        x = np.asarray(x, dtype="float64")
        valid = ~np.isnan(x)
        nb = valid.sum(axis=0).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            mb = np.where(nb > 0, np.nansum(x, axis=0) / nb, 0.0)
            m2b = np.where(nb > 0,
                           np.nansum((x - mb) ** 2, axis=0), 0.0)
            tot = self.n + nb
            delta = mb - self.mean
            frac = np.where(tot > 0, nb / tot, 0.0)
            self.mean = self.mean + delta * frac
            self.m2 = self.m2 + m2b + np.where(
                tot > 0, delta ** 2 * self.n * nb / tot, 0.0)
        self.n = tot

    def std(self):
        return np.sqrt(self.m2 / np.maximum(self.n, 1))


def main():
    splits = pd.read_csv(SPLIT_FILE).set_index("flight_id")["split"]
    files = sorted(FEAT_DIR.glob("*.parquet"))
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    first = pd.read_parquet(files[0])
    features = [c for c in first.columns if c not in ID_COLUMNS]
    k = len(features)

    print("=" * 65)
    print("AEROVIGIL - TRAINING-ONLY NORMALIZATION")
    print("=" * 65)
    print(f"Feature files: {len(files)}   Features: {k}\n")

    # ---- Pass 1: fit on training flights only --------------------------
    fit = RunningStats(k)
    train_ids = []
    for path in files:
        g = pd.read_parquet(path)
        fid = g["flight_id"].iloc[0]
        if splits[fid] == "train":
            fit.update(g[features].to_numpy())
            train_ids.append(fid)

    mean, std = fit.mean, fit.std()
    constant = std < 1e-12
    safe_std = np.where(constant, 1.0, std)

    params = {
        "method": "standardization z = (x - mean) / std",
        "std_type": "population (ddof=0), NaN ignored",
        "fitted_on_split": "train",
        "n_training_flights": len(train_ids),
        "training_flight_ids": sorted(train_ids),
        "feature_order": features,
        "mean": dict(zip(features, map(float, mean))),
        "std": dict(zip(features, map(float, safe_std))),
        "valid_sample_count": dict(zip(features, map(int, fit.n))),
        "constant_features": [f for f, c in zip(features, constant) if c],
    }
    with PARAM_FILE.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # ---- Pass 2: apply same parameters to every flight ------------------
    check = {s: RunningStats(k) for s in ("train", "validation", "test")}
    rows = 0
    for path in files:
        g = pd.read_parquet(path)
        fid = g["flight_id"].iloc[0]
        x = (g[features].to_numpy(dtype="float64") - mean) / safe_std
        check[splits[fid]].update(x)
        out = g[ID_COLUMNS].copy()
        for i, name in enumerate(features):
            out[name] = x[:, i].astype("float32")
        out.to_parquet(NORM_DIR / path.name, index=False)
        rows += len(out)

    # ---- Report ----------------------------------------------------------
    table = pd.DataFrame({"train_mean": mean, "train_std": std},
                         index=features)
    print("TRAINING PARAMETERS (physical units)")
    print(table.round(5).to_string())

    print("\nNORMALIZED MEAN / STD BY PARTITION")
    summary = pd.DataFrame({
        "train_mean": check["train"].mean,
        "train_std": check["train"].std(),
        "val_mean": check["validation"].mean,
        "val_std": check["validation"].std(),
        "test_mean": check["test"].mean,
        "test_std": check["test"].std(),
    }, index=features)
    print(summary.round(3).to_string())

    train_ok = (np.allclose(check["train"].mean, 0, atol=1e-6)
                and np.allclose(check["train"].std()[~constant], 1, atol=1e-6))
    checks = {
        "Fitted on training flights only (252)": len(train_ids) == 252,
        "No validation/test flight in fit":
            all(splits[f] == "train" for f in train_ids),
        "Training data normalizes to mean 0, std 1": bool(train_ok),
        "No constant features": not constant.any(),
        "Row count preserved (7,893,049)": rows == 7_893_049,
        "One normalized file per flight":
            len(list(NORM_DIR.glob("*.parquet"))) == len(files),
        "Feature order saved": len(params["feature_order"]) == k,
    }
    print("\nACCEPTANCE CHECKS")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nSaved: {PARAM_FILE}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Normalization checks FAILED")


if __name__ == "__main__":
    main()