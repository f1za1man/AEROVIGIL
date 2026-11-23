"""Step 3: check everything the Module 05 baselines depend on."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    UNIT_TO_SECONDS,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)
from aerovigil.baselines.windows import summarise_flight  # noqa: E402
from aerovigil.features.build_features import build_flight  # noqa: E402

META = ["flight_id", "mission_type", "time_usec", "attack_label", "after_gap"]


def main():
    cfg = load_yaml(ROOT / "configs" / "baselines" / "baselines.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    win_s = float(pre["processing"]["window_duration_seconds"])
    hop_s = win_s * (1.0 - float(pre["processing"]["window_overlap"]))
    scale = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    problems = []

    print("=" * 65)
    print("AEROVIGIL - MODULE 05 INPUT CHECK")
    print("=" * 65)
    print(
        f"python packages : pandas {pd.__version__}, numpy {np.__version__}, "
        f"scikit-learn {sklearn.__version__}"
    )
    print(f"CPU threads     : {os.cpu_count()}")
    print(f"Window / hop    : {win_s:g} s / {hop_s:g} s (from preprocessing.yaml)")
    print(
        f"Training sources: {cfg['training_sources']} | balance train/val: "
        f"{cfg.get('balance_train_val', True)}"
    )

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    feat_dir = resolve(cfg["paths"]["features_dir"])
    missing = [f for f in report["file"] if not (feat_dir / f).exists()]
    print(f"\nFlights in cleaning report      : {len(report)}")
    print(
        f"Flights with a split            : {int(report['flight_id'].astype(str).isin(split_map).sum())}"
    )
    print(f"Module 03 feature files missing : {len(missing)}")
    if missing:
        problems.append("feature files missing")

    f = pd.read_parquet(feat_dir / report["file"].iloc[0])
    feats = [c for c in f.columns if c not in META]
    print(f"\nFeature file columns ({len(f.columns)}): {list(f.columns)}")
    print(
        f"Feature count                   : {len(feats)} (expected {cfg['expected_feature_count']})"
    )
    print(f"'after_gap' column present      : {'after_gap' in f.columns}")
    if len(feats) != cfg["expected_feature_count"] or "after_gap" not in f.columns:
        problems.append("feature file layout")

    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    main_set = man[man["attack_set"] == cfg["windows"]["synthetic_attack_set"]]
    att_dir = resolve(cfg["paths"]["attacked_dir"])
    miss_att = [x for x in main_set["file"] if not (att_dir / x).exists()]
    print(
        f"\nManifest rows ({cfg['windows']['synthetic_attack_set']} set)      : {len(main_set)}"
    )
    print(f"Attacked flight files missing   : {len(miss_att)}")
    if miss_att:
        problems.append("attacked files missing")

    # Window estimate and per-flight speed on a few real flights.
    n_windows, sample = 0, report.sample(min(5, len(report)), random_state=0)
    t0 = time.time()
    for _, r in sample.iterrows():
        g = pd.read_parquet(feat_dir / r["file"])
        t = (g["time_usec"].to_numpy("int64") - g["time_usec"].iloc[0]) * scale
        stats, meta, _ = summarise_flight(
            g[feats].to_numpy(float),
            t,
            g["attack_label"].to_numpy(),
            g["after_gap"].to_numpy(),
            win_s,
            hop_s,
        )
        n_windows += len(meta)
    per_flight = (time.time() - t0) / len(sample)
    est_total = n_windows / len(sample) * len(report)
    print(
        f"\nWindows per original flight (5 sampled): {n_windows / len(sample):.0f} "
        f"-> about {est_total:,.0f} original windows"
    )
    print(f"Summarising one original flight takes about {per_flight:.2f} s")

    # Speed of the Module 03 feature builder on attacked flights.
    gps = pre["sensors"]["gps"]
    c = {
        "flight": pre["dataset"]["flight_id_column"],
        "time": pre["dataset"]["timestamp_column"],
        "label": pre["labels"]["attack_column"],
        "lat": gps["latitude"],
        "lon": gps["longitude"],
        "alt": gps["altitude"],
        "vn": gps["velocity_north"],
        "ve": gps["velocity_east"],
        "vu": gps["velocity_up"],
        "baro": pre["sensors"]["barometer"]["altitude"],
    }
    t0 = time.time()
    for _, m in main_set.head(3).iterrows():
        build_flight(pd.read_parquet(att_dir / m["file"]), c, scale)
    per_att = (time.time() - t0) / 3
    print(
        f"Feature builder on one attacked flight : {per_att:.2f} s "
        f"-> about {per_att * len(main_set) / 60:.0f} min for {len(main_set)} flights"
    )

    labels = f["attack_label"].unique()
    print(f"\nLabel values in a feature file: {sorted(labels.tolist())}")
    print(
        "\nRESULT:",
        "ready to build window tables" if not problems else f"STOP - {problems}",
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
