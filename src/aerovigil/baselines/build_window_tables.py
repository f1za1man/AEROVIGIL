"""Build the window summary tables used by the Module 05 baselines.

Original flights: read the Module 03 feature files.
Synthetic flights: run the SAME Module 03 feature builder on the attacked flights
(Module 04), then keep only the windows that overlap the attack.

Usage:
    python src\\aerovigil\\baselines\\build_window_tables.py
    python src\\aerovigil\\baselines\\build_window_tables.py --limit 3     (smoke test)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    UNIT_TO_SECONDS,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)
from aerovigil.baselines.windows import (
    label_window,
    stat_names,  # noqa: E402
    summarise_flight,
)
from aerovigil.features.build_features import build_flight  # noqa: E402

META_COLS = ["flight_id", "mission_type", "time_usec", "attack_label", "after_gap"]


def feature_columns(frame):
    return [c for c in frame.columns if c not in META_COLS]


def flight_windows(
    feats, t_s, labels, after_gap, info, win_s, hop_s, wcfg, rel_start=None
):
    stats, meta, dropped = summarise_flight(
        feats.to_numpy(float),
        t_s,
        labels,
        after_gap,
        win_s,
        hop_s,
        wcfg["min_valid_fraction"],
    )
    if not meta:
        return None, dropped
    m = pd.DataFrame(meta)
    lab = [
        label_window(f, wcfg["label"]["positive_min_fraction"])
        for f in m["attack_fraction"]
    ]
    m["y"] = [x[0] for x in lab]
    m["ambiguous"] = [x[1] for x in lab]
    if rel_start is not None:
        m["window_end_rel_attack_start_s"] = rel_start[m["last_row"].to_numpy()]
    else:
        m["window_end_rel_attack_start_s"] = np.nan
    for k, v in info.items():
        m[k] = v
    m = m.drop(columns="last_row")
    stat_df = pd.DataFrame(
        stats.astype("float32"), columns=stat_names(list(feats.columns))
    )
    return pd.concat([m.reset_index(drop=True), stat_df], axis=1), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(ROOT / "configs" / "baselines" / "baselines.yaml")
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="flights per origin (smoke test)"
    )
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    win_s = float(pre["processing"]["window_duration_seconds"])
    hop_s = win_s * (1.0 - float(pre["processing"]["window_overlap"]))
    scale = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
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
    wcfg = cfg["windows"]

    out_dir = (
        Path(a.output_dir) if a.output_dir else resolve(cfg["paths"]["output_dir"])
    )
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    man = man[man["attack_set"] == wcfg["synthetic_attack_set"]]

    if a.limit:
        report = report.groupby("mission_type", group_keys=False).head(a.limit)
        man = man.groupby("attack_type", group_keys=False).head(a.limit)

    print("=" * 65)
    print("AEROVIGIL - WINDOW TABLES")
    print("=" * 65)
    print(
        f"Window {win_s:g} s, hop {hop_s:g} s | original flights {len(report)} | "
        f"synthetic flights {len(man)}"
    )
    started = time.time()
    dropped_total = {}

    def add_dropped(d):
        for k, v in d.items():
            dropped_total[k] = dropped_total.get(k, 0) + v

    # ---------------- original flights ----------------
    orig_parts, feature_names = [], None
    for n, (_, r) in enumerate(report.iterrows(), start=1):
        f = pd.read_parquet(resolve(cfg["paths"]["features_dir"]) / r["file"])
        feats = f[feature_columns(f)]
        if feature_names is None:
            feature_names = list(feats.columns)
            if len(feature_names) != cfg["expected_feature_count"]:
                raise SystemExit(
                    f"Expected {cfg['expected_feature_count']} features, "
                    f"found {len(feature_names)}: {feature_names}"
                )
        t_s = (f["time_usec"].to_numpy("int64") - f["time_usec"].iloc[0]) * scale
        fid = str(r["flight_id"])
        info = {
            "flight_id": fid,
            "source_flight_id": fid,
            "origin": "orig",
            "split": split_map[fid],
            "mission_type": r["mission_type"],
            "attack_type": "original" if r["attack_rows"] > 0 else "none",
            "velocity_mode": "na",
        }
        w, d = flight_windows(
            feats,
            t_s,
            f["attack_label"].to_numpy(),
            f["after_gap"].to_numpy(),
            info,
            win_s,
            hop_s,
            wcfg,
        )
        add_dropped(d)
        if w is not None:
            orig_parts.append(w)
        if n % 60 == 0 or n == len(report):
            print(f"Original flights {n}/{len(report)}")
    orig = pd.concat(orig_parts, ignore_index=True)

    # ---------------- synthetic flights ----------------
    syn_parts, kept_stats = [], {"windows_before_filter": 0, "windows_kept": 0}
    for n, (_, m) in enumerate(man.iterrows(), start=1):
        g = pd.read_parquet(resolve(cfg["paths"]["attacked_dir"]) / m["file"])
        f = build_flight(g, c, scale)  # the Module 03 feature builder
        feats = f[feature_columns(f)]
        t_s = (f["time_usec"].to_numpy("int64") - f["time_usec"].iloc[0]) * scale
        info = {
            "flight_id": m["attacked_flight_id"],
            "source_flight_id": m["source_flight_id"],
            "origin": "syn",
            "split": m["split"],
            "mission_type": m["mission_type"],
            "attack_type": m["attack_type"],
            "velocity_mode": m["velocity_mode"],
        }
        w, d = flight_windows(
            feats,
            t_s,
            f["attack_label"].to_numpy(),
            f["after_gap"].to_numpy(),
            info,
            win_s,
            hop_s,
            wcfg,
            rel_start=g["time_from_attack_start_s"].to_numpy(),
        )
        add_dropped(d)
        if w is not None:
            kept_stats["windows_before_filter"] += len(w)
            w = w[w["attack_fraction"] > 0]
            kept_stats["windows_kept"] += len(w)
            syn_parts.append(w)
        if n % 60 == 0 or n == len(man):
            print(f"Synthetic flights {n}/{len(man)}  ({time.time() - started:.0f} s)")
    syn = pd.concat(syn_parts, ignore_index=True)

    orig.to_parquet(out_dir / "windows_orig.parquet", index=False)
    syn.to_parquet(out_dir / "windows_syn.parquet", index=False)
    pd.Series(feature_names, name="feature").to_csv(
        out_dir / "feature_names.csv", index=False
    )

    print("\n" + "-" * 65)
    print(
        f"Original windows : {len(orig):,}   Synthetic (attack-overlapping) windows: {len(syn):,}"
    )
    print(
        f"Synthetic windows before the attack-overlap filter: {kept_stats['windows_before_filter']:,}"
    )
    print(f"Dropped windows (all flights): {dropped_total}")
    print(
        f"Summary features per window  : {len(feature_names)} x 6 = {len(feature_names) * 6}"
    )
    for name, df in (("ORIGINAL", orig), ("SYNTHETIC", syn)):
        print(f"\n{name} WINDOWS: split x label (1 attack, 0 normal, -1 ambiguous)")
        print(pd.crosstab(df["split"], df["y"], margins=True).to_string())
    print("\nSYNTHETIC ATTACK WINDOWS (y=1): attack type x velocity mode")
    print(
        pd.crosstab(
            syn.loc[syn.y == 1, "attack_type"], syn.loc[syn.y == 1, "velocity_mode"]
        ).to_string()
    )
    print("\nORIGINAL WINDOWS: mission type x label")
    print(pd.crosstab(orig["mission_type"], orig["y"]).to_string())
    stat_cols = [c for c in orig.columns if "__" in c]
    nan_frac = float(orig[stat_cols].isna().to_numpy().mean())
    inf_cnt = int(np.isinf(orig[stat_cols].to_numpy()).sum())
    print(
        f"\nNaN share of summary values (original): {nan_frac:.4%}   infinite values: {inf_cnt}"
    )
    print(f"Elapsed: {time.time() - started:.0f} s")
    print(f"Saved to: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
