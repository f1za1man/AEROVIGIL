"""Validate the Module 05 window tables.

Usage:
    python src\\aerovigil\\baselines\\validate_windows.py
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (UNIT_TO_SECONDS, load_split_map,  # noqa: E402
                                      load_yaml, resolve)
from aerovigil.baselines.datasets import stat_columns  # noqa: E402
from aerovigil.baselines.windows import STATS  # noqa: E402

FORBIDDEN = ("attack", "label", "flight", "source", "spoof", "time_from", "split", "mission")
N_RECOMPUTE = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "baselines" / "baselines.yaml"))
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    win_s = float(pre["processing"]["window_duration_seconds"])
    hop_s = win_s * (1.0 - float(pre["processing"]["window_overlap"]))
    scale = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    pos_min = cfg["windows"]["label"]["positive_min_fraction"]
    out_dir = Path(a.output_dir) if a.output_dir else resolve(cfg["paths"]["output_dir"])
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir

    orig = pd.read_parquet(out_dir / "windows_orig.parquet")
    syn = pd.read_parquet(out_dir / "windows_syn.parquet")
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    clean_ids = set(report.loc[report["attack_rows"] == 0, "flight_id"].astype(str))
    cols = stat_columns(orig)
    fails = defaultdict(list)

    print("=" * 65)
    print("AEROVIGIL - WINDOW TABLE VALIDATION")
    print("=" * 65)
    print(f"Original windows {len(orig):,} | synthetic windows {len(syn):,} | "
          f"summary features {len(cols)}")

    # -------- splits --------
    if not (orig["split"] == orig["flight_id"].map(split_map)).all():
        fails["original windows carry their flight's split"].append("orig")
    if (orig.groupby("flight_id")["split"].nunique() > 1).any():
        fails["no flight appears in two splits"].append("orig")
    msplit = dict(zip(man["attacked_flight_id"], man["split"]))
    if not (syn["split"] == syn["flight_id"].map(msplit)).all() or \
            not (syn["split"] == syn["source_flight_id"].map(split_map)).all():
        fails["synthetic windows inherit the source flight's split"].append("syn")
    if (syn.groupby("flight_id")["split"].nunique() > 1).any():
        fails["no flight appears in two splits"].append("syn")
    if not set(syn["source_flight_id"]).issubset(clean_ids):
        fails["synthetic sources are fully normal flights"].append("syn")
    if set(orig["flight_id"]) & set(syn["flight_id"]):
        fails["synthetic flight IDs differ from original IDs"].append("overlap")

    # -------- windows and labels --------
    both = pd.concat([orig, syn], ignore_index=True)
    if not np.allclose(both["window_end_s"] - both["window_start_s"], win_s):
        fails["every window is the configured length"].append("length")
    starts_ok = True
    for fid, g in both.groupby("flight_id"):
        d = np.diff(np.sort(g["window_start_s"].to_numpy()))
        if len(d) and not np.allclose(d / hop_s, np.round(d / hop_s), atol=1e-6):
            starts_ok = False
    if not starts_ok:
        fails["window starts lie on the hop grid"].append("grid")
    if both.duplicated(["flight_id", "window_start_s"]).any():
        fails["no duplicate windows"].append("dup")
    expect_y = np.where(both["attack_fraction"] == 0, 0,
                        np.where(both["attack_fraction"] >= pos_min, 1, -1))
    if not (both["y"].to_numpy() == expect_y).all() or \
            not (both["ambiguous"].to_numpy() == (expect_y == -1)).all():
        fails["label follows the configured rule"].append("labels")
    if not (syn["attack_fraction"] > 0).all():
        fails["synthetic windows all overlap the attack"].append("syn")
    pos = syn[syn["y"] == 1]
    if not (pos["window_end_rel_attack_start_s"] > 0).all():
        fails["positive synthetic windows end after the attack start"].append("rel")

    # -------- features --------
    bad_names = [c for c in cols if any(t in c.lower() for t in FORBIDDEN)]
    if bad_names:
        fails["no label/ID/ground-truth in feature names"].append(bad_names[:3])
    if len(cols) != cfg["expected_feature_count"] * len(STATS):
        fails["feature count"].append(len(cols))
    arr = both[cols].to_numpy(float)
    if np.isinf(arr).any():
        fails["no infinite values"].append("inf")
    nan_share = float(np.isnan(arr).mean())
    if nan_share > 0.01:
        fails["NaN share below 1%"].append(f"{nan_share:.3%}")

    # -------- usable data per split --------
    usable = both[~both["ambiguous"]]
    for sp in ("train", "validation", "test"):
        ys = set(usable.loc[usable["split"] == sp, "y"])
        if ys != {0, 1}:
            fails["every split has both classes"].append(sp)

    # -------- independent recomputation from the Module 03 feature files --------
    rng = np.random.default_rng(0)
    file_of = dict(zip(report["flight_id"].astype(str), report["file"]))
    idx = rng.choice(len(orig), min(N_RECOMPUTE, len(orig)), replace=False)
    worst = 0.0
    for i in idx:
        w = orig.iloc[i]
        f = pd.read_parquet(resolve(cfg["paths"]["features_dir"]) / file_of[w["flight_id"]])
        t = (f["time_usec"].to_numpy("int64") - f["time_usec"].iloc[0]) * scale
        a0, b0 = np.searchsorted(t, [w["window_start_s"], w["window_end_s"]], side="left")
        sl = f.iloc[a0:b0]
        if len(sl) != w["n_rows"] or sl["after_gap"].to_numpy()[1:].any():
            fails["recomputed window matches rows and has no gap"].append(w["flight_id"])
            continue
        if abs(sl["attack_label"].mean() - w["attack_fraction"]) > 1e-9:
            fails["recomputed attack fraction matches"].append(w["flight_id"])
        for feat in ("res_horizontal_velocity", "gps_speed_from_position", "res_altitude"):
            x = sl[feat].to_numpy(float)
            for stat, val in (("mean", np.nanmean(x)), ("max", np.nanmax(x)), ("min", np.nanmin(x)),
                              ("std", np.nanstd(x))):
                got = w[f"{feat}__{stat}"]
                worst = max(worst, abs(got - val) / (abs(val) + 1e-6))
    if worst > 1e-3:
        fails["recomputed statistics match the table (rel. error < 1e-3)"].append(f"{worst:.2e}")
    print(f"Largest relative error of {len(idx)} recomputed windows: {worst:.2e}")

    checks = [
        "original windows carry their flight's split",
        "synthetic windows inherit the source flight's split",
        "no flight appears in two splits",
        "synthetic sources are fully normal flights",
        "synthetic flight IDs differ from original IDs",
        "every window is the configured length",
        "window starts lie on the hop grid",
        "no duplicate windows",
        "label follows the configured rule",
        "synthetic windows all overlap the attack",
        "positive synthetic windows end after the attack start",
        "no label/ID/ground-truth in feature names",
        "feature count",
        "no infinite values",
        "NaN share below 1%",
        "every split has both classes",
        "recomputed window matches rows and has no gap",
        "recomputed attack fraction matches",
        "recomputed statistics match the table (rel. error < 1e-3)",
    ]
    print("\nVALIDATION RESULTS")
    for c in checks:
        ok = c not in fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {c}" + ("" if ok else f"  -> {fails[c][:3]}"))
    print("=" * 65)
    if fails:
        raise SystemExit("Window validation FAILED")


if __name__ == "__main__":
    main()