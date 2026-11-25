r"""AEROVIGIL Module 07 v2 step 1: build the window sequences the LSTMs train on.

Reads the normalized Module 03 v2 flights and the normalized Module 04 v2 attacks, and
writes one binned sequence per window with the measured phase attached.

Original flights contribute every window, because the alarm timeline of a flight needs
them all. Synthetic flights of the main set contribute only windows that overlap the
attack: their clean stretches are copies of a source flight that is already in the
original table, and counting them again is the repeated-negatives mistake. Sweep
flights contribute every window, because the Stealth Frontier needs their whole
timeline to measure a detection delay.

Normal flights are split into four roles, not three. A third of the training normals
become `calib`: they are never trained on, and Module 07 v2 sets its alarm threshold on
them together with the validation normals. Module 07 v1 calibrated on 21 validation
flights alone, which is 0.7 h of normal flight, so a single outlier window set the
threshold and the false-alarm rate on test came out at 6.1 per hour against a budget
of 1. A synthetic attack built on a calib source flight is also marked calib, so no
model sees any part of a calibration flight.

Writes data/processed/module07_v2/X_orig.npy, meta_orig.parquet,
       X_syn.npy, meta_syn.parquet and feature_names.csv

Usage:
    python src\aerovigil\lstm\build_sequences_v2.py
    python src\aerovigil\lstm\build_sequences_v2.py --limit 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.lstm.sequences_v2 import (
    BIN_S,
    T_STEPS,
    flight_sequences,
)
from aerovigil.preprocessing.normalize_features_v2 import NON_FEATURES
from aerovigil.windowing.make_windows_v2 import SECONDS, STEP_S

ROOT = Path(__file__).resolve().parents[3]
M03_NORM = ROOT / "data" / "processed" / "module03_v2" / "normalized"
M04_NORM = ROOT / "data" / "processed" / "module04_v2" / "normalized"
BASE_REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
MANIFEST = ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"
SPLITS = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
OUT_DIR = ROOT / "data" / "processed" / "module07_v2"
CALIB_EVERY = 2


def roles_for(rep, splits):
    """Role of every original flight: train, calib, validation or test.

    Every second training normal flight, in flight_id order, becomes a calibration
    flight: about 50 flights, which is 2 h of normal time, against the 0.7 h that
    Module 07 v1 calibrated on. The choice is deterministic, so a re-run reproduces it.
    """
    role = {fid: splits[fid] for fid in rep["flight_id"]}
    normals = sorted(rep.loc[rep["attacked"] == 0, "flight_id"])
    train_normals = [f for f in normals if splits[f] == "train"]
    for f in train_normals[::CALIB_EVERY]:
        role[f] = "calib"
    return role


def orig_jobs(rep, splits, role, limit):
    if limit:
        rep = rep.groupby("mission", group_keys=False).head(limit)
    jobs = []
    for _, r in rep.iterrows():
        fid = r["flight_id"]
        jobs.append({
            "path": M03_NORM / r["file"], "flight_id": fid, "source_flight_id": fid,
            "origin": "orig", "attack_set": "orig", "split": splits[fid],
            "role": role[fid], "mission_type": r["mission"],
            "attack_type": "original" if r["attacked"] else "none",
            "velocity_mode": "na", "drift_rate_mps": np.nan,
        })  # fmt: skip
    return jobs


def syn_jobs(role, limit):
    man = pd.read_csv(MANIFEST)
    if limit:
        key = ["attack_set", "attack_type", "velocity_mode"]
        man = man.groupby(key, group_keys=False).head(limit)
    jobs = []
    for _, r in man.iterrows():
        src = r["source_flight_id"]
        jobs.append({
            "path": M04_NORM / r["file"], "flight_id": r["attacked_flight_id"],
            "source_flight_id": src, "origin": "syn", "attack_set": r["attack_set"],
            "split": r["split"], "role": role.get(src, r["split"]),
            "mission_type": r["mission_type"], "attack_type": r["attack_type"],
            "velocity_mode": r["velocity_mode"],
            "drift_rate_mps": float(r.get("drift_rate_mps", np.nan)),
        })  # fmt: skip
    return jobs


def build(jobs, features, overlap_only_sets=("main",)):
    """Sequences and metadata for a list of flights."""
    parts, metas, dropped = [], [], {}
    for n, info in enumerate(jobs, start=1):
        path = info.pop("path")
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Build the Module 03/04 v2 features first."
            )
        g = pd.read_parquet(path)
        if features is None:
            features = [c for c in g.columns if c not in NON_FEATURES]
        overlap = info["attack_set"] in overlap_only_sets
        x, m, d = flight_sequences(g, features, info, overlap)
        for k, v in d.items():
            dropped[k] = dropped.get(k, 0) + v
        if len(x):
            parts.append(x)
            metas.append(m)
        if n % 100 == 0 or n == len(jobs):
            print(f"  {n}/{len(jobs)} flights")
    if not parts:
        raise SystemExit("No windows survived the drop rules.")
    return np.concatenate(parts), pd.concat(metas, ignore_index=True), dropped, features


def run(sources=("orig", "syn"), limit=None, output_dir=None):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(BASE_REPORT).sort_values("flight_id")
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    role = roles_for(rep, splits)
    print("=" * 70)
    print("AEROVIGIL - MODULE 07 v2 SEQUENCES (v2 features, measured phases)")
    print("=" * 70)
    print(f"Window {SECONDS:g} s, hop {STEP_S:g} s, bin {BIN_S:g} s -> {T_STEPS} steps")
    started, features, counts = time.time(), None, {}
    for name in sources:
        jobs = (
            orig_jobs(rep, splits, role, limit)
            if name == "orig"
            else syn_jobs(role, limit)
        )
        print(f"\n{name}: {len(jobs)} flights")
        x, meta, dropped, features = build(jobs, features)
        np.save(out / f"X_{name}.npy", x)
        meta.to_parquet(out / f"meta_{name}.parquet", index=False)
        counts[name] = (x, meta)
        print(f"  windows {len(x):,}   shape {x.shape}   dropped {dropped}")
    pd.Series(features, name="feature").to_csv(out / "feature_names.csv", index=False)
    report(counts, features, role, out, time.time() - started, bool(limit))
    return counts


def report(counts, features, role, out, elapsed, partial=False):
    pd.set_option("display.width", 200)
    orig = counts["orig"][1]
    print(f"\nFeatures: {len(features)}")
    print("\nORIGINAL WINDOWS: role x window class")
    print(pd.crosstab(orig["role"], orig["window_class"], margins=True).to_string())
    print(
        "\nORIGINAL WINDOWS: the two labels compared (y_window rows, y_event columns)"
    )
    print(pd.crosstab(orig["y_window"], orig["y_event"], margins=True).to_string())
    n_calib = sum(1 for v in role.values() if v == "calib")
    clean = orig[(orig["y_event"] == 0) & (orig["attack_type"] == "none")]
    pool = clean[clean["role"] == "calib"]
    val = clean[clean["role"] == "validation"]
    print(f"\nCalibration flights: {n_calib}   clean calibration windows {len(pool):,} "
          f"({len(pool) * STEP_S / 3600.0:.2f} h of normal flight; Module 07 v1 used the "
          f"{len(val):,} validation windows, {len(val) * STEP_S / 3600.0:.2f} h)")  # fmt: skip
    if "syn" in counts:
        syn = counts["syn"][1]
        print("\nSYNTHETIC EVENT WINDOWS: attack type x velocity mode (main set)")
        hit = syn[(syn["y_event"] == 1) & (syn["attack_set"] == "main")]
        print(pd.crosstab(hit["attack_type"], hit["velocity_mode"]).to_string())
        sw = syn[syn["attack_set"] == "sweep"]
        if len(sw):
            print("\nSWEEP WINDOWS per drift rate")
            print(sw.groupby("drift_rate_mps").size().to_string())
    checks = {
        "every role has clean windows": clean["role"].nunique() == 4,
        "calibration pool beats the v1 one": len(pool) > len(val),
        "no normal flight has an event window": not orig[orig["y_event"] == 1][
            "attack_type"
        ]
        .eq("none")
        .any(),
        "no NaN left in the sequences": np.isfinite(counts["orig"][0]).all(),
        "sequences are the right shape": counts["orig"][0].shape[1:]
        == (
            T_STEPS,
            len(features),
        ),
    }
    print(f"\nElapsed {elapsed:.0f} s   saved to {out}\n\nACCEPTANCE CHECKS")
    if partial:
        print("  skipped: --limit was used, so a role can legitimately be empty")
        return
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("Module 07 v2 sequence checks FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="orig,syn", help="orig or orig,syn")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(tuple(s.strip() for s in a.sources.split(",")), a.limit, a.output_dir)


if __name__ == "__main__":
    main()
