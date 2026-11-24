r"""AEROVIGIL Module 05 v2: window summary tables on the Module 03 v2 features.

Same window engine as Module 05 (windows.py, 5 s windows on a 2.5 s hop, six summary
statistics per feature), but on the 36 normalized v2 features instead of the 22 v1
ones, and with the measured phase of each window attached. That lets the re-scoring
compare three things on identical windows:

  * the Module 05 label, attack_fraction >= 0.5, which counts the 256 s curved
    plateau as 102 separate attack windows (column y_window);
  * the event label, ramp or snapback against clean, which is what the features can
    actually see (column y_event, with plateau and post windows marked -1);
  * the feature set, since the IMU columns can be dropped to reproduce Module 05.

Original flights contribute every window. Synthetic flights contribute only windows
that overlap the attack, because their clean stretches are copies of a source flight
that is already in the original table.

Writes data/processed/module05_v2/windows_orig.parquet,
       data/processed/module05_v2/windows_syn.parquet (with --sources orig,syn) and
       data/processed/module05_v2/feature_names.csv

Usage:
    python src\aerovigil\baselines\build_window_tables_v2.py
    python src\aerovigil\baselines\build_window_tables_v2.py --sources orig,syn
    python src\aerovigil\baselines\build_window_tables_v2.py --limit 5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.baselines.windows import stat_names, summarise_flight
from aerovigil.preprocessing.normalize_features_v2 import NON_FEATURES
from aerovigil.windowing.make_windows_v2 import SECONDS, STEP_S, window_class

ROOT = Path(__file__).resolve().parents[3]
M03_NORM = ROOT / "data" / "processed" / "module03_v2" / "normalized"
M04_NORM = ROOT / "data" / "processed" / "module04_v2" / "normalized"
BASE_REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
MANIFEST = ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"
SPLITS = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
OUT_DIR = ROOT / "data" / "processed" / "module05_v2"
POSITIVE_MIN_FRACTION = 0.5
EVENT = ("ramp", "snapback")
CLEAN = ("normal", "pre")
IMU_PREFIX = ("res_accel", "res_course")


def imu_columns(features):
    return [f for f in features if f.startswith(IMU_PREFIX)]


def flight_windows(g, features, info, attack_overlap_only=False):
    """Summary statistics plus the measured phase of every kept window."""
    t = g["t_s"].to_numpy(float)
    stats, meta, dropped = summarise_flight(
        g[features].to_numpy(float),
        t,
        g["attack_label"].to_numpy(float),
        g["after_gap"].to_numpy(bool),
        SECONDS,
        STEP_S,
    )
    if not meta:
        return None, dropped
    m = pd.DataFrame(meta)
    i1 = m["last_row"].to_numpy() + 1
    i0 = i1 - m["n_rows"].to_numpy()
    ph = g["phase"].to_numpy()
    truth = g["spoof_truth"].to_numpy()
    offset = g["offset_m"].to_numpy(float)
    since = g["seconds_since_onset"].to_numpy(float)
    m["window_class"] = [window_class(ph[a:b]) for a, b in zip(i0, i1)]
    m["spoof_fraction"] = [float((truth[a:b] == 1).mean()) for a, b in zip(i0, i1)]
    m["max_offset_m"] = [float(np.nanmax(offset[a:b])) for a, b in zip(i0, i1)]
    m["seconds_since_onset_end"] = since[i1 - 1]
    m["y_window"] = np.where(
        m["attack_fraction"] >= POSITIVE_MIN_FRACTION,
        1,
        np.where(m["attack_fraction"] == 0.0, 0, -1),
    )
    m["y_event"] = np.where(
        m["window_class"].isin(EVENT), 1, np.where(m["window_class"].isin(CLEAN), 0, -1)
    )
    for k, v in info.items():
        m[k] = v
    m = m.drop(columns=["last_row", "n_rows"])
    cols = stat_names(features)
    frame = pd.concat(
        [m.reset_index(drop=True), pd.DataFrame(stats.astype("float32"), columns=cols)],
        axis=1,
    )
    if attack_overlap_only:
        frame = frame[frame["attack_fraction"] > 0].reset_index(drop=True)
    return frame, dropped


def orig_jobs(limit, splits):
    rep = pd.read_csv(BASE_REPORT).sort_values("flight_id")
    if limit:
        rep = rep.groupby("mission", group_keys=False).head(limit)
    jobs = []
    for _, r in rep.iterrows():
        fid = r["flight_id"]
        jobs.append({
            "path": M03_NORM / r["file"], "flight_id": fid, "source_flight_id": fid,
            "origin": "orig", "split": splits[fid], "mission_type": r["mission"],
            "attack_type": "original" if r["attacked"] else "none",
            "velocity_mode": "na", "attack_set": "orig",
        })  # fmt: skip
    return jobs


def syn_jobs(limit):
    man = pd.read_csv(MANIFEST)
    if limit:
        key = ["attack_type", "velocity_mode"]
        man = man.groupby(key, group_keys=False).head(limit)
    jobs = []
    for _, r in man.iterrows():
        jobs.append({
            "path": M04_NORM / r["file"], "flight_id": r["attacked_flight_id"],
            "source_flight_id": r["source_flight_id"], "origin": "syn",
            "split": r["split"], "mission_type": r["mission_type"],
            "attack_type": r["attack_type"], "velocity_mode": r["velocity_mode"],
            "attack_set": r["attack_set"],
        })  # fmt: skip
    return jobs


def read_sources(sources, limit):
    """Flights to summarise, and whether to keep attack-overlapping windows only."""
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    jobs = {}
    if "orig" in sources:
        jobs["orig"] = (orig_jobs(limit, splits), False)
    if "syn" in sources:
        jobs["syn"] = (syn_jobs(limit), True)
    return jobs


def run(sources=("orig",), limit=None, output_dir=None):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    jobs = read_sources(sources, limit)
    if not jobs:
        raise SystemExit("No sources selected (use --sources orig or orig,syn)")
    print("=" * 70)
    print("AEROVIGIL - MODULE 05 v2 WINDOW TABLES (v2 features, measured phases)")
    print("=" * 70)
    print(f"Window {SECONDS:g} s, hop {STEP_S:g} s")
    features, started, tables = None, time.time(), {}
    for name, (rows, overlap_only) in jobs.items():
        parts, dropped_all = [], {}
        print(f"\n{name}: {len(rows)} flights")
        for n, info in enumerate(rows, start=1):
            path = info.pop("path")
            if not path.is_file():
                raise SystemExit(
                    f"Missing {path}. Build the Module 03/04 v2 features first."
                )
            g = pd.read_parquet(path)
            if features is None:
                features = [c for c in g.columns if c not in NON_FEATURES]
                print(f"Features: {len(features)} ({len(imu_columns(features))} IMU)")
            w, d = flight_windows(g, features, info, overlap_only)
            for k, v in d.items():
                dropped_all[k] = dropped_all.get(k, 0) + v
            if w is not None:
                parts.append(w)
            if n % 100 == 0 or n == len(rows):
                print(f"  {name} {n}/{len(rows)}  ({time.time() - started:.0f} s)")
        tables[name] = pd.concat(parts, ignore_index=True)
        tables[name].to_parquet(out / f"windows_{name}.parquet", index=False)
        print(f"  dropped windows: {dropped_all}")
    pd.Series(features, name="feature").to_csv(out / "feature_names.csv", index=False)
    report(tables, features, out, time.time() - started)
    return tables


def report(tables, features, out, elapsed):
    pd.set_option("display.width", 200)
    orig = tables["orig"]
    print("\nWindows: " + "  ".join(f"{k} {len(v):,}" for k, v in tables.items()))
    print(f"Summary columns per window: {len(features)} x 6 = {len(features) * 6}")
    print("\nORIGINAL WINDOWS: mission type x window class")
    print(
        pd.crosstab(
            orig["mission_type"], orig["window_class"], margins=True
        ).to_string()
    )
    print(
        "\nORIGINAL WINDOWS: the two labels compared (y_window rows, y_event columns)"
    )
    print(pd.crosstab(orig["y_window"], orig["y_event"], margins=True).to_string())
    print("\nThe cell y_window=1, y_event=-1 is the plateau that Module 05 asked the")
    print("models to classify; y_window=0, y_event=-1 is the post-attack tail.")
    ev = orig[orig["y_event"] == 1]
    print("\nEVENT WINDOWS PER SPLIT (ramp and snapback of the real attacks)")
    print(pd.crosstab(ev["split"], ev["window_class"], margins=True).to_string())
    if "syn" in tables:
        syn = tables["syn"]
        print("\nSYNTHETIC EVENT WINDOWS: attack type x velocity mode")
        hit = syn[syn["y_event"] == 1]
        print(pd.crosstab(hit["attack_type"], hit["velocity_mode"]).to_string())
    checks = {
        "every split has event windows": ev["split"].nunique() == 3,
        "every split has clean windows": (
            orig[orig["y_event"] == 0]["split"].nunique() == 3
        ),
        "no normal flight has an event window": not ev["attack_type"].eq("none").any(),
        "no infinite summary values": not np.isinf(
            orig[stat_names(features)].to_numpy()
        ).any(),
    }  # fmt: skip
    print(f"\nElapsed {elapsed:.0f} s   saved to {out}\n\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("Module 05 v2 window checks FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="orig", help="orig or orig,syn")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(tuple(s.strip() for s in a.sources.split(",")), a.limit, a.output_dir)


if __name__ == "__main__":
    main()
