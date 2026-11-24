r"""AEROVIGIL Module 04 v2: where the synthetic attacks are visible, per phase.

This replaces the Module 04 v1 feature check, which averaged over the whole attack
window and so concluded that a 20-500 m jump was invisible: the jump only touches
about 1 s of a 15-90 s window. Here every flight is cut into 5 s blocks, each block
is classed by its phase exactly as the Module 03 v2 windows are, and each feature
gets one number per block (max |z|). The negatives are the blocks of the 144 real
normal flights, used once, never the repeated clean rows of the attacked copies.

Reported per attack type, velocity mode and phase class:
  * AUC of each feature against the normal blocks;
  * the share of attacked flights with at least one block above the 99.9th
    percentile of the normal blocks, which is detection at a fixed false-alarm rate;
  * the same share by peak offset, and for the drift sweep by drift rate.

Needs features_v2_m04.py and the Module 03 v2 normalized flights.

Writes data/interim/module04_v2/attack_block_auc.csv

Usage:
    python src\aerovigil\attacks\attack_separability.py
    python src\aerovigil\attacks\attack_separability.py --limit 20
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.preprocessing.normalize_features_v2 import NON_FEATURES
from aerovigil.windowing.make_windows_v2 import SECONDS, window_class

ROOT = Path(__file__).resolve().parents[3]
M03_NORM = ROOT / "data" / "processed" / "module03_v2" / "normalized"
BASE_REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
M04_NORM = ROOT / "data" / "processed" / "module04_v2" / "normalized"
MANIFEST = ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"
AUC_FILE = ROOT / "data" / "interim" / "module04_v2" / "attack_block_auc.csv"
QUANTILE = 0.999
MIN_VALID = 0.5
CLASSES = ("ramp", "plateau", "snapback", "post")
BAND_NAMES = ["<2 m", "2-20 m", ">20 m"]
BAND_NAMES = ["<2 m", "2-20 m", ">20 m"]
# a block inside an IMU dropout is all-NaN on purpose; nanmax says so loudly
warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)


def blocks(path, features):
    """Per 5 s block: max |z| of every feature, its phase class and its flight."""
    g = pd.read_parquet(path, columns=["flight_id", "t_s", "phase", *features])
    t = g["t_s"].to_numpy(float)
    if t[-1] < SECONDS:
        return None, None
    z = np.abs(g[features].to_numpy(np.float64))
    ph = g["phase"].to_numpy()
    edges = np.arange(0, t[-1] + SECONDS, SECONDS)
    cut = np.searchsorted(t, edges)
    stats, classes = [], []
    for a, b in itertools.pairwise(cut):
        if b - a < 2:
            continue
        with np.errstate(invalid="ignore"):
            m = np.nanmax(z[a:b], axis=0)
        m[np.isfinite(z[a:b]).mean(axis=0) < MIN_VALID] = np.nan
        stats.append(m)
        classes.append(window_class(ph[a:b]))
    if not stats:
        return None, None
    return np.asarray(stats), np.asarray(classes)


def auc(pos, neg):
    """Mann-Whitney AUC per column, NaN-aware, 0.5 when a side is empty."""
    out = np.full(pos.shape[1], np.nan)
    for j in range(pos.shape[1]):
        p, n = pos[:, j], neg[:, j]
        p, n = p[np.isfinite(p)], n[np.isfinite(n)]
        if len(p) == 0 or len(n) == 0:
            continue
        both = np.concatenate([p, n])
        r = pd.Series(both).rank().to_numpy()
        out[j] = (r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
    return out


def normal_flights(limit=None):
    """The real normal flights, counted once; they are the negatives and the sources."""
    rep = pd.read_csv(BASE_REPORT)
    rep = rep[rep["attacked"] == 0].sort_values("flight_id")
    rep = rep[[(M03_NORM / f).is_file() for f in rep["file"]]]
    if rep.empty:
        raise SystemExit(f"No normalized normal flights in {M03_NORM}")
    return rep.head(limit) if limit else rep


def collect_normal(features, rep):
    files = [M03_NORM / f for f in rep["file"]]
    parts = []
    for n, f in enumerate(files, start=1):
        s, _ = blocks(f, features)
        if s is not None:
            parts.append(s)
        if n % 40 == 0 or n == len(files):
            print(f"  normal flights read {n}/{len(files)}")
    neg = np.concatenate(parts)
    print(f"Normal blocks (negatives): {len(neg):,} from {len(files)} flights")
    return neg


def collect_attacks(features, man):
    """Per attacked flight and class: the block statistics, kept as a list."""
    paths = dict(zip(man["attacked_flight_id"], man["file"]))
    ids = list(paths)
    rows, kept = [], 0
    for n, fid in enumerate(ids, start=1):
        path = M04_NORM / paths[fid]
        if not path.is_file():
            continue
        s, c = blocks(path, features)
        if s is None:
            continue
        kept += 1
        for cls in CLASSES:
            m = c == cls
            if m.any():
                rows.append((fid, cls, s[m]))
        if n % 100 == 0 or n == len(ids):
            print(f"  attacked flights read {n}/{len(ids)}")
    print(f"Attacked flights with blocks: {kept}")
    return rows


def flagged(rows, man, threshold, features, name):
    """One row per attacked flight and class: does any block pass the threshold."""
    j = features.index(name)
    out = []
    for fid, cls, s in rows:
        v = s[:, j]
        hit = bool(np.nanmax(v) > threshold) if np.isfinite(v).any() else False
        out.append({"attacked_flight_id": fid, "window_class": cls, "flagged": hit})
    keys = [
        "attacked_flight_id", "attack_type", "velocity_mode", "attack_set",
        "peak_offset_m", "drift_rate_mps", "mission_type",
    ]  # fmt: skip
    return pd.DataFrame(out).merge(man[keys], on="attacked_flight_id", how="left")


def run(limit=None):
    man = pd.read_csv(MANIFEST)
    first = sorted(M04_NORM.glob("*.parquet"))
    if not first:
        raise SystemExit(
            f"No normalized flights in {M04_NORM}. Run features_v2_m04.py first."
        )
    features = [c for c in pd.read_parquet(first[0]).columns if c not in NON_FEATURES]
    print("=" * 70)
    print("AEROVIGIL - MODULE 04 v2 BLOCK SEPARABILITY (5 s blocks, max |z|)")
    print("=" * 70)
    rep = normal_flights(limit)
    man = man[man["source_flight_id"].isin(set(rep["flight_id"]))]
    neg = collect_normal(features, rep)
    rows = collect_attacks(features, man)
    meta = man.set_index("attacked_flight_id")
    cells = {}
    for fid, cls, stats in rows:
        r = meta.loc[fid]
        key = (r["attack_set"], r["attack_type"], r["velocity_mode"], cls)
        cells.setdefault(key, []).append(stats)
    records = []
    for (aset, typ, mode, cls), parts in sorted(cells.items()):
        pos = np.concatenate(parts)
        for f, v in zip(features, auc(pos, neg)):
            records.append({
                "attack_set": aset, "attack_type": typ, "velocity_mode": mode,
                "window_class": cls, "feature": f, "auc": v,
                "blocks": len(pos), "flights": len(parts),
            })  # fmt: skip
    table = pd.DataFrame(records)
    AUC_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(AUC_FILE, index=False)
    report(table, rows, man, neg, features)
    return table


def pivot(df, index, values, agg="mean", columns="velocity_mode"):
    return df.pivot_table(
        index=index, columns=columns, values=values, aggfunc=agg, observed=True
    )


def show(title, frame):
    print(f"\n{title}")
    print(frame.round(3).to_string())


def report(table, rows, man, neg, features):
    pd.set_option("display.width", 220)
    main = table[table["attack_set"] == "main"]
    cell = ["attack_type", "velocity_mode", "window_class"]
    by_class = ["attack_type", "window_class"]
    counts = pivot(
        main.drop_duplicates(cell), cell[:2], "blocks", "sum", "window_class"
    )
    show("BLOCKS PER CLASS (main set, both modes)", counts.fillna(0).astype(int))
    best = main.loc[main.groupby(cell)["auc"].idxmax()]
    show(
        "BEST FEATURE PER ATTACK TYPE x MODE x CLASS (AUC vs normal blocks)",
        pivot(best, by_class, "auc"),
    )
    print("\n  (feature that wins each cell)")
    print(best.set_index(by_class + cell[1:2])["feature"].unstack().to_string())
    imu = main[main["feature"].str.startswith(("res_accel", "res_course"))]
    show(
        "IMU CONSISTENCY FEATURES ONLY: best AUC per mode",
        pivot(imu.loc[imu.groupby(cell)["auc"].idxmax()], by_class, "auc"),
    )
    name = main.loc[main["auc"].idxmax(), "feature"]
    thr = float(np.nanquantile(neg[:, features.index(name)], QUANTILE))
    f = flagged(rows, man, thr, features, name)
    print("\nDETECTION AT A FIXED FALSE-ALARM RATE")
    print(
        f"Feature {name}, threshold {thr:.2f} = the {QUANTILE:.1%} percentile "
        "of the normal blocks (1 false alarm in 1000 normal blocks)"
    )
    sub = f[f["attack_set"] == "main"]
    show(
        "SHARE OF ATTACKED FLIGHTS WITH A BLOCK ABOVE THE THRESHOLD",
        pivot(sub, by_class, "flagged"),
    )
    bands = pd.cut(sub["peak_offset_m"], [0, 2, 20, 1e9], labels=BAND_NAMES)
    show(
        "SAME, BY PEAK OFFSET OF THE FLIGHT",
        pivot(sub.assign(band=bands), "band", "flagged", columns="window_class"),
    )
    sweep = f[f["attack_set"] == "sweep"]
    if len(sweep):
        show(
            "DRIFT SWEEP (test flights, consistent velocity): flagged share by rate",
            pivot(sweep, "drift_rate_mps", "flagged", columns="window_class"),
        )
    print(f"\nAUC table: {AUC_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N source flights")
    run(ap.parse_args().limit)


if __name__ == "__main__":
    main()
