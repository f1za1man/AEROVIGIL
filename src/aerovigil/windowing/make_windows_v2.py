r"""AEROVIGIL Module 03 v2: phase-aware 5 s windows from the normalized Raw features.

Same geometry as Module 03: 5 s windows, 2.5 s hop, linear interpolation onto a 625
point 125 Hz grid. A window is dropped for a raw step above 0.05 s, missing data at
an edge, or any NaN feature. What is new are the labels, taken from the measured
spoofing offset and not only from attack_enabled:

    window_class : normal | pre | ramp | plateau | snapback | post
        ramp      contains any ramp rows (the offset is growing)
        snapback  contains plateau rows and post rows (the offset returns to 0)
        pre/post  only pre-attack / only post-attack rows
        plateau   only plateau rows (the constant offset; not visible to features)
    attack_fraction : share of grid points with the ORIGINAL label (as in Module 03)
    spoof_fraction  : share of grid points inside the measured spoofed interval
    max_offset_m, end_offset_m, seconds_since_onset_end

Reuses data/interim/module03/flight_splits.csv unchanged.

Writes data/processed/module03_v2/windows/windows_<split>.npz,
       window_manifest_<split>.csv and window_manifest.csv

Usage:
    python -m aerovigil.windowing.make_windows_v2
"""

from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.preprocessing.normalize_features_v2 import NON_FEATURES

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "module03_v2"
NORM_DIR = OUT_DIR / "normalized"
WIN_DIR = OUT_DIR / "windows"
SPLIT_FILE = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
SPLITS = ("train", "validation", "test")
HZ, SECONDS, STEP_S, MAX_STEP_S = 125, 5.0, 2.5, 0.05
N = round(SECONDS * HZ)
GRID = np.arange(N) / HZ
CLASSES = ["normal", "pre", "ramp", "plateau", "snapback", "post"]


def window_class(ph):
    """Class of one window from the phase code of its grid points."""
    if (ph == 2).any():
        return "ramp"
    if (ph == 3).any() and (ph == 4).any():
        return "snapback"
    if (ph == 3).any():
        return "plateau"
    if (ph == 4).any():
        return "post"
    return "pre" if (ph == 1).any() else "normal"


def keep_mask(t, x, starts):
    rows = len(t)
    i0 = np.searchsorted(t, starts, side="left")
    i1 = np.searchsorted(t, starts + SECONDS, side="right")
    i0c, i1c = np.minimum(i0, rows - 1), np.clip(i1 - 1, 0, rows - 1)
    cover = (
        ((i1 - i0) >= 2)
        & (t[i0c] - starts <= MAX_STEP_S)
        & (starts + SECONDS - t[i1c] <= MAX_STEP_S)
    )
    bad = np.concatenate([[0], np.cumsum(np.diff(t) > MAX_STEP_S)])
    ok_gap = cover & ((bad[i1c] - bad[i0c]) == 0)
    nan_rows = np.concatenate([[0], np.cumsum(np.isnan(x).any(axis=1))])
    ok_nan = (nan_rows[i1] - nan_rows[i0]) == 0
    return i0, i1, ok_gap, ok_nan


def window_flight(g, features, split):
    t = g["t_s"].to_numpy(float)
    x = g[features].to_numpy(np.float32)
    if t[-1] < SECONDS:
        return None, [], 0, 0, 0
    starts = np.arange(0, t[-1] - SECONDS + 1e-9, STEP_S)
    i0, i1, ok_gap, ok_nan = keep_mask(t, x, starts)
    keep = np.flatnonzero(ok_gap & ok_nan)
    stats = (len(starts), int((~ok_gap).sum()), int((ok_gap & ~ok_nan).sum()))
    if len(keep) == 0:
        return None, [], *stats
    cols = {c: g[c].to_numpy() for c in ("attack_label", "spoof_truth", "phase")}
    offset = g["offset_m"].to_numpy(float)
    since = g["seconds_since_onset"].to_numpy(float)
    hz = round(1 / float(np.median(np.diff(t))))
    wins = np.empty((len(keep), N, len(features)), dtype=np.float32)
    records = []
    for out_i, w in enumerate(keep):
        a, b = i0[w], i1[w]
        grid = starts[w] + GRID
        for j in range(len(features)):
            wins[out_i, :, j] = np.interp(grid, t[a:b], x[a:b, j])
        idx = np.clip(np.searchsorted(t[a:b], grid, side="right") - 1, 0, b - a - 1)
        y = cols["attack_label"][a:b][idx] == 1
        ph = cols["phase"][a:b][idx]
        off = offset[a:b][idx]
        records.append(
            {
                "flight_id": g["flight_id"].iloc[0],
                "split": split,
                "mission_type": g["mission_type"].iloc[0],
                "original_hz": hz,
                "start_s": float(starts[w]),
                "end_s": float(starts[w] + GRID[-1]),
                "raw_samples": int(b - a),
                "max_raw_step_s": float(np.diff(t[a:b]).max()),
                "window_class": window_class(ph),
                "attack_fraction": float(y.mean()),
                "any_attack": int(y.any()),
                "attack_onset_in_window": int(y.any() and not y[0]),
                "spoof_fraction": float((cols["spoof_truth"][a:b][idx] == 1).mean()),
                "max_offset_m": (
                    float(np.nanmax(off)) if np.isfinite(off).any() else np.nan
                ),
                "end_offset_m": float(off[-1]),
                "seconds_since_onset_end": float(since[a:b][idx][-1]),
            }
        )
    return wins, records, *stats


def main():
    splits = pd.read_csv(SPLIT_FILE).set_index("flight_id")["split"]
    files = sorted(NORM_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No normalized files. Run normalize_features_v2.py first.")
    features = [c for c in pd.read_parquet(files[0]).columns if c not in NON_FEATURES]
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("AEROVIGIL - PHASE-AWARE WINDOWS v2")
    print("=" * 65)
    print(
        f"Window {SECONDS:g} s | hop {STEP_S:g} s | grid {N} @ {HZ} Hz | {len(features)} features\n"
    )
    X = {s: [] for s in SPLITS}
    recs = {s: [] for s in SPLITS}
    cand = drop_gap = drop_nan = 0
    for n, path in enumerate(files, start=1):
        g = pd.read_parquet(path)
        split = splits[g["flight_id"].iloc[0]]
        wins, records, c, dg, dn = window_flight(g, features, split)
        cand, drop_gap, drop_nan = cand + c, drop_gap + dg, drop_nan + dn
        if wins is not None:
            X[split].append(wins)
            recs[split].extend(records)
        if n % 40 == 0 or n == len(files):
            print(f"Windowed {n}/{len(files)} flights")
    manifests, shapes = [], {}
    for s in SPLITS:
        if not X[s]:
            raise SystemExit(f"No windows for split '{s}'")
        arr = np.concatenate(X[s])
        man = pd.DataFrame(recs[s])
        man.insert(0, "window_index", np.arange(len(man)))
        codes = man["window_class"].map(CLASSES.index).to_numpy("int8")
        np.savez_compressed(
            WIN_DIR / f"windows_{s}.npz",
            X=arr,
            attack_fraction=man["attack_fraction"].to_numpy("float32"),
            spoof_fraction=man["spoof_fraction"].to_numpy("float32"),
            window_class=codes,
            class_names=np.array(CLASSES),
            feature_names=np.array(features),
        )
        man.to_csv(WIN_DIR / f"window_manifest_{s}.csv", index=False)
        shapes[s] = arr.shape
        manifests.append(man)
        X[s] = None
    full = pd.concat(manifests, ignore_index=True)
    full.to_csv(WIN_DIR / "window_manifest.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + "-" * 65)
    print(
        f"Candidate windows: {cand:,}  dropped gap/coverage: {drop_gap:,}  dropped NaN: {drop_nan:,}"
    )
    print(f"Windows kept     : {len(full):,}")
    print("\nWINDOWS PER SPLIT AND CLASS")
    table = full.groupby(["split", "window_class"]).size().unstack(fill_value=0)
    print(table.reindex(index=list(SPLITS), columns=CLASSES, fill_value=0).to_string())
    print("\nORIGINAL LABEL vs MEASURED CLASS (any_attack by class, all splits)")
    print(
        full.groupby("window_class")["any_attack"]
        .agg(["count", "mean"])
        .round(3)
        .to_string()
    )
    print("\nArray shapes:", {s: tuple(v) for s, v in shapes.items()})
    fl = {s: set(full.loc[full["split"] == s, "flight_id"]) for s in SPLITS}
    sp = {s: set(splits[splits == s].index) for s in SPLITS}
    checks = {
        f"window shape is ({N}, {len(features)}) in every split": all(
            v[1:] == (N, len(features)) for v in shapes.values()
        ),
        "manifest rows match array rows": all(
            shapes[s][0] == (full["split"] == s).sum() for s in SPLITS
        ),
        "no window contains a dropout": bool(
            (full["max_raw_step_s"] <= MAX_STEP_S).all()
        ),
        "no NaN in any saved window": all(
            not np.isnan(np.load(WIN_DIR / f"windows_{s}.npz")["X"]).any()
            for s in SPLITS
        ),
        "train/validation/test flights disjoint": not (fl["train"] & fl["validation"])
        and not (fl["train"] & fl["test"])
        and not (fl["validation"] & fl["test"]),
        "windows only from assigned split flights": all(fl[s] <= sp[s] for s in SPLITS),
    }
    missing = [
        c
        for c in CLASSES
        if c not in set(full.loc[full["split"] == "train", "window_class"])
    ]
    print("\nclasses missing from train:", missing if missing else "none")
    print("\nACCEPTANCE CHECKS")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\nSaved to:", WIN_DIR)
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Window checks FAILED")


if __name__ == "__main__":
    main()
