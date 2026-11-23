from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "preprocessing" / "preprocessing.yaml"
OUT_DIR = ROOT / "data" / "processed" / "module03"
NORM_DIR = OUT_DIR / "normalized"
WIN_DIR = OUT_DIR / "windows"
SPLIT_FILE = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
CLEAN_REPORT = OUT_DIR / "cleaning_report.csv"

TARGET_HZ = 125  # fixed output grid for every window
GRID_DT = 1.0 / TARGET_HZ
MAX_STEP_S = 0.05  # a raw step larger than this counts as a dropout
ID_COLUMNS = ["flight_id", "mission_type", "time_usec", "attack_label", "after_gap"]
SPLITS = ("train", "validation", "test")


def main():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seconds = float(cfg["processing"]["window_duration_seconds"])
    overlap = float(cfg["processing"]["window_overlap"])
    attack_value = cfg["labels"]["attack_value"]

    N = int(round(seconds * TARGET_HZ))  # 625 samples
    STEP_S = seconds * (1 - overlap)  # 2.5 s
    GRID = np.arange(N) * GRID_DT  # 0 .. 4.992 s

    splits = pd.read_csv(SPLIT_FILE).set_index("flight_id")["split"]
    clean = pd.read_csv(CLEAN_REPORT).set_index("flight_id")
    files = sorted(NORM_DIR.glob("*.parquet"))
    WIN_DIR.mkdir(parents=True, exist_ok=True)

    features = [c for c in pd.read_parquet(files[0]).columns if c not in ID_COLUMNS]
    k = len(features)

    print("=" * 65)
    print("AEROVIGIL - TEMPORAL WINDOW GENERATION")
    print("=" * 65)
    print(f"Window: {seconds:g} s | step {STEP_S:g} s | " f"overlap {overlap:.0%}")
    print(
        f"Output grid: {N} samples @ {TARGET_HZ} Hz | "
        f"max raw step allowed {MAX_STEP_S} s\n"
    )

    X = {s: [] for s in SPLITS}
    records = {s: [] for s in SPLITS}
    candidates = dropped_gap = dropped_nan = 0

    for n, path in enumerate(files, start=1):
        g = pd.read_parquet(path)
        fid = g["flight_id"].iloc[0]
        split = splits[fid]

        t_us = g["time_usec"].to_numpy(dtype="int64")
        t = (t_us - t_us[0]) * 1e-6
        rows = len(t)
        y = g["attack_label"].to_numpy() == attack_value
        x = g[features].to_numpy(dtype="float32")
        mission = g["mission_type"].iloc[0]
        hz = int(round(1 / float(clean.loc[fid, "median_interval_s"])))

        if n % 40 == 0 or n == len(files):
            print(f"Windowing flight {n}/{len(files)}")
        if t[-1] < seconds:
            continue

        starts = np.arange(0, t[-1] - seconds + 1e-9, STEP_S)
        candidates += len(starts)

        i0 = np.searchsorted(t, starts, side="left")
        i1 = np.searchsorted(t, starts + seconds, side="right")
        i0c = np.minimum(i0, rows - 1)
        i1c = np.clip(i1 - 1, 0, rows - 1)

        # coverage: data must exist close to both window edges
        cover = (
            ((i1 - i0) >= 2)
            & (t[i0c] - starts <= MAX_STEP_S)
            & (starts + seconds - t[i1c] <= MAX_STEP_S)
        )

        # no dropout inside the window
        bad = (np.diff(t) > MAX_STEP_S).astype("int64")
        cb = np.concatenate([[0], np.cumsum(bad)])
        no_gap = (cb[i1c] - cb[i0c]) == 0
        ok_gap = cover & no_gap

        # no NaN feature inside the window
        nan_row = np.isnan(x).any(axis=1).astype("int64")
        cs = np.concatenate([[0], np.cumsum(nan_row)])
        ok_nan = (cs[i1] - cs[i0]) == 0

        dropped_gap += int((~ok_gap).sum())
        dropped_nan += int((ok_gap & ~ok_nan).sum())
        keep = np.flatnonzero(ok_gap & ok_nan)
        if len(keep) == 0:
            continue

        t_attack_start = t[np.argmax(y)] if y.any() else np.nan

        wins = np.empty((len(keep), N, k), dtype="float32")
        for out_i, w in enumerate(keep):
            a, b = i0[w], i1[w]
            tw = t[a:b]
            grid = starts[w] + GRID
            xw = x[a:b]
            for j in range(k):
                wins[out_i, :, j] = np.interp(grid, tw, xw[:, j])

            idx = np.clip(np.searchsorted(tw, grid, side="right") - 1, 0, len(tw) - 1)
            yg = y[a:b][idx]
            any_attack = bool(yg.any())
            first = int(np.argmax(yg)) if any_attack else -1

            records[split].append(
                {
                    "flight_id": fid,
                    "split": split,
                    "mission_type": mission,
                    "original_hz": hz,
                    "start_time_usec": int(t_us[0] + round(starts[w] * 1e6)),
                    "end_time_usec": int(t_us[0] + round((starts[w] + GRID[-1]) * 1e6)),
                    "duration_s": float(GRID[-1]),
                    "raw_samples": int(b - a),
                    "max_raw_step_s": float(np.diff(tw).max()),
                    "attack_fraction": float(yg.mean()),
                    "any_attack": int(any_attack),
                    "attack_onset_in_window": int(any_attack and not yg[0]),
                    "onset_offset_s": first * GRID_DT if any_attack else np.nan,
                    "seconds_since_flight_attack_start": (
                        float(grid[-1] - t_attack_start)
                        if not np.isnan(t_attack_start)
                        else np.nan
                    ),
                }
            )
        X[split].append(wins)

    # ---- write one compressed array + manifest per split -----------------
    manifests, shapes = [], {}
    for s in SPLITS:
        if not X[s]:
            raise SystemExit(f"No windows produced for split '{s}'")
        arr = np.concatenate(X[s], axis=0)
        man = pd.DataFrame(records[s])
        man.insert(0, "window_index", np.arange(len(man)))
        np.savez_compressed(
            WIN_DIR / f"windows_{s}.npz",
            X=arr,
            attack_fraction=man["attack_fraction"].to_numpy("float32"),
            any_attack=man["any_attack"].to_numpy("int8"),
            feature_names=np.array(features),
        )
        man.to_csv(WIN_DIR / f"window_manifest_{s}.csv", index=False)
        shapes[s] = arr.shape
        manifests.append(man)
        X[s] = None
        del arr
    full = pd.concat(manifests, ignore_index=True)
    full.to_csv(WIN_DIR / "window_manifest.csv", index=False)

    # ---- summary -----------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"Candidate windows        : {candidates:,}")
    print(f"Dropped (gap / coverage) : {dropped_gap:,}")
    print(f"Dropped (NaN features)   : {dropped_nan:,}")
    print(f"Windows kept             : {len(full):,}")

    summ = (
        full.groupby("split")
        .agg(
            windows=("window_index", "count"),
            flights=("flight_id", "nunique"),
            mean_attack_fraction=("attack_fraction", "mean"),
            windows_any_attack=("any_attack", "sum"),
            windows_with_onset=("attack_onset_in_window", "sum"),
        )
        .loc[list(SPLITS)]
    )
    print("\n" + summ.round(3).to_string())
    print("\nArray shapes:", {s: tuple(v) for s, v in shapes.items()})
    print("\nRaw samples per window:")
    print(full["raw_samples"].describe().round(1).to_string())
    print(
        "\nLargest raw step inside any window (s):",
        round(full["max_raw_step_s"].max(), 4),
    )

    fl = {s: set(full.loc[full["split"] == s, "flight_id"]) for s in SPLITS}
    sp = {s: set(splits[splits == s].index) for s in SPLITS}
    checks = {
        f"Window shape is ({N}, {k}) in every split": all(
            v[1:] == (N, k) for v in shapes.values()
        ),
        "Manifest rows match array rows": all(
            shapes[s][0] == (full["split"] == s).sum() for s in SPLITS
        ),
        "Every window has identical duration": bool(
            np.allclose(full["duration_s"], GRID[-1])
        ),
        "No window contains a dropout": bool(
            (full["max_raw_step_s"] <= MAX_STEP_S).all()
        ),
        "Overlap is 50% (step/size)": abs(STEP_S / seconds - 0.5) < 1e-9,
        "Train/validation flights disjoint": not (fl["train"] & fl["validation"]),
        "Train/test flights disjoint": not (fl["train"] & fl["test"]),
        "Validation/test flights disjoint": not (fl["validation"] & fl["test"]),
        "Windows only from assigned split flights": all(fl[s] <= sp[s] for s in SPLITS),
    }
    print("\nACCEPTANCE CHECKS")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nSaved to: {WIN_DIR}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Window checks FAILED")


if __name__ == "__main__":
    main()
