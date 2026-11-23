r"""Step 4: build the window sequence table used by every Module 07 model.

Original flights : read the Module 03 per-flight feature files.
Synthetic flights: run the SAME Module 03 feature builder on the Module 04 attacked flights.
Every valid window of every flight is kept (alarm timelines need them all).

Usage:
    python src\aerovigil\lstm\build_sequences.py
    python src\aerovigil\lstm\build_sequences.py --limit 2 --output-dir data\interim\module07\smoke
"""

import argparse
import json
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
from aerovigil.lstm.sequences import (
    apply_scaler,
    fit_scaler,
    flight_windows,
)  # noqa: E402

META_COLS = ["flight_id", "mission_type", "time_usec", "attack_label", "after_gap"]


def feature_columns(frame):
    return [c for c in frame.columns if c not in META_COLS]


def windows_of(f, win_s, hop_s, scale, wcfg):
    """Windows of one feature table f (Module 03 layout). Returns X, meta DataFrame, dropped."""
    feats = f[feature_columns(f)]
    t_s = (f["time_usec"].to_numpy("int64") - f["time_usec"].iloc[0]) * scale
    X, meta, dropped = flight_windows(
        feats.to_numpy(float),
        t_s,
        f["attack_label"].to_numpy(),
        f["after_gap"].to_numpy(),
        win_s,
        hop_s,
        wcfg["bin_s"],
        wcfg["max_missing_fraction"],
        positive_min_fraction=wcfg["positive_min_fraction"],
    )
    return X, pd.DataFrame(meta), dropped, list(feats.columns), t_s


def label_span(t_s, labels):
    on = t_s[labels > 0.5]
    return (float(on[0]), float(on[-1])) if len(on) else (np.inf, np.inf)


def run(config, limit=None, output_dir=None):
    from aerovigil.features.build_features import (
        build_flight,
    )  # the Module 03 feature builder

    cfg = load_yaml(config)
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
    out = Path(output_dir) if output_dir else resolve(cfg["paths"]["output_dir"])
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    man = man[man["attack_set"].isin(wcfg["synthetic_sets"])]
    man = man[(man["attack_set"] != "sweep") | man["split"].isin(wcfg["sweep_splits"])]
    if limit:
        report = report.groupby("mission_type", group_keys=False).head(limit)
        man = man.groupby(["attack_set", "attack_type"], group_keys=False).head(limit)

    print("=" * 65)
    print("AEROVIGIL - WINDOW SEQUENCES")
    print("=" * 65)
    print(
        f"Window {win_s:g} s, hop {hop_s:g} s, bin {wcfg['bin_s']:g} s -> "
        f"{int(round(win_s / wcfg['bin_s']))} steps | original flights {len(report)} | "
        f"synthetic flights {len(man)}"
    )
    started, dropped_total = time.time(), {}
    Xs, metas, flights, names = [], [], [], None

    def add(X, m, info, dropped):
        for k, v in dropped.items():
            dropped_total[k] = dropped_total.get(k, 0) + v
        if len(m):
            for k, v in info.items():
                m[k] = v
            Xs.append(X)
            metas.append(m)

    for n, (_, r) in enumerate(report.iterrows(), start=1):
        f = pd.read_parquet(resolve(cfg["paths"]["features_dir"]) / r["file"])
        X, m, dropped, names, t_s = windows_of(f, win_s, hop_s, scale, wcfg)
        if len(names) != wcfg["expected_feature_count"]:
            raise SystemExit(
                f"Expected {wcfg['expected_feature_count']} features, found {len(names)}"
            )
        fid, normal = str(r["flight_id"]), bool(r["attack_rows"] == 0)
        s, e = (
            (np.inf, np.inf)
            if normal
            else label_span(t_s, f["attack_label"].to_numpy())
        )
        if len(m):
            m = m.drop(columns="last_row").assign(offset_end_m=np.nan, rel_end_s=np.nan)
        info = {
            "flight_id": fid,
            "source_flight_id": fid,
            "origin": "orig",
            "split": split_map[fid],
            "mission_type": r["mission_type"],
            "flight_kind": "normal" if normal else "attacked",
            "attack_type": "none" if normal else "original",
            "velocity_mode": "na",
            "drift_rate_mps": np.nan,
        }
        add(X, m, info, dropped)
        flights.append(
            {**info, "attack_start_s": s, "attack_end_s": e, "windows": len(m)}
        )
        if n % 60 == 0 or n == len(report):
            print(f"Original flights {n}/{len(report)} ({time.time() - started:.0f} s)")
    for n, (_, r) in enumerate(man.iterrows(), start=1):
        g = pd.read_parquet(resolve(cfg["paths"]["attacked_dir"]) / r["file"])
        f = build_flight(g, c, scale)
        X, m, dropped, _, t_s = windows_of(f, win_s, hop_s, scale, wcfg)
        if len(m):
            rows = m["last_row"].to_numpy(int)
            m["offset_end_m"] = g["spoof_offset_m"].to_numpy(float)[rows]
            m["rel_end_s"] = g["time_from_attack_start_s"].to_numpy(float)[rows]
            m = m.drop(columns="last_row")
        info = {
            "flight_id": r["attacked_flight_id"],
            "source_flight_id": r["source_flight_id"],
            "origin": "syn" if r["attack_set"] == "main" else "sweep",
            "split": r["split"],
            "mission_type": r["mission_type"],
            "flight_kind": "attacked",
            "attack_type": r["attack_type"],
            "velocity_mode": r["velocity_mode"],
            "drift_rate_mps": r.get("drift_rate_mps", np.nan),
        }
        add(X, m, info, dropped)
        flights.append(
            {
                **info,
                "attack_start_s": float(r["start_s"]),
                "attack_end_s": float(r["end_s"]),
                "windows": len(m),
            }
        )
        if n % 60 == 0 or n == len(man):
            print(f"Synthetic flights {n}/{len(man)} ({time.time() - started:.0f} s)")

    X = np.concatenate(Xs)
    meta = pd.concat(metas, ignore_index=True)
    tr = (
        (meta["origin"] == "orig")
        & (meta["split"] == "train")
        & (meta["flight_kind"] == "normal")
        & (meta["y"] == 0)
    ).to_numpy()
    scaler = fit_scaler(X[tr], wcfg["scaler_clip"])
    scaler["features"] = names
    scaler["fitted_on"] = "original, train split, normal flights, y == 0 windows"
    scaler["n_windows"] = int(tr.sum())
    np.savez_compressed(out / "windows.npz", X=X)
    meta.to_parquet(out / "windows_meta.parquet", index=False)
    pd.DataFrame(flights).to_parquet(out / "flights.parquet", index=False)
    (out / "scaler.json").write_text(json.dumps(scaler, indent=2))

    print("\n" + "-" * 65)
    print(f"Windows: {len(meta):,} | array {X.shape} | {X.nbytes / 1e6:.0f} MB")
    print(f"Dropped windows: {dropped_total}")
    print(
        f"NaN share of values: {np.isnan(X).mean():.4%} | scaler fitted on {int(tr.sum()):,} normal train windows"
    )
    print("\nWINDOWS BY ORIGIN x SPLIT x LABEL (1 attack, 0 normal, -1 ambiguous)")
    print(pd.crosstab([meta["origin"], meta["split"]], meta["y"]).to_string())
    print("\nORIGINAL NORMAL FLIGHTS BY MISSION TYPE x SPLIT")
    fl = pd.DataFrame(flights)
    print(
        pd.crosstab(
            fl.loc[
                (fl.origin == "orig") & (fl.flight_kind == "normal"), "mission_type"
            ],
            fl.loc[(fl.origin == "orig") & (fl.flight_kind == "normal"), "split"],
        ).to_string()
    )
    Xs_ = apply_scaler(X[tr][:2000], scaler)
    print(
        f"\nScaled normal windows: mean {Xs_.mean():.3f}, std {Xs_.std():.3f}, "
        f"share clipped {(np.abs(Xs_) >= scaler['clip'] - 1e-6).mean():.3%}"
    )
    print(f"Elapsed: {time.time() - started:.0f} s | saved to {out}")
    print("=" * 65)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument(
        "--limit", type=int, default=None, help="flights per group (smoke test)"
    )
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.limit, a.output_dir)


if __name__ == "__main__":
    main()
