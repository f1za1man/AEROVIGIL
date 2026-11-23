r"""Step 5: acceptance checks for the window sequence table (leakage, geometry, labels)."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    UNIT_TO_SECONDS,
    load_split_map,
    load_yaml,
    resolve,
)  # noqa: E402
from aerovigil.lstm.build_sequences import feature_columns  # noqa: E402
from aerovigil.lstm.common import out_dir  # noqa: E402
from aerovigil.lstm.datasets import ae_masks, load_windows  # noqa: E402
from aerovigil.lstm.sequences import apply_scaler, fit_scaler  # noqa: E402

FORBIDDEN = ("attack", "offset", "spoof", "label", "rel_", "time_from")


def run(config, output_dir=None, n_recompute=30):
    cfg = load_yaml(config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    win_s = float(pre["processing"]["window_duration_seconds"])
    hop_s = win_s * (1.0 - float(pre["processing"]["window_overlap"]))
    scale = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    wcfg = cfg["windows"]
    out = out_dir(cfg, output_dir)
    X, meta = load_windows(out)
    flights = pd.read_parquet(out / "flights.parquet")
    sc = json.loads((out / "scaler.json").read_text())
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    fails, notes = defaultdict(list), {}

    def check(name, ok, detail=""):
        if not ok:
            fails[name].append(detail)

    T = int(round(win_s / wcfg["bin_s"]))
    check(
        "array shape matches the table and the window geometry",
        X.shape == (len(meta), T, wcfg["expected_feature_count"]),
        str(X.shape),
    )
    check("no infinite values", not np.isinf(X).any())
    notes["NaN share"] = f"{np.isnan(X).mean():.4%}"
    check(
        "window length equals the configured window",
        np.allclose(meta["window_end_s"] - meta["window_start_s"], win_s),
    )
    steps = meta.groupby("flight_id")["window_start_s"].apply(
        lambda s: bool(
            np.allclose(np.round(s.to_numpy() / hop_s) * hop_s, s.to_numpy(), atol=1e-6)
        )
    )
    check(
        "window starts lie on the hop grid",
        bool(steps.all()),
        str(list(steps[~steps].index[:2])),
    )
    pos = meta["y"] == 1
    check(
        "labels follow the majority rule",
        bool(
            ((meta["y"] == 0) == (meta["attack_fraction"] == 0)).all()
            and (
                meta.loc[pos, "attack_fraction"] >= wcfg["positive_min_fraction"]
            ).all()
            and (
                meta.loc[meta["y"] == -1, "attack_fraction"]
                < wcfg["positive_min_fraction"]
            ).all()
        ),
    )
    check(
        "no ground-truth column among the model features",
        not any(any(w in n.lower() for w in FORBIDDEN) for n in sc["features"]),
        str(sc["features"]),
    )
    own = meta[meta["origin"] == "orig"]
    check(
        "original windows inherit the flight split",
        bool((own["split"] == own["flight_id"].map(split_map)).all()),
    )
    syn = meta[meta["origin"].isin(["syn", "sweep"])]
    check(
        "synthetic windows inherit the split of their source flight",
        bool((syn["split"] == syn["source_flight_id"].map(split_map)).all()),
    )
    check(
        "a source flight never appears in two splits",
        meta.groupby("source_flight_id")["split"].nunique().max() == 1,
    )
    check(
        "sweep windows exist only in the configured splits",
        set(meta.loc[meta["origin"] == "sweep", "split"]) <= set(wcfg["sweep_splits"]),
    )
    att = flights[(flights["origin"] != "orig")]
    has_pos = meta[meta["y"] == 1].groupby("flight_id").size()
    check(
        "every synthetic attacked flight has attack windows",
        bool(att["flight_id"].isin(has_pos.index).all()),
        str(list(att.loc[~att["flight_id"].isin(has_pos.index), "flight_id"][:3])),
    )
    tr, va = ae_masks(meta)
    check(
        "autoencoder training windows contain no attack",
        bool(
            (meta.loc[tr, "attack_fraction"] == 0).all()
            and (meta.loc[tr, "flight_kind"] == "normal").all()
        ),
    )
    check(
        "autoencoder training and validation windows are flight-disjoint",
        not set(meta.loc[tr, "flight_id"]) & set(meta.loc[va, "flight_id"]),
    )
    again = fit_scaler(X[tr.to_numpy()], wcfg["scaler_clip"])
    check(
        "scaler equals a refit on normal training windows only",
        np.allclose(again["center"], sc["center"])
        and np.allclose(again["scale"], sc["scale"]),
    )
    z = apply_scaler(X[tr.to_numpy()][:3000], sc)
    check(
        "scaled normal training windows are centred",
        abs(float(np.median(z))) < 0.2,
        f"{np.median(z):.3f}",
    )
    check(
        "windows are in strict time order within each flight",
        bool((meta.groupby("flight_id")["window_start_s"].diff().dropna() > 0).all()),
    )

    # Independent recomputation of random original windows from the Module 03 feature files.
    rng = np.random.default_rng(0)
    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    file_of = dict(zip(report["flight_id"].astype(str), report["file"]))
    cand = np.flatnonzero((meta["origin"] == "orig").to_numpy())
    checked, bad = 0, 0
    for i in rng.choice(cand, size=min(n_recompute, len(cand)), replace=False):
        m = meta.iloc[i]
        f = pd.read_parquet(
            resolve(cfg["paths"]["features_dir"]) / file_of[m["flight_id"]]
        )
        feats = f[feature_columns(f)].to_numpy(float)
        t = (f["time_usec"].to_numpy("int64") - f["time_usec"].iloc[0]) * scale
        ref = np.full((T, feats.shape[1]), np.nan)
        for j in range(T):
            lo = m["window_start_s"] + j * wcfg["bin_s"]
            sel = (t >= lo - 1e-9) & (t < lo + wcfg["bin_s"] - 1e-9)
            if sel.any():
                with np.errstate(all="ignore"):
                    ref[j] = np.nanmean(
                        np.where(np.isfinite(feats[sel]), feats[sel], np.nan), axis=0
                    )
        checked += 1
        if not np.allclose(ref, X[i], atol=1e-4, equal_nan=True):
            bad += 1
    check(
        "independently recomputed windows match the table",
        bad == 0,
        f"{bad}/{checked} differ",
    )
    notes["recomputed windows"] = f"{checked}"

    all_checks = [
        "array shape matches the table and the window geometry",
        "no infinite values",
        "window length equals the configured window",
        "window starts lie on the hop grid",
        "labels follow the majority rule",
        "no ground-truth column among the model features",
        "original windows inherit the flight split",
        "synthetic windows inherit the split of their source flight",
        "a source flight never appears in two splits",
        "sweep windows exist only in the configured splits",
        "every synthetic attacked flight has attack windows",
        "autoencoder training windows contain no attack",
        "autoencoder training and validation windows are flight-disjoint",
        "scaler equals a refit on normal training windows only",
        "scaled normal training windows are centred",
        "windows are in strict time order within each flight",
        "independently recomputed windows match the table",
    ]
    print("=" * 65)
    print("AEROVIGIL - SEQUENCE VALIDATION")
    print("=" * 65)
    print(f"Windows: {len(meta):,} | {notes}\n")
    for c in all_checks:
        ok = c not in fails
        print(
            f" [{'PASS' if ok else 'FAIL'}] {c}" + ("" if ok else f" -> {fails[c][:2]}")
        )
    print("=" * 65)
    if fails:
        raise SystemExit("Sequence validation FAILED")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.output_dir)


if __name__ == "__main__":
    main()
