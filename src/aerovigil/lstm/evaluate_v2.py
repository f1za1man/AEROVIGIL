r"""AEROVIGIL Module 07 v2 step 3: score the LSTM detectors the way Module 05 v2 did.

Window level: ROC-AUC of event windows (ramp or snapback) against the clean windows of
the normal test flights. Plateau and post windows are excluded, because no feature of a
spoofed position can tell a steady 9.7 m offset from an ordinary flight path.

Flight level: one threshold per model from the calibration flights at a 1 alarm per
flight-hour budget, then a latch. A flight is detected if any window after the measured
onset crosses the threshold; the delay is that window's end minus the onset. This is the
scoring Module 05 v2 used, so the numbers sit next to each other.

Stealth frontier: the same latch over the sweep set, which is one linear drift per rate
per test flight. Module 06's chi-square test detected 5% at 0.05 and 0.10 m/s and 57% at
0.25 m/s. Whether the LSTM moves that frontier down is the question Module 07 exists to
answer.

Writes results_window.csv, results_window_raw.csv, flights.csv, false_alarms.csv,
       frontier.csv and thresholds.csv in data/processed/module07_v2

Usage:
    python src\aerovigil\lstm\evaluate_v2.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aerovigil.lstm.build_sequences_v2 import OUT_DIR, ROOT
from aerovigil.lstm.train_v2 import FAMILIES, load
from aerovigil.windowing.make_windows_v2 import STEP_S

BUDGET_PER_HOUR = 1.0
MODES = ("consistent", "position_only")


def detector_of(row):
    base = {"ae": "lstm_ae", "clf": "lstm_supervised"}[row["kind"]]
    return base if row["protocol"] == "seen" else f"{base}_unseen"


def hours(n_windows):
    return n_windows * STEP_S / 3600.0


def groups_of(meta):
    """{group name: mask of positive test event windows}."""
    test = meta["role"] == "test"
    out = {
        "original": test & (meta["origin"] == "orig") & (meta["y_event"] == 1),
    }
    syn = (
        test
        & (meta["origin"] == "syn")
        & (meta["attack_set"] == "main")
        & (meta["y_event"] == 1)
    )
    for fam in FAMILIES:
        if fam == "original":
            continue
        out[fam] = syn & (meta["attack_type"] == fam)
        for mode in MODES:
            out[f"{fam}|{mode}"] = out[fam] & (meta["velocity_mode"] == mode)
    out["synthetic_all"] = syn
    return {k: v.to_numpy() for k, v in out.items() if v.any()}


def negative_pool(meta):
    """Clean windows of the normal test flights: the shared negatives of every AUC."""
    return (
        (meta["role"] == "test")
        & (meta["origin"] == "orig")
        & (meta["attack_type"] == "none")
        & (meta["y_event"] == 0)
    ).to_numpy()


def calibration_pool(meta):
    """Clean windows of the calibration flights, which no model was fitted on."""
    return (
        (meta["role"] == "calib")
        & (meta["origin"] == "orig")
        & (meta["attack_type"] == "none")
        & (meta["y_event"] == 0)
    ).to_numpy()


def threshold(scores, alarms_per_hour=BUDGET_PER_HOUR):
    """Highest score the calibration pool may reach at the given alarm budget."""
    allowed = max(1, round(alarms_per_hour * hours(len(scores))))
    return float(np.sort(scores)[-allowed]), len(scores), hours(len(scores)), allowed


def window_rows(score, groups, pool):
    """ROC-AUC and the TPR at the pool's 99th percentile, per attack group."""
    neg = score[pool]
    cut = float(np.quantile(neg, 0.99))
    rows = []
    for name, mask in groups.items():
        pos = score[mask]
        if len(pos) == 0:
            continue
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        rows.append({
            "group": name, "n_pos": len(pos), "n_neg": len(neg),
            "roc_auc": float(roc_auc_score(y, np.concatenate([pos, neg]))),
            "tpr_at_1pct_fpr": float((pos > cut).mean()),
        })  # fmt: skip
    return rows


def latch(score, meta, thr):
    """Per test flight: the first alarm, whether it lands after onset, and the delay."""
    te = meta.loc[meta["role"] == "test"].copy()
    te["score"] = score[(meta["role"] == "test").to_numpy()]
    te["alarm"] = te["score"] > thr
    rows = []
    for fid, g in te.groupby("flight_id", sort=False):
        g = g.sort_values("window_start_s")
        attacked = g["attack_type"].iloc[0] != "none"
        hit = g[g["alarm"]]
        after = hit[hit["seconds_since_onset_end"] >= 0] if attacked else hit.iloc[:0]
        rows.append({
            "flight_id": fid, "origin": g["origin"].iloc[0],
            "attack_set": g["attack_set"].iloc[0],
            "mission_type": g["mission_type"].iloc[0],
            "attack_type": g["attack_type"].iloc[0],
            "velocity_mode": g["velocity_mode"].iloc[0],
            "drift_rate_mps": g["drift_rate_mps"].iloc[0],
            "attacked": attacked, "windows": len(g), "alarms": int(g["alarm"].sum()),
            "detected": bool(len(after)) if attacked else False,
            "delay_s": (
                float(after["seconds_since_onset_end"].iloc[0]) if len(after) else np.nan
            ),
            "offset_at_alarm_m": (
                float(after["offset_end_m"].iloc[0]) if len(after) else np.nan
            ),
            "max_offset_m": float(g["max_offset_m"].max()),
            "false_alarm_before_onset": bool(
                attacked and (hit["seconds_since_onset_end"] < 0).any()
            ),
        })  # fmt: skip
    flights = pd.DataFrame(rows)
    normal = te[te["attack_type"] == "none"]
    rate = float(normal["alarm"].sum() / hours(len(normal))) if len(normal) else np.nan
    return flights, rate


def run(output_dir=None, budget=BUDGET_PER_HOUR):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    sdir = out / "scores"
    idx = pd.read_csv(sdir / "index.csv")
    _, meta = load(out)
    groups, pool, calib = groups_of(meta), negative_pool(meta), calibration_pool(meta)
    print("=" * 70)
    print("AEROVIGIL - MODULE 07 v2 EVALUATION (event windows, latched alarms)")
    print("=" * 70)
    print(
        f"Models {len(idx)}   test clean windows {pool.sum():,} ({hours(pool.sum()):.2f} h)"
        f"   calibration windows {calib.sum():,} ({hours(calib.sum()):.2f} h)"
    )
    wrows, frows, trows = [], [], []
    for _, r in idx.iterrows():
        score = np.load(sdir / f"{r['name']}.npy")
        det, held = detector_of(r), r["held_out"]
        keep = groups
        if r["protocol"] == "unseen":
            keep = {
                k: v for k, v in groups.items() if k == held or k.startswith(f"{held}|")
            }
        for row in window_rows(score, keep, pool):
            wrows.append({"detector": det, "protocol": r["protocol"],
                          "held_out": held, "seed": r["seed"], **row})  # fmt: skip
        thr, n_pool, pool_h, allowed = threshold(score[calib], budget)
        trows.append({"name": r["name"], "detector": det, "seed": r["seed"],
                      "threshold": thr, "calibration_windows": n_pool,
                      "calibration_hours": pool_h, "allowed_alarms": allowed})  # fmt: skip
        flights, rate = latch(score, meta, thr)
        if r["protocol"] == "unseen":
            flights = flights[
                (flights["attack_type"] == held)
                | (~flights["attacked"])
                | (flights["attack_set"] == "sweep")
            ]
        flights.insert(0, "seed", r["seed"])
        flights.insert(0, "held_out", held)
        flights.insert(0, "detector", det)
        frows.append(flights)
        att = flights[flights["attacked"] & (flights["attack_set"] != "sweep")]
        print(
            f"  {r['name']:38s} thr={thr:.5f}  detected "
            f"{att['detected'].mean() if len(att) else np.nan:.2f}  "
            f"{rate:.2f} alarms/h on test normals"
        )
    wraw = pd.DataFrame(wrows)
    wraw.to_csv(out / "results_window_raw.csv", index=False)
    cols = ["roc_auc", "tpr_at_1pct_fpr"]
    wres = (
        wraw.groupby(["detector", "protocol", "group"])[cols]
        .agg(["mean", "std"])
        .round(3)
    )
    wres.columns = [f"{a}_{b}" for a, b in wres.columns]
    wres = wres.reset_index()
    wres["seeds"] = (
        wraw.groupby(["detector", "protocol", "group"])["seed"].nunique().to_numpy()
    )
    wres.to_csv(out / "results_window.csv", index=False)
    flights = pd.concat(frows, ignore_index=True)
    flights.to_csv(out / "flights.csv", index=False)
    pd.DataFrame(trows).to_csv(out / "thresholds.csv", index=False)
    report(wres, flights, out)
    return wres, flights


def by_seed(frame, keys):
    """Mean over seeds of the per-seed detection rate, delay and displacement."""
    if not len(frame):
        return pd.DataFrame()
    per = (
        frame.groupby(keys + ["seed"])
        .agg(
            n=("flight_id", "count"),
            detection_rate=("detected", "mean"),
            median_delay_s=("delay_s", "median"),
            median_offset_m=("offset_at_alarm_m", "median"),
        )
        .reset_index()
    )
    return (
        per.groupby(keys)
        .agg(
            n=("n", "first"),
            seeds=("seed", "nunique"),
            detection_rate=("detection_rate", "mean"),
            detection_sd=("detection_rate", "std"),
            median_delay_s=("median_delay_s", "mean"),
            median_offset_m=("median_offset_m", "mean"),
        )
        .reset_index()
    )


def report(wres, flights, out):
    pd.set_option("display.width", 220)
    print("\nWINDOW-LEVEL ROC-AUC on event windows (mean over seeds)")
    head = wres[~wres["group"].str.contains("\\|")]
    print(
        head.pivot_table(index="group", columns="detector", values="roc_auc_mean")
        .round(3)
        .to_string()
    )
    print("\nVELOCITY MODE (synthetic families, seen classifier and autoencoder)")
    split = wres[wres["group"].str.contains("\\|")].copy()
    split[["family", "mode"]] = split["group"].str.split("|", expand=True)
    print(
        split.pivot_table(
            index="family", columns=["detector", "mode"], values="roc_auc_mean"
        )
        .round(3)
        .to_string()
    )
    main = flights[flights["attack_set"] != "sweep"]
    print("\nDETECTION AFTER THE LATCH (test flights, mean over seeds)")
    print(
        by_seed(main[main["attacked"]], ["detector", "attack_type", "velocity_mode"])
        .round(3)
        .to_string(index=False)
    )
    normal = main[~main["attacked"]]
    fa = (
        normal.groupby(["detector", "seed"])
        .agg(alarms=("alarms", "sum"), windows=("windows", "sum"))
        .reset_index()
    )
    fa["alarms_per_hour"] = fa["alarms"] / hours(fa["windows"])
    fa_s = (
        fa.groupby("detector")
        .agg(
            alarms=("alarms", "mean"),
            hours=("windows", lambda v: hours(v.mean())),
            alarms_per_hour=("alarms_per_hour", "mean"),
            sd=("alarms_per_hour", "std"),
        )
        .reset_index()
    )
    fa_s.to_csv(out / "false_alarms.csv", index=False)
    print("\nFALSE ALARMS on the normal test flights (budget 1 per hour)")
    print(fa_s.round(3).to_string(index=False))
    sweep = flights[flights["attack_set"] == "sweep"]
    frontier = by_seed(sweep[sweep["attacked"]], ["detector", "drift_rate_mps"])
    frontier.to_csv(out / "frontier.csv", index=False)
    if len(frontier):
        print("\nSTEALTH FRONTIER (sweep set; Module 06 chi-square: 5% at 0.05 and")
        print("0.10 m/s, 57% at 0.25 m/s, 100% from 0.50 m/s up)")
        print(frontier.round(3).to_string(index=False))
    print(f"\nSaved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=BUDGET_PER_HOUR)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.output_dir, a.budget)


if __name__ == "__main__":
    main()
