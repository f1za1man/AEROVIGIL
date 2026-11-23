r"""Step 8: window-level metrics, false-alarm-budget thresholds, alarms and result tables.

Window level (comparable to Module 05): ROC-AUC, TPR at 1% FPR, precision / recall / F1 at the
validation 1%-FPR threshold. The normal pool is always the normal windows of the original
TEST flights. Seen protocol: every attack family in training. Unseen protocol: one family
held out of the supervised model's training and validation data.

Flight level (comparable to Module 06): one threshold per model from the shared false-alarm
budget on NORMAL VALIDATION flights, then alarm time, delay, displacement before the alarm and
false alarms per flight-hour on the TEST flights.

Usage:
    python src\aerovigil\lstm\evaluate_lstm.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.lstm.alarms import (
    calibrate_threshold,
    flight_metrics,
    window_metrics,
)  # noqa: E402
from aerovigil.lstm.common import load_yaml, out_dir, resolve  # noqa: E402
from aerovigil.lstm.datasets import (
    ae_masks,
    load_windows,
    eval_attack_groups,
    eval_pool,
)  # noqa: E402


def detector_of(row):
    return {"ae": "lstm_ae", "clf": "lstm_supervised"}[row["kind"]] + (
        "" if row["protocol"] == "seen" else "_unseen"
    )


def run(config, output_dir=None):
    cfg = load_yaml(config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    out = out_dir(cfg, output_dir)
    d = cfg["detector"]
    _, meta = load_windows(out)
    flights = pd.read_parquet(out / "flights.parquet").set_index("flight_id")
    sdir = out / "scores"
    idx = pd.read_csv(sdir / "index.csv")
    win_s = float((meta["window_end_s"] - meta["window_start_s"]).iloc[0])
    hop_s = win_s * (1.0 - float(pre["processing"]["window_overlap"]))
    scores = {r["name"]: np.load(sdir / f"{r['name']}.npy") for _, r in idx.iterrows()}

    pool = eval_pool(meta).to_numpy()
    groups = {k: v.to_numpy() for k, v in eval_attack_groups(meta).items()}
    _, ae_val = ae_masks(meta)
    val_normal = ae_val.to_numpy()
    fpr = d["operating_point_fpr"]

    print("=" * 65)
    print("AEROVIGIL - LSTM EVALUATION")
    print("=" * 65)
    print(
        f"Models scored: {len(idx)} | test normal windows {int(pool.sum()):,} | "
        f"window {win_s:g} s, hop {hop_s:g} s"
    )

    # ---------------- window level ----------------
    rows = []
    for _, r in idx.iterrows():
        s = scores[r["name"]]
        thr = float(np.quantile(s[val_normal], 1.0 - fpr))
        for g, mask in groups.items():
            if r["protocol"] == "unseen":
                held = r["held_out"]
                if g != held and not g.startswith(f"{held}|"):
                    continue
            m = window_metrics(s, mask, pool, thr, fpr)
            if m:
                rows.append(
                    {
                        "detector": detector_of(r),
                        "protocol": r["protocol"],
                        "held_out": r["held_out"],
                        "seed": r["seed"],
                        "group": g,
                        **m,
                    }
                )
    wres = pd.DataFrame(rows)
    wres.to_csv(out / "results_window_raw.csv", index=False)
    metric_cols = [
        c
        for c in wres.columns
        if c in ("roc_auc", "precision", "recall", "f1") or c.startswith("tpr_at")
    ]
    agg = (
        wres.groupby(["detector", "protocol", "group"])[metric_cols]
        .agg(["mean", "std"])
        .round(3)
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["seeds"] = (
        wres.groupby(["detector", "protocol", "group"])["seed"].nunique().to_numpy()
    )
    agg.to_csv(out / "results_window.csv", index=False)

    # ---------------- flight level ----------------
    val_flights = meta[
        (meta["origin"] == "orig")
        & (meta["split"] == "validation")
        & (meta["flight_kind"] == "normal")
    ]
    val_groups = [
        np.flatnonzero((meta["flight_id"] == f).to_numpy())
        for f in val_flights["flight_id"].unique()
    ]
    test_flights = flights[flights["split"] == "test"]
    pos_by_flight = {
        f: np.asarray(v) for f, v in meta.groupby("flight_id").indices.items()
    }
    t_end, offset_end = meta["window_end_s"].to_numpy(), meta["offset_end_m"].to_numpy()
    alarm_rows, thr_info = [], {}
    refr = d["refractory_s"]
    for _, r in idx.iterrows():
        s = scores[r["name"]]
        cal_flights = [(s[i], t_end[i]) for i in val_groups]
        thr, hours, allowed, used = calibrate_threshold(
            cal_flights, d["false_alarm_budget_per_hour"], hop_s, refr
        )
        thr_info[r["name"]] = {
            "threshold": thr,
            "calibration_hours": hours,
            "allowed_alarms": allowed,
            "alarms_used": used,
        }
        for fid, fl in test_flights.iterrows():
            if (
                r["protocol"] == "unseen"
                and fl["flight_kind"] == "attacked"
                and fl["origin"] != "sweep"
            ):
                if fl["attack_type"] != r["held_out"]:
                    continue
            if r["protocol"] == "unseen" and fl["origin"] == "sweep":
                continue
            ii = pos_by_flight.get(fid)
            if ii is None or len(ii) == 0:
                continue
            m = flight_metrics(
                s[ii],
                t_end[ii],
                offset_end[ii],
                thr,
                refr,
                fl["attack_start_s"],
                fl["attack_end_s"],
                hop_s,
            )
            alarm_rows.append(
                {
                    "flight_id": fid,
                    "set": fl["origin"],
                    "split": "test",
                    "mission_type": fl["mission_type"],
                    "attack_type": fl["attack_type"],
                    "velocity_mode": fl["velocity_mode"],
                    "drift_rate_mps": fl["drift_rate_mps"],
                    "detector": detector_of(r),
                    "held_out": r["held_out"],
                    "seed": r["seed"],
                    **m,
                }
            )
    alarms = pd.DataFrame(alarm_rows)
    alarms.to_csv(out / "alarms.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(thr_info, indent=2))

    def by_seed(df, keys):
        g = (
            df.groupby(keys + ["seed"])
            .agg(
                n=("flight_id", "count"),
                detection_rate=("detected", "mean"),
                median_delay_s=("delay_s", "median"),
                median_displacement_m=("displacement_m", "median"),
            )
            .reset_index()
        )
        return (
            g.groupby(keys)
            .agg(
                n=("n", "first"),
                seeds=("seed", "nunique"),
                detection_rate=("detection_rate", "mean"),
                detection_rate_sd=("detection_rate", "std"),
                median_delay_s=("median_delay_s", "mean"),
                median_displacement_m=("median_displacement_m", "mean"),
            )
            .reset_index()
        )

    att = alarms[alarms["attacked"] & alarms["set"].isin(["orig", "syn"])]
    summary = by_seed(att, ["set", "attack_type", "velocity_mode", "detector"])
    summary.to_csv(out / "summary_by_attack.csv", index=False)
    normal = alarms[(alarms["set"] == "orig") & (~alarms["attacked"])]
    fa = (
        normal.groupby(["detector", "held_out", "seed"], dropna=False)
        .agg(alarms=("false_alarms", "sum"), hours=("normal_hours", "sum"))
        .reset_index()
    )
    fa["false_alarms_per_hour"] = fa["alarms"] / fa["hours"]
    fa_s = (
        fa.groupby("detector")
        .agg(
            alarms=("alarms", "mean"),
            hours=("hours", "mean"),
            false_alarms_per_hour=("false_alarms_per_hour", "mean"),
            sd=("false_alarms_per_hour", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    # exact 95% Poisson interval for the alarm rate (few alarms = a wide interval)
    k = fa_s["alarms"].round().to_numpy()
    fa_s["fa_per_hour_ci_low"] = chi2.ppf(0.025, 2 * k) / 2 / fa_s["hours"]
    fa_s["fa_per_hour_ci_high"] = chi2.ppf(0.975, 2 * k + 2) / 2 / fa_s["hours"]
    fa_s.to_csv(out / "false_alarms.csv", index=False)
    sw = alarms[(alarms["set"] == "sweep") & alarms["attacked"]]
    frontier = (
        by_seed(sw, ["detector", "drift_rate_mps"]) if len(sw) else pd.DataFrame()
    )
    frontier.to_csv(out / "sweep_frontier.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nWINDOW-LEVEL ROC-AUC (mean over seeds), test flights")
    pv = agg[
        agg["group"].isin(
            ["original", "synthetic_all"]
            + [
                g
                for g in groups
                if "|" not in g and g not in ("original", "synthetic_all")
            ]
        )
    ]
    print(
        pv.pivot_table(index="group", columns=["detector"], values="roc_auc_mean")
        .round(3)
        .to_string()
    )
    print(
        f"\nTHRESHOLDS from {int(len(val_groups))} normal validation flights "
        f"({next(iter(thr_info.values()))['calibration_hours']:.2f} h; budget "
        f"{d['false_alarm_budget_per_hour']:g}/h -> at most "
        f"{next(iter(thr_info.values()))['allowed_alarms']:.1f} alarms)"
    )
    print("\nFALSE ALARMS on normal time of original test flights")
    print(fa_s.round(3).to_string(index=False))
    print("\nDETECTION BY ATTACK (test flights, mean over seeds)")
    print(summary.round(3).to_string(index=False))
    if len(frontier):
        print(
            "\nPRELIMINARY STEALTH FRONTIER (sweep, consistent-velocity attacker, test flights)"
        )
        print(frontier.round(2).to_string(index=False))
    print(f"\nSaved to: {out}")
    print("=" * 65)
    return agg, summary, fa_s, frontier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.output_dir)


if __name__ == "__main__":
    main()
