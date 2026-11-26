r"""AEROVIGIL plan M5 (repo Module 05 v2): all six classical models, tuned and latched.

Module 05 v2 re-scored logistic regression and random forest as event detectors with
a per-flight latch. This adds the other four models the plan names, Naive Bayes, KNN,
a decision tree and an Isolation Forest, and scores all six the same way on Module 05's
headline setup: event labels (ramp or snapback against clean), all 216 window features
(GPS + IMU), trained on the real attacks plus the synthetic main set.

Every model gets a small grid (configs/baselines/baselines.yaml) and keeps the setting
with the best ROC-AUC on the VALIDATION flights. ROC-AUC is used for all six because it
does not depend on a threshold; the threshold comes afterwards from the alarm budget.
The Isolation Forest never sees an attack: it is fitted on normal windows only.

Threshold: 1 alarm per flight-hour, the budget every detector in the project uses. The
pool of clean scores is out-of-fold scores on the training flights' normal windows
(folds grouped by SOURCE flight, so a synthetic copy never sits on the other side of a
fold from the flight it was made from) plus the validation normal windows. A window
alarms above the threshold, alarms closer than 10 s are merged (the refractory every
other detector uses), and the alarm latches per flight.

Two protocols:
    seen       train / validation / test flights as split by seed 42; every mission
               appears in training
    unseen     leave one mission out: tune, train and calibrate on the other two
               missions, then test on every flight of the held-out mission

Reported per model and protocol
    window level (event labels, at the latch threshold): accuracy, precision, recall,
        F1, ROC-AUC and the confusion matrix (TN, FP, FN, TP), plus the ROC-AUC against
        position-only and velocity-consistent synthetic attacks
    flight level: share of attacked flights caught, median delay from onset, and false
        alarms on the normal flights with count, hours and Poisson 95% upper bound;
        hours come from timestamps (Module 10's monitored time, signals_index.csv)

Writes data/processed/module05_v2/all_models_window.csv, all_models_flight.csv,
all_models_tuning.csv and reports/tables/B_classical_models.csv

Usage:
    python src\aerovigil\baselines\all_models_v2.py
    python src\aerovigil\baselines\all_models_v2.py --models naive_bayes,knn --quick
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold

from aerovigil.baselines.build_window_tables_v2 import OUT_DIR
from aerovigil.baselines.models import (
    SUPERVISED,
    UNSUPERVISED,
    build_model,
    fit_model,
    grid,
    score_model,
)
from aerovigil.windowing.make_windows_v2 import STEP_S

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "baselines" / "baselines.yaml"
SIGNALS_INDEX = ROOT / "data" / "processed" / "module10" / "signals_index.csv"
TABLE = ROOT / "reports" / "tables" / "B_classical_models.csv"
MODELS = SUPERVISED + UNSUPERVISED
MISSIONS = ("curved", "random", "straight")
SEED = 42
FOLDS = 4
BUDGET_PER_HOUR = 1.0
REFRACTORY_S = 10.0
Y = "y_event"


# ------------------------------------------------------------------- data --


def feature_columns(frame):
    return [c for c in frame.columns if "__" in c]


def load(out_dir):
    orig = pd.read_parquet(out_dir / "windows_orig.parquet")
    syn_path = out_dir / "windows_syn.parquet"
    if not syn_path.is_file():
        raise SystemExit(
            f"Missing {syn_path}. Run build_window_tables_v2.py --sources orig,syn"
        )
    syn = pd.read_parquet(syn_path)
    syn = syn[syn["attack_set"] == "main"]  # the sweep is chosen, not sampled
    return orig, syn


def protocol_sets(orig, syn, held_out=None):
    """(train, validation, clean-train, test-orig, test-syn) for one protocol."""
    if held_out is None:
        o_in, s_in = orig, syn
        o_test = orig[orig["split"] == "test"]
        s_test = syn[syn["split"] == "test"]
        o_in = o_in[o_in["split"] != "test"]
        s_in = s_in[s_in["split"] != "test"]
    else:
        o_in = orig[orig["mission_type"] != held_out]
        s_in = syn[syn["mission_type"] != held_out]
        o_test = orig[orig["mission_type"] == held_out]
        s_test = syn[syn["mission_type"] == held_out]
    tr = pd.concat(
        [o_in[o_in["split"] == "train"], s_in[s_in["split"] == "train"]],
        ignore_index=True,
    )
    va = pd.concat(
        [o_in[o_in["split"] == "validation"], s_in[s_in["split"] == "validation"]],
        ignore_index=True,
    )
    clean_va = o_in[(o_in["split"] == "validation") & (o_in["attack_type"] == "none")]
    return tr[tr[Y] >= 0], va[va[Y] >= 0], clean_va, o_test, s_test


# ---------------------------------------------------------------- scoring --


def xy(frame, cols):
    return frame[cols].to_numpy("float32"), frame[Y].to_numpy()


def fit(name, params, frame, cols, seed=SEED):
    x, y = xy(frame, cols)
    return fit_model(name, build_model(name, params, seed), x, y)


def score(name, pipe, frame, cols):
    """Attack score, higher = more attack-like.

    Naive Bayes scores with its log-odds, not its probability: on these features its
    probability rounds to exactly 1.0 for many clean windows, so the threshold landed
    on 1.0 and nothing could ever exceed it. The log-odds rank windows identically
    (same AUC) but do not saturate.
    """
    if len(frame) == 0:
        return np.zeros(0)
    x = frame[cols].to_numpy("float32")
    if name == "naive_bayes":
        jll = pipe[-1].predict_joint_log_proba(pipe[:-1].transform(x))
        return jll[:, 1] - jll[:, 0]
    return score_model(name, pipe, x)


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def tune(name, params_grid, tr, va, cols):
    """Best grid point by ROC-AUC on the validation windows."""
    rows = []
    for params in params_grid:
        pipe = fit(name, params, tr, cols)
        rows.append({"params": json.dumps(params),
                     "val_auc": auc(va[Y], score(name, pipe, va, cols))})  # fmt: skip
    tab = pd.DataFrame(rows)
    best = tab.loc[tab["val_auc"].fillna(-1).idxmax(), "params"]
    return json.loads(best), tab


def clean_pool(name, params, tr, clean_va, final, cols):
    """Honest scores on clean windows: out-of-fold on the training flights' normal
    windows, grouped by source flight, plus the validation normal windows."""
    groups = tr["source_flight_id"].fillna(tr["flight_id"]).to_numpy()
    normal = (tr["attack_type"] == "none").to_numpy()
    parts = []
    n = min(FOLDS, len(np.unique(groups)))
    if n >= 2 and normal.any():
        for a, b in GroupKFold(n_splits=n).split(tr, groups=groups):
            held = tr.iloc[b][normal[b]]
            if len(held):
                pipe = fit(name, params, tr.iloc[a], cols)
                parts.append(score(name, pipe, held, cols))
    parts.append(score(name, final, clean_va, cols))
    return np.concatenate(parts)


def threshold_for(pool, budget=BUDGET_PER_HOUR):
    """Score the clean pool may reach budget x hours times. Hours: one hop per window,
    since the windows tile the timestamps at STEP_S seconds."""
    hours = len(pool) * STEP_S / 3600.0
    allowed = max(1, int(round(budget * hours)))
    return float(np.sort(pool)[-allowed]), hours, allowed


def refractory_count(times, seconds=REFRACTORY_S):
    times = np.sort(np.asarray(times, float))
    n, last = 0, -np.inf
    for t in times:
        if t >= last + seconds:
            n, last = n + 1, t
    return n


def latch(frame, scores, thr):
    """Per flight: detection, delay from onset, and refractory-merged alarm count."""
    f = frame.assign(score=scores, alarm=scores > thr)
    rows = []
    for fid, g in f.groupby("flight_id", sort=False):
        attacked = g["attack_type"].iloc[0] != "none"
        hit = g[g["alarm"]].sort_values("window_end_s")
        after = hit[hit["seconds_since_onset_end"] >= 0] if attacked else hit.iloc[:0]
        rows.append({
            "flight_id": fid, "origin": g["origin"].iloc[0],
            "mission_type": g["mission_type"].iloc[0],
            "attack_type": g["attack_type"].iloc[0],
            "velocity_mode": g["velocity_mode"].iloc[0], "attacked": attacked,
            "alarms": refractory_count(hit["window_end_s"]),
            "detected": bool(len(after)),
            "delay_s": float(after["seconds_since_onset_end"].iloc[0])
            if len(after) else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def window_metrics(y, s, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(s) > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "windows": len(y), "positives": int(y.sum()),
        "accuracy": (tp + tn) / len(y) if len(y) else np.nan,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": auc(y, s), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tp": int(tp),
    }  # fmt: skip


def poisson_upper(k, hours, level=0.95):
    return float(chi2.ppf(level, 2 * (k + 1)) / 2.0 / hours) if hours > 0 else np.nan


# -------------------------------------------------------------- protocols --


def run_protocol(name, params_grid, orig, syn, cols, hours, held_out=None):
    tr, va, clean_va, o_test, s_test = protocol_sets(orig, syn, held_out)
    params, tuning = tune(name, params_grid, tr, va, cols)
    final = fit(name, params, tr, cols)
    thr, pool_h, allowed = threshold_for(
        clean_pool(name, params, tr, clean_va, final, cols)
    )
    so, ss = score(name, final, o_test, cols), score(name, final, s_test, cols)
    proto = "seen" if held_out is None else f"unseen_{held_out}"
    lab = o_test[Y] >= 0
    win = {"protocol": proto, "model": name, "params": json.dumps(params),
           "threshold": thr, "calibration_hours": pool_h,
           "calibration_alarms": allowed,
           **window_metrics(o_test.loc[lab, Y], so[lab.to_numpy()], thr)}  # fmt: skip
    neg = so[(o_test[Y] == 0).to_numpy()]
    for mode in ("position_only", "consistent"):
        m = (
            (s_test["velocity_mode"] == mode)
            & (s_test[Y] == 1)
            & (s_test["attack_type"] != "abrupt_jump")  # the mode is a no-op there
        ).to_numpy()
        y = np.r_[np.ones(m.sum()), np.zeros(len(neg))]
        win[f"auc_syn_{mode}"] = auc(y, np.r_[ss[m], neg])
    flights = pd.concat(
        [latch(o_test, so, thr), latch(s_test, ss, thr)], ignore_index=True
    )
    flights.insert(0, "model", name)
    flights.insert(0, "protocol", proto)
    normal = flights[~flights["attacked"]]
    h = float(hours.reindex(normal["flight_id"]).sum())
    if normal["flight_id"].isin(hours.index).mean() < 1:
        raise SystemExit("normal test flights missing from signals_index.csv")
    k = int(normal["alarms"].sum())
    win.update({"fa_flights": len(normal), "fa_alarms": k, "fa_hours": h,
                "fa_per_hour": k / h if h else np.nan,
                "fa_upper_95": poisson_upper(k, h)})  # fmt: skip
    tuning.insert(0, "model", name)
    tuning.insert(0, "protocol", proto)
    return win, flights, tuning


def summary(windows, flights):
    """One row per model and protocol (unseen = the three held-out missions pooled)."""
    f = flights.assign(
        group=np.where(flights["protocol"] == "seen", "seen", "unseen missions")
    )
    w = windows.assign(
        group=np.where(windows["protocol"] == "seen", "seen", "unseen missions")
    )
    rows = []
    for (grp, model), g in f.groupby(["group", "model"], sort=False):
        ww = w[(w["group"] == grp) & (w["model"] == model)]
        tp, fp, fn, tn = (int(ww[c].sum()) for c in ("tp", "fp", "fn", "tn"))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        real = g[g["attacked"] & (g["origin"] == "orig")]
        po = g[g["attacked"] & (g["velocity_mode"] == "position_only")]
        cv = g[g["attacked"] & (g["velocity_mode"] == "consistent")]
        k, h = int(ww["fa_alarms"].sum()), float(ww["fa_hours"].sum())
        rows.append({
            "protocol": grp, "model": model,
            "accuracy": (tp + tn) / (tp + tn + fp + fn), "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "roc_auc": float(ww["roc_auc"].mean()),
            "auc_syn_position_only": float(ww["auc_syn_position_only"].mean()),
            "auc_syn_consistent": float(ww["auc_syn_consistent"].mean()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "real_detected": float(real["detected"].mean()),
            "real_delay_s": float(real.loc[real["detected"], "delay_s"].median()),
            "position_only_detected": float(po["detected"].mean()),
            "consistent_detected": float(cv["detected"].mean()),
            "fa_alarms": k, "fa_hours": h, "fa_per_hour": k / h if h else np.nan,
            "fa_upper_95": poisson_upper(k, h),
        })  # fmt: skip
    return pd.DataFrame(rows)


def run(models=MODELS, quick=False, out_dir=None):
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())["models"]
    orig, syn = load(OUT_DIR)  # inputs always come from Module 05's own folder
    cols = feature_columns(orig)
    hours = pd.read_csv(SIGNALS_INDEX).set_index("flight_id")["vel_hours"]
    protocols = [None] if quick else [None, *MISSIONS]
    print("=" * 78)
    print("AEROVIGIL - PLAN M5: SIX CLASSICAL MODELS, EVENT LABELS, LATCHED")
    print("=" * 78)
    print(
        f"windows: real {len(orig):,}, synthetic main {len(syn):,}   features "
        f"{len(cols)}   budget {BUDGET_PER_HOUR:g}/h   refractory {REFRACTORY_S:g} s"
    )
    started, wins, flights, tunes = time.time(), [], [], []
    for name in models:
        params_grid = grid(cfg[name])
        if quick:
            params_grid = params_grid[:1]
        for held in protocols:
            w, f, t = run_protocol(name, params_grid, orig, syn, cols, hours, held)
            wins.append(w)
            flights.append(f)
            tunes.append(t)
            print(
                f"  {name:20s} {w['protocol']:16s} AUC {w['roc_auc']:.3f}  "
                f"F1 {w['f1']:.3f}  FA {w['fa_alarms']}/{w['fa_hours']:.2f} h  "
                f"best {w['params']}   [{time.time() - started:.0f} s]"
            )
    windows = pd.DataFrame(wins)
    flights = pd.concat(flights, ignore_index=True)
    windows.to_csv(out / "all_models_window.csv", index=False)
    flights.to_csv(out / "all_models_flight.csv", index=False)
    pd.concat(tunes, ignore_index=True).to_csv(
        out / "all_models_tuning.csv", index=False
    )
    table = summary(windows, flights)
    table_path = out / TABLE.name if (quick or out_dir) else TABLE
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    report(table)
    print(f"\nElapsed {time.time() - started:.0f} s   saved to {out} and {table_path}")
    return windows, flights, table


def report(table):
    pd.set_option("display.width", 250)
    order = {"seen": 1, "unseen missions": 0}  # plan: lead with unseen missions
    t = table.sort_values(["protocol", "model"], key=lambda s: s.map(order).fillna(s))
    show = t[["protocol", "model", "accuracy", "precision", "recall", "f1",
              "roc_auc", "auc_syn_consistent", "real_detected", "real_delay_s",
              "consistent_detected", "fa_alarms", "fa_hours",
              "fa_per_hour"]]  # fmt: skip
    print("\nWINDOW METRICS (event labels, real test windows, at the latch threshold)")
    print("and FLIGHT METRICS (latched); unseen missions = three held-out missions")
    print(show.round(3).to_string(index=False))
    print("\nCONFUSION MATRICES (windows)")
    print(t[["protocol", "model", "tn", "fp", "fn", "tp"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--quick", action="store_true", help="first grid point, seen only")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(tuple(a.models.split(",")), a.quick, a.output_dir)


if __name__ == "__main__":
    main()
