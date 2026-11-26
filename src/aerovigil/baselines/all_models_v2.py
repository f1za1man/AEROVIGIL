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

Protocols (Step B ran the first two; --full adds the rest for plan M9):
    seen       train / validation / test flights as split by seed 42; every mission
               appears in training
    mission    leave one mission out: tune, train and calibrate on the other two
               missions, then test on every flight of the held-out mission
    type       leave one attack family out of training and validation; test on that
               family (and the normal test flights, for false alarms)
    cvfa       six folds over all 144 normal flights, stratified by mission: train
               and calibrate without fold k (and without any synthetic copy of a fold-k
               flight), count false alarms on fold k

Seeds: the decision tree, random forest and Isolation Forest have randomness. Each is
tuned once at seed 42, then refitted and recalibrated with five seeds (42-46) and every
metric is reported as mean and standard deviation over them. The other three models
are deterministic and run once.

Reported per model and protocol
    window level (event labels, at the latch threshold): accuracy, precision, recall,
        F1, ROC-AUC and the confusion matrix (TN, FP, FN, TP), plus the ROC-AUC against
        position-only and velocity-consistent synthetic attacks
    flight level: share of attacked flights caught, median delay from onset, and false
        alarms on the normal flights with count, hours and Poisson 95% upper bound;
        hours come from timestamps (Module 10's monitored time, signals_index.csv)

Writes data/processed/module05_v2/all_models_window.csv, all_models_flight.csv,
all_models_tuning.csv and reports/tables/B_classical_models.csv; with --full the same
files with a _full suffix and reports/tables/C_classical_models.csv

Usage:
    python src\aerovigil\baselines\all_models_v2.py
    python src\aerovigil\baselines\all_models_v2.py --full
    python src\aerovigil\baselines\all_models_v2.py --models naive_bayes,knn --quick
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from aerovigil.baselines.build_window_tables_v2 import OUT_DIR
from aerovigil.baselines.models import (
    STOCHASTIC,
    SUPERVISED,
    UNSUPERVISED,
    build_model,
    fit_model,
    grid,
    score_model,
)
from aerovigil.evaluation.metrics import (
    poisson_upper,
    refractory_count,
    roc_auc,
    window_metrics,
)
from aerovigil.fusion.fuse import folds_of
from aerovigil.windowing.make_windows_v2 import STEP_S

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "baselines" / "baselines.yaml"
SIGNALS_INDEX = ROOT / "data" / "processed" / "module10" / "signals_index.csv"
TABLE = ROOT / "reports" / "tables" / "B_classical_models.csv"
TABLE_FULL = ROOT / "reports" / "tables" / "C_classical_models.csv"
MODELS = SUPERVISED + UNSUPERVISED
MISSIONS = ("curved", "random", "straight")
FAMILIES = ("abrupt_jump", "constant_offset", "linear_drift", "exponential_ramp",
            "low_amplitude_wander", "original")  # fmt: skip
SEED = 42
SEEDS = (42, 43, 44, 45, 46)
FOLDS = 4
CV_FOLDS = 6
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


def source_of(frame):
    return frame["source_flight_id"].fillna(frame["flight_id"])


def protocol_sets(orig, syn, held_out=None, protocol=None, fold_of=None):
    """(train, validation, clean-validation, test-orig, test-syn) for one protocol.

    protocol None or "seen" / "mission" takes held_out as a mission (None = seen);
    "type" takes it as an attack family, "cvfa" as a fold number.
    """
    protocol = protocol or ("seen" if held_out is None else "mission")
    if protocol == "mission":
        o_in = orig[orig["mission_type"] != held_out]
        s_in = syn[syn["mission_type"] != held_out]
        o_test = orig[orig["mission_type"] == held_out]
        s_test = syn[syn["mission_type"] == held_out]
    elif protocol == "cvfa":
        out_o = source_of(orig).map(fold_of) == int(held_out)
        out_s = source_of(syn).map(fold_of) == int(held_out)
        o_in, s_in = orig[~out_o], syn[~out_s]
        o_test = orig[out_o & (orig["attack_type"] == "none")]
        s_test = syn.iloc[:0]
    else:
        o_in = orig[orig["split"] != "test"]
        s_in = syn[syn["split"] != "test"]
        o_test = orig[orig["split"] == "test"]
        s_test = syn[syn["split"] == "test"]
        if protocol == "type":
            if held_out == "original":
                o_in = o_in[~((o_in["attack_type"] == "original") & (o_in[Y] == 1))]
                s_test = s_test.iloc[:0]
            else:
                s_in = s_in[s_in["attack_type"] != held_out]
                s_test = s_test[s_test["attack_type"] == held_out]
                o_test = o_test[o_test["attack_type"] == "none"]
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
    return roc_auc(y, s)


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


def clean_pool(name, params, tr, clean_va, final, cols, seed=SEED):
    """Honest scores on clean windows: out-of-fold on the training flights' normal
    windows, grouped by source flight, plus the validation normal windows."""
    groups = source_of(tr).to_numpy()
    normal = (tr["attack_type"] == "none").to_numpy()
    parts = []
    n = min(FOLDS, len(np.unique(groups)))
    if n >= 2 and normal.any():
        for a, b in GroupKFold(n_splits=n).split(tr, groups=groups):
            held = tr.iloc[b][normal[b]]
            if len(held):
                pipe = fit(name, params, tr.iloc[a], cols, seed)
                parts.append(score(name, pipe, held, cols))
    parts.append(score(name, final, clean_va, cols))
    return np.concatenate(parts)


def threshold_for(pool, budget=BUDGET_PER_HOUR):
    """Score the clean pool may reach budget x hours times. Hours: one hop per window,
    since the windows tile the timestamps at STEP_S seconds."""
    hours = len(pool) * STEP_S / 3600.0
    allowed = max(1, int(round(budget * hours)))
    return float(np.sort(pool)[-allowed]), hours, allowed


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
            "alarms": refractory_count(hit["window_end_s"], REFRACTORY_S),
            "detected": bool(len(after)),
            "delay_s": float(after["seconds_since_onset_end"].iloc[0])
            if len(after) else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


# -------------------------------------------------------------- protocols --


def protocol_name(protocol, held):
    return "seen" if protocol == "seen" else f"{protocol}_{held}"


def evaluate_seed(name, params, sets, cols, hours, protocol, held, seed):
    """Fit, calibrate and score one model at one seed on one protocol."""
    tr, va, clean_va, o_test, s_test = sets
    final = fit(name, params, tr, cols, seed)
    thr, pool_h, allowed = threshold_for(
        clean_pool(name, params, tr, clean_va, final, cols, seed)
    )
    so, ss = score(name, final, o_test, cols), score(name, final, s_test, cols)
    proto = protocol_name(protocol, held)
    lab = (o_test[Y] >= 0).to_numpy()
    win = {"protocol": proto, "model": name, "seed": seed,
           "params": json.dumps(params), "threshold": thr,
           "calibration_hours": pool_h, "calibration_alarms": allowed}  # fmt: skip
    if lab.any() and protocol != "cvfa":
        win.update(window_metrics(o_test.loc[lab, Y], so[lab], thr))
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
    flights.insert(0, "seed", seed)
    flights.insert(0, "model", name)
    flights.insert(0, "protocol", proto)
    normal = flights[~flights["attacked"]]
    if normal["flight_id"].isin(hours.index).mean() < 1:
        raise SystemExit("normal test flights missing from signals_index.csv")
    h = float(hours.reindex(normal["flight_id"]).sum())
    k = int(normal["alarms"].sum())
    win.update({"fa_flights": len(normal), "fa_alarms": k, "fa_hours": h,
                "fa_per_hour": k / h if h else np.nan,
                "fa_upper_95": poisson_upper(k, h)})  # fmt: skip
    return win, flights


def run_protocol(name, params_grid, orig, syn, cols, hours, held=None,
                 protocol=None, seeds=(SEED,), fold_of=None, params=None):  # fmt: skip
    """Tune once (unless params are given), then evaluate every seed."""
    protocol = protocol or ("seen" if held is None else "mission")
    sets = protocol_sets(orig, syn, held, protocol, fold_of)
    tuning = pd.DataFrame()
    if params is None:
        params, tuning = tune(name, params_grid, sets[0], sets[1], cols)
        tuning.insert(0, "model", name)
        tuning.insert(0, "protocol", protocol_name(protocol, held))
    wins, flights = [], []
    for seed in seeds:
        w, f = evaluate_seed(name, params, sets, cols, hours, protocol, held, seed)
        wins.append(w)
        flights.append(f)
    return wins, pd.concat(flights, ignore_index=True), tuning, params


GROUPS = {"seen": "seen", "mission": "unseen missions", "type": "unseen attack types",
          "cvfa": "cross-validated false alarms"}  # fmt: skip


def group_of(protocol):
    # Step B saved the held-out missions as unseen_<mission>; later runs mission_<m>
    head = protocol.split("_")[0]
    return GROUPS["mission" if head == "unseen" else head]


def summary(windows, flights):
    """Per model, protocol group and seed, then mean and sd over seeds."""
    f = flights.assign(group=flights["protocol"].map(group_of))
    w = windows.assign(group=windows["protocol"].map(group_of))
    per = []
    for (grp, model, seed), g in f.groupby(["group", "model", "seed"], sort=False):
        ww = w[(w["group"] == grp) & (w["model"] == model) & (w["seed"] == seed)]
        row = {"protocol": grp, "model": model, "seed": seed}
        if {"tp", "fp", "fn", "tn"} <= set(ww.columns) and ww["tp"].notna().any():
            tp, fp, fn, tn = (int(ww[c].sum()) for c in ("tp", "fp", "fn", "tn"))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            row.update({
                "accuracy": (tp + tn) / (tp + tn + fp + fn), "precision": prec,
                "recall": rec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                "roc_auc": float(ww["roc_auc"].mean()),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })  # fmt: skip
        row["auc_syn_position_only"] = float(ww["auc_syn_position_only"].mean())
        row["auc_syn_consistent"] = float(ww["auc_syn_consistent"].mean())
        real = g[g["attacked"] & (g["origin"] == "orig")]
        po = g[g["attacked"] & (g["velocity_mode"] == "position_only")]
        cv = g[g["attacked"] & (g["velocity_mode"] == "consistent")]
        att = g[g["attacked"]]
        k, h = int(ww["fa_alarms"].sum()), float(ww["fa_hours"].sum())
        row.update({
            "detected": float(att["detected"].mean()) if len(att) else np.nan,
            "real_detected": float(real["detected"].mean()) if len(real) else np.nan,
            "real_delay_s": float(real.loc[real["detected"], "delay_s"].median())
            if real["detected"].any() else np.nan,
            "position_only_detected": float(po["detected"].mean())
            if len(po) else np.nan,
            "consistent_detected": float(cv["detected"].mean()) if len(cv) else np.nan,
            "fa_alarms": k, "fa_hours": h, "fa_per_hour": k / h if h else np.nan,
            "fa_upper_95": poisson_upper(k, h),
        })  # fmt: skip
        per.append(row)
    per = pd.DataFrame(per)
    keys = ["protocol", "model"]
    vals = [c for c in per.columns if c not in keys + ["seed"]]
    mean = per.groupby(keys, sort=False)[vals].mean()
    sd = per.groupby(keys, sort=False)[vals].std().add_suffix("_sd")
    out = mean.join(sd)
    out["seeds"] = per.groupby(keys, sort=False)["seed"].nunique()
    return out.reset_index(), per


def run(models=MODELS, quick=False, out_dir=None, full=False):
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())["models"]
    orig, syn = load(OUT_DIR)  # inputs always come from Module 05's own folder
    cols = feature_columns(orig)
    hours = pd.read_csv(SIGNALS_INDEX).set_index("flight_id")["vel_hours"]
    normals = orig[orig["attack_type"] == "none"].drop_duplicates("flight_id")
    fold_of = folds_of(normals[["flight_id", "mission_type"]])
    protocols = [("seen", None)]
    if not quick:
        protocols += [("mission", m) for m in MISSIONS]
    if full:
        protocols += [("type", f) for f in FAMILIES]
        protocols += [("cvfa", str(k)) for k in range(CV_FOLDS)]
    print("=" * 78)
    print("AEROVIGIL - PLAN M5: SIX CLASSICAL MODELS, EVENT LABELS, LATCHED")
    print("=" * 78)
    print(
        f"windows: real {len(orig):,}, synthetic main {len(syn):,}   features "
        f"{len(cols)}   budget {BUDGET_PER_HOUR:g}/h   refractory {REFRACTORY_S:g} s"
        f"   protocols {len(protocols)}"
    )
    started, wins, flights, tunes = time.time(), [], [], []
    for name in models:
        params_grid = grid(cfg[name])
        if quick:
            params_grid = params_grid[:1]
        seeds = SEEDS if (full and name in STOCHASTIC) else (SEED,)
        seen_params = None
        for protocol, held in protocols:
            # the false-alarm folds reuse the seen setting: they measure the
            # calibration on unseen normal flights, not the tuning
            given = seen_params if protocol == "cvfa" else None
            w, f, t, params = run_protocol(
                name, params_grid, orig, syn, cols, hours, held, protocol,
                seeds if protocol != "cvfa" else (SEED,), fold_of, given,
            )  # fmt: skip
            if protocol == "seen":
                seen_params = params
            wins += w
            flights.append(f)
            tunes.append(t)
            last = w[-1]
            print(
                f"  {name:20s} {last['protocol']:28s} seeds {len(w)}  "
                f"AUC {last.get('roc_auc', np.nan):.3f}  "
                f"FA {last['fa_alarms']}/{last['fa_hours']:.2f} h  "
                f"best {last['params']}   [{time.time() - started:.0f} s]",
                flush=True,
            )
    windows = pd.DataFrame(wins)
    flights = pd.concat(flights, ignore_index=True)
    suffix = "_full" if full else ""
    windows.to_csv(out / f"all_models_window{suffix}.csv", index=False)
    flights.to_csv(out / f"all_models_flight{suffix}.csv", index=False)
    pd.concat(tunes, ignore_index=True).to_csv(
        out / f"all_models_tuning{suffix}.csv", index=False
    )
    table, per = summary(windows, flights)
    per.to_csv(out / f"all_models_per_seed{suffix}.csv", index=False)
    target = TABLE_FULL if full else TABLE
    table_path = out / target.name if (quick or out_dir) else target
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    report(table)
    print(f"\nElapsed {time.time() - started:.0f} s   saved to {out} and {table_path}")
    return windows, flights, table


def report(table):
    pd.set_option("display.width", 250)
    order = {
        "unseen missions": 0,
        "unseen attack types": 1,
        "seen": 2,
        "cross-validated false alarms": 3,
    }  # plan: lead with unseen missions
    t = table.sort_values(
        ["protocol", "model"], key=lambda s: s.map(order) if s.name == "protocol" else s
    )  # fmt: skip
    cols = ["protocol", "model", "seeds", "accuracy", "precision", "recall", "f1",
            "roc_auc", "auc_syn_consistent", "detected", "real_detected", "real_delay_s",
            "consistent_detected", "fa_alarms", "fa_hours", "fa_per_hour"]  # fmt: skip
    print("\nWINDOW METRICS (event labels, real test windows, at the latch threshold)")
    print("and FLIGHT METRICS (latched); mean over seeds where a model has them")
    print(t[[c for c in cols if c in t]].round(3).to_string(index=False))
    if "tp" in t:
        print("\nCONFUSION MATRICES (windows, mean over seeds)")
        cm = t[t["tp"].notna()][["protocol", "model", "tn", "fp", "fn", "tp"]]
        print(cm.round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--quick", action="store_true", help="first grid point, seen only")
    ap.add_argument(
        "--full",
        action="store_true",
        help="plan M9: five seeds, unseen attack types and CV false alarms",
    )
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(tuple(a.models.split(",")), a.quick, a.output_dir, a.full)


if __name__ == "__main__":
    main()
