r"""Plan M9 for the LSTM detectors: three protocols, five seeds, ablations, CV alarms.

Module 07 v2 trained the autoencoder and the supervised LSTM on the seen split with
three seeds, and held each attack family out for two seeds. This trains everything the
evaluation framework needs, with five seeds each, and saves the weights (Step D attacks
them):

    seen           the flight-disjoint split (train / validation / calib / test roles)
    mission:<m>    leave one mission out: fit, stop and calibrate on the other two
                   missions, test on every flight of the held-out mission
    type:<f>       the classifier with attack family f left out of training and
                   validation, tested on family f only (the autoencoder never sees an
                   attack, so for it this protocol is the seen one)
    size:<h>_<l>   the seen protocol at another model size (hidden units x layers)
    cvfa:<k>       six folds over all 144 normal flights, stratified by mission: fit
                   and calibrate without fold k (and without any synthetic copy made
                   from a fold-k flight), then count false alarms on fold k

Alarm threshold: 1 alarm per flight-hour on the clean calibration windows of the
protocol's own calibration flights, never a tested flight. Alarms closer than 10 s are
merged, false-alarm hours come from timestamps (Module 10's monitored time) and every
rate carries its count and Poisson 95% bound.

Writes data/processed/module07_v2/protocols/{weights,scores}/<job>.{pt,npy},
protocols/jobs.csv, protocols/flights.csv, protocols/windows.csv and
reports/tables/C_lstm.csv

Usage:
    python src\aerovigil\evaluation\lstm_protocols.py            (train what is missing, then evaluate)
    python src\aerovigil\evaluation\lstm_protocols.py --evaluate-only
    python src\aerovigil\evaluation\lstm_protocols.py --seeds 1 --only seen
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from aerovigil.evaluation.metrics import (
    BUDGET_PER_HOUR,
    false_alarms,
    refractory_count,
    roc_auc,
    window_metrics,
)
from aerovigil.fusion.fuse import folds_of
from aerovigil.lstm.build_sequences_v2 import OUT_DIR, ROOT
from aerovigil.lstm.models import build_model, fit, predict
from aerovigil.lstm.train_v2 import (
    AE_PARAMS,
    CLF_PARAMS,
    FAMILIES,
    MAX_EPOCHS,
    PATIENCE,
    THREADS,
    load,
)
from aerovigil.windowing.make_windows_v2 import STEP_S

SEEDS = (42, 43, 44, 45, 46)
MISSIONS = ("curved", "random", "straight")
BASE_SIZE = (32, 1)
SIZES = ((64, 1), (32, 2), (64, 2))
CV_FOLDS = 6
CV_SEED = 42
PDIR = OUT_DIR / "protocols"
SIGNALS_INDEX = ROOT / "data" / "processed" / "module10" / "signals_index.csv"
TABLE = ROOT / "reports" / "tables" / "C_lstm.csv"


# ------------------------------------------------------------------- jobs --


def job_list(seeds=SEEDS, only=None):
    """(name, kind, protocol, held, hidden, layers, seed) for every model to train."""
    jobs = []

    def add(kind, protocol, held, size, seed):
        h, layer = size
        tag = protocol if held is None else f"{protocol}-{held}"
        if size != BASE_SIZE:
            tag = f"size-h{h}l{layer}"
        jobs.append({"name": f"{kind}_{tag}_s{seed}", "kind": kind,
                     "protocol": protocol, "held": held, "hidden": h,
                     "layers": layer, "seed": seed})  # fmt: skip

    for seed in seeds:
        for kind in ("ae", "clf"):
            add(kind, "seen", None, BASE_SIZE, seed)
            for m in MISSIONS:
                add(kind, "mission", m, BASE_SIZE, seed)
            for size in SIZES:
                add(kind, "size", None, size, seed)
        for fam in FAMILIES:
            add("clf", "type", fam, BASE_SIZE, seed)
    for k in range(CV_FOLDS):
        for kind in ("ae", "clf"):
            add(kind, "cvfa", str(k), BASE_SIZE, CV_SEED)
    if only:
        jobs = [j for j in jobs if j["protocol"] in only]
    return jobs


# ------------------------------------------------------------------ masks --


def cv_fold_of(meta):
    """Fold of every normal original flight, the same folds Module 10 uses."""
    normal = meta[(meta["origin"] == "orig") & (meta["attack_type"] == "none")]
    flights = normal.drop_duplicates("flight_id")[["flight_id", "mission_type"]]
    return folds_of(flights)


def masks(meta, job, fold_of=None):
    """(train, validation, calibration, test) window masks for one job."""
    orig = meta["origin"] == "orig"
    clean = orig & (meta["y_event"] == 0)
    normal_clean = clean & (meta["attack_type"] == "none")
    role = meta["role"]
    if job["kind"] == "ae":
        fit_set = normal_clean
    else:
        real = orig & (meta["y_event"] == 1)
        syn = (
            (meta["origin"] == "syn")
            & (meta["attack_set"] == "main")
            & (meta["y_event"] == 1)
        )
        if job["protocol"] == "type":
            if job["held"] == "original":
                real = real & False
            else:
                syn = syn & (meta["attack_type"] != job["held"])
        fit_set = clean | real | syn
    tr, va = fit_set & (role == "train"), fit_set & (role == "validation")
    cal = normal_clean & (role == "calib")
    test = role == "test"
    if job["protocol"] == "mission":
        keep = meta["mission_type"] != job["held"]
        tr, va, cal = tr & keep, va & keep, cal & keep
        test = ~keep
    elif job["protocol"] == "cvfa":
        fold = meta["source_flight_id"].fillna(meta["flight_id"]).map(fold_of)
        out = fold == int(job["held"])
        tr, va, cal = tr & ~out, va & ~out, cal & ~out
        test = out & orig & (meta["attack_type"] == "none")
    return tuple(np.asarray(m, bool) for m in (tr, va, cal, test))


# --------------------------------------------------------------- training --


def params_for(job):
    base = dict(AE_PARAMS if job["kind"] == "ae" else CLF_PARAMS)
    base["hidden"], base["layers"] = int(job["hidden"]), int(job["layers"])
    return base


def train_one(job, x, meta, fold_of, wdir, sdir):
    tr, va, _, _ = masks(meta, job, fold_of)
    y = meta["y_event"].to_numpy().astype(np.float32)
    yt, yv = (None, None) if job["kind"] == "ae" else (y[tr], y[va])
    params = params_for(job)
    model, hist = fit(
        job["kind"],
        build_model(job["kind"], x.shape[-1], params, job["seed"]),
        x[tr], yt, x[va], yv, params, job["seed"], MAX_EPOCHS, PATIENCE,
    )  # fmt: skip
    score, _ = predict(job["kind"], model, x)
    # write beside the target and rename, so an interrupted run never leaves a
    # half-written file that the next run would take as finished
    for path, save in (
        (wdir / f"{job['name']}.pt", lambda p: torch.save(model.state_dict(), p)),
        (sdir / f"{job['name']}.npy",
         lambda p: np.save(p, np.asarray(score, np.float32))),
    ):  # fmt: skip
        tmp = path.with_name(path.stem + ".partial" + path.suffix)
        save(tmp)
        os.replace(tmp, path)
    return {**job, "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "best_epoch": hist["best_epoch"], "best_val": hist["best_val"]}  # fmt: skip


def train(jobs, x, meta, fold_of):
    wdir, sdir = PDIR / "weights", PDIR / "scores"
    wdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    for stale in list(wdir.glob("*.partial.*")) + list(sdir.glob("*.partial.*")):
        stale.unlink()
    torch.set_num_threads(THREADS)
    index_path = PDIR / "jobs.csv"
    done = pd.read_csv(index_path) if index_path.is_file() else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    have = {r["name"] for r in rows}
    todo = [
        j for j in jobs
        if not ((sdir / f"{j['name']}.npy").is_file()
                and (wdir / f"{j['name']}.pt").is_file() and j["name"] in have)
    ]  # fmt: skip
    print(
        f"Jobs {len(jobs)}: {len(jobs) - len(todo)} already trained, {len(todo)} to go"
    )
    started = time.time()
    for n, job in enumerate(todo, start=1):
        rows = [r for r in rows if r["name"] != job["name"]]
        rows.append(train_one(job, x, meta, fold_of, wdir, sdir))
        pd.DataFrame(rows).to_csv(index_path, index=False)
        print(f"  [{n}/{len(todo)}] {job['name']:28s} epoch "
              f"{rows[-1]['best_epoch']:2d}  ({time.time() - started:.0f} s)",
              flush=True)  # fmt: skip
    return pd.DataFrame(rows)


# ------------------------------------------------------------- evaluation --


def threshold_for(cal_scores, budget=BUDGET_PER_HOUR):
    hours = len(cal_scores) * STEP_S / 3600.0
    allowed = max(1, int(round(budget * hours)))
    return float(np.sort(cal_scores)[-allowed]), hours, allowed


def latch(meta, score, test, thr, hours):
    """Per tested flight: detection after onset, delay, displacement, alarms."""
    te = meta.loc[test].assign(score=score[test])
    te["alarm"] = te["score"] > thr
    rows = []
    for fid, g in te.groupby("flight_id", sort=False):
        attacked = g["attack_type"].iloc[0] != "none"
        hit = g[g["alarm"]].sort_values("window_end_s")
        after = hit[hit["seconds_since_onset_end"] >= 0] if attacked else hit.iloc[:0]
        rows.append({
            "flight_id": fid, "origin": g["origin"].iloc[0],
            "attack_set": g["attack_set"].iloc[0],
            "mission_type": g["mission_type"].iloc[0],
            "attack_type": g["attack_type"].iloc[0],
            "velocity_mode": g["velocity_mode"].iloc[0],
            "drift_rate_mps": g["drift_rate_mps"].iloc[0], "attacked": attacked,
            "alarms": refractory_count(hit["window_end_s"]),
            "hours": float(hours.get(fid, np.nan)) if not attacked else np.nan,
            "detected": bool(len(after)),
            "delay_s": float(after["seconds_since_onset_end"].iloc[0])
            if len(after) else np.nan,
            "displacement_m": float(after["offset_end_m"].iloc[0])
            if len(after) else np.nan,
            "final_offset_m": float(g["max_offset_m"].max()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def flight_hours():
    """Monitored hours of every original flight (Module 10's timestamp hours)."""
    return pd.read_csv(SIGNALS_INDEX).set_index("flight_id")["vel_hours"].to_dict()


def evaluate_job(job, score, meta, fold_of, hours, threshold=None):
    """(per-flight frame, window record) for one job's scores on every window.

    The threshold is calibrated on the job's calibration windows unless one is given
    (Step F uses that to keep a full-precision threshold for a quantised model)."""
    _, _, cal, test = masks(meta, job, fold_of)
    thr, cal_h, allowed = threshold_for(score[cal])
    if threshold is not None:
        thr = float(threshold)
    f = latch(meta, score, test, thr, hours)
    if job["protocol"] == "type":  # scored on the held-out family only
        f = f[(f["attack_type"] == job["held"]) | ~f["attacked"]]
    meta_cols = {k: job[k] for k in ("name", "kind", "protocol", "held",
                                     "hidden", "layers", "seed")}  # fmt: skip
    f = f.assign(**meta_cols, threshold=thr)
    lab = (
        test & (meta["origin"] == "orig").to_numpy() & (meta["y_event"] >= 0).to_numpy()
    )
    w = {**meta_cols, "threshold": thr, "calibration_hours": cal_h,
         "calibration_alarms": allowed}  # fmt: skip
    if job["protocol"] != "cvfa":
        w.update(window_metrics(meta["y_event"].to_numpy()[lab], score[lab], thr))
        neg = score[lab & (meta["y_event"] == 0).to_numpy()]
        for mode in ("position_only", "consistent"):
            pos = test & (
                (meta["origin"] == "syn") & (meta["attack_set"] == "main")
                & (meta["y_event"] == 1) & (meta["velocity_mode"] == mode)
                & (meta["attack_type"] != "abrupt_jump")
            ).to_numpy()  # fmt: skip
            if job["protocol"] == "type":
                pos &= (meta["attack_type"] == job["held"]).to_numpy()
            y = np.r_[np.ones(pos.sum()), np.zeros(len(neg))]
            w[f"auc_syn_{mode}"] = roc_auc(y, np.r_[score[pos], neg])
    return f, w


def evaluate(jobs, meta, fold_of):
    sdir = PDIR / "scores"
    hours = flight_hours()
    flights, windows = [], []
    for _, job in jobs.iterrows():
        path = sdir / f"{job['name']}.npy"
        if not path.is_file():
            continue
        f, w = evaluate_job(job, np.load(path), meta, fold_of, hours)
        flights.append(f)
        windows.append(w)
    flights = pd.concat(flights, ignore_index=True)
    windows = pd.DataFrame(windows)
    flights.to_csv(PDIR / "flights.csv", index=False)
    windows.to_csv(PDIR / "windows.csv", index=False)
    return flights, windows


def protocol_group(row):
    if row["protocol"] == "mission":
        return "unseen missions"
    if row["protocol"] == "type":
        return "unseen attack types"
    if row["protocol"] == "size":
        return f"size h{row['hidden']} l{row['layers']}"
    if row["protocol"] == "cvfa":
        return "cross-validated false alarms"
    return "seen"


def summarise(flights, windows):
    """Per detector and protocol: per-seed metrics, then mean and sd over seeds."""
    det = {"ae": "lstm_ae", "clf": "lstm_supervised"}
    f = flights.assign(group=flights.apply(protocol_group, axis=1))
    w = windows.assign(group=windows.apply(protocol_group, axis=1))
    per_seed = []
    for (grp, kind, seed), g in f.groupby(["group", "kind", "seed"]):
        main = g[g["attack_set"] != "sweep"]
        real = main[main["attacked"] & (main["origin"] == "orig")]
        syn = main[main["attacked"] & (main["origin"] == "syn")]
        nor = main[~main["attacked"]]
        ww = w[(w["group"] == grp) & (w["kind"] == kind) & (w["seed"] == seed)]
        row = {"protocol": grp, "detector": det[kind], "seed": seed,
               "real_detected": real["detected"].mean() if len(real) else np.nan,
               "real_delay_s": real.loc[real["detected"], "delay_s"].median(),
               "syn_detected": syn["detected"].mean() if len(syn) else np.nan,
               "consistent_detected": syn.loc[
                   syn["velocity_mode"] == "consistent", "detected"].mean(),
               **false_alarms(nor["alarms"], nor["hours"].fillna(0))}  # fmt: skip
        tp, fp, fn, tn = (ww[c].sum() for c in ("tp", "fp", "fn", "tn"))
        if tp + fp + fn + tn:
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            row.update({
                "accuracy": (tp + tn) / (tp + fp + fn + tn), "precision": prec,
                "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                "roc_auc": ww["roc_auc"].mean(),
                "auc_syn_consistent": ww["auc_syn_consistent"].mean(),
            })  # fmt: skip
        per_seed.append(row)
    per = pd.DataFrame(per_seed)
    vals = [c for c in per.columns if c not in ("protocol", "detector", "seed")]
    agg = per.groupby(["protocol", "detector"])[vals].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["seeds"] = per.groupby(["protocol", "detector"])["seed"].nunique()
    return agg.reset_index(), per


def run(seeds=SEEDS, only=None, evaluate_only=False):
    x, meta = load(OUT_DIR)
    fold_of = cv_fold_of(meta)
    jobs = job_list(seeds, only)
    print("=" * 78)
    print("AEROVIGIL - PLAN M9: LSTM PROTOCOLS (seen / unseen missions / unseen types)")
    print("=" * 78)
    print(f"Windows {len(x):,}   seeds {list(seeds)}   jobs {len(jobs)}")
    if not evaluate_only:
        train(jobs, x, meta, fold_of)
    flights, windows = evaluate(pd.DataFrame(jobs), meta, fold_of)
    table, per = summarise(flights, windows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    per.to_csv(PDIR / "per_seed.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["protocol", "detector", "seeds", "real_detected_mean", "real_detected_std",
            "consistent_detected_mean", "f1_mean", "roc_auc_mean", "fa_alarms_mean",
            "fa_hours_mean", "fa_per_hour_mean"]  # fmt: skip
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
    print(f"\nSaved {TABLE}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="use the first N seeds")
    ap.add_argument(
        "--only",
        default=None,
        help="comma list of protocols: seen,mission,type,size,cvfa",
    )
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    seeds = SEEDS[: a.seeds] if a.seeds else SEEDS
    only = tuple(a.only.split(",")) if a.only else None
    run(seeds, only, a.evaluate_only)


if __name__ == "__main__":
    main()
