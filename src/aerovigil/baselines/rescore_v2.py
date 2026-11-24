r"""AEROVIGIL Module 05 v2: method, feature or data? The three causes, separated.

Three things are varied on identical windows and an identical flight split, so the
three causes can be told apart:

  feature set   gps = the 22 Module 05 features, all = those plus the 14 IMU ones
  task          window = the Module 05 label (the 256 s curved plateau counts as 102
                attack windows), event = ramp or snapback against clean
  sources       orig = the real attacks only, orig+syn = plus the synthetic main set,
                which is the only training data containing velocity-consistent attacks
  scoring       per window (AUC), and per flight after latching the alarm

The flight-level scoring is the operational one: the threshold is set on the normal
VALIDATION flights to allow a fixed number of alarms per hour, the alarm latches once
it fires, and the test set reports what share of attacked flights are caught, how many
seconds after onset, and the alarm rate actually achieved on the normal test flights.

Models are fixed, not tuned, so every number here is a lower bound; Module 05 tunes.

Needs build_window_tables_v2.py. Writes data/processed/module05_v2/rescore_window.csv
and rescore_flight.csv.

Usage:
    python src\aerovigil\baselines\rescore_v2.py
    python src\aerovigil\baselines\rescore_v2.py --alarms-per-hour 1.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aerovigil.baselines.build_window_tables_v2 import IMU_PREFIX, OUT_DIR
from aerovigil.windowing.make_windows_v2 import SECONDS, STEP_S

SEED = 42
TASKS = ("window", "event")
FEATURE_SETS = ("gps", "all")
TRAIN_SOURCES = ("orig", "orig+syn")


def models():
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=SEED,
            ),
        ),
    }


def summary_columns(df, feature_set):
    cols = [c for c in df.columns if "__" in c]
    if feature_set == "gps":
        cols = [c for c in cols if not c.startswith(IMU_PREFIX)]
    return cols


def main_set(frame):
    """Drop the drift sweep: its rates are chosen to be stealthy, not sampled."""
    return frame[frame["attack_set"] == "main"]


def fit_and_score(orig, syn, task, feature_set, name, model, sources):
    """Train on the real flights, optionally plus the synthetic main set."""
    y = f"y_{task}"
    cols = summary_columns(orig, feature_set)
    tr = orig[(orig["split"] == "train") & (orig[y] >= 0)]
    if sources == "orig+syn" and syn is not None:
        extra = main_set(syn)
        tr = pd.concat(
            [tr, extra[(extra["split"] == "train") & (extra[y] >= 0)]],
            ignore_index=True,
        )
    model.fit(tr[cols].to_numpy("float32"), tr[y].to_numpy())
    te = orig[(orig["split"] == "test") & (orig[y] >= 0)]
    row = {
        "task": task, "features": feature_set, "model": name, "sources": sources,
        "train_windows": len(tr), "train_positive": int(tr[y].sum()),
        "test_windows": len(te), "auc_orig": auc_of(model, te, cols, te[y]),
    }  # fmt: skip
    if syn is not None:
        neg = orig[(orig["split"] == "test") & (orig[y] == 0)]
        for mode in ("position_only", "consistent"):
            pos = main_set(syn)
            pos = pos[
                (pos["split"] == "test")
                & (pos[y] == 1)
                & (pos["velocity_mode"] == mode)
                & (pos["attack_type"] != "abrupt_jump")  # mode is a no-op there
            ]
            both = pd.concat([pos, neg], ignore_index=True)
            row[f"auc_syn_{mode}"] = auc_of(model, both, cols, both[y])
    return row


def auc_of(model, frame, cols, y):
    if len(frame) == 0 or y.nunique() < 2:
        return np.nan
    p = model.predict_proba(frame[cols].to_numpy("float32"))[:, 1]
    return float(roc_auc_score(y, p))


def hours(frame):
    """Flight hours represented by a set of windows, counting the hop once each."""
    return len(frame) * STEP_S / 3600.0


def latch_threshold(model, orig, cols, alarms_per_hour):
    """Threshold from the normal VALIDATION flights: the model never trained on them,
    so its scores there are honest. Training normals would set it far too low."""
    pool = orig[(orig["split"] == "validation") & (orig["attack_type"] == "none")]
    p = model.predict_proba(pool[cols].to_numpy("float32"))[:, 1]
    allowed = max(1, round(alarms_per_hour * hours(pool)))
    thr = float(np.sort(p)[-allowed])
    return thr, len(pool), hours(pool), allowed


def latch_evaluate(model, orig, cols, thr):
    """Per test flight: first alarm, whether it lands after onset, and the delay."""
    te = orig[orig["split"] == "test"].copy()
    te["score"] = model.predict_proba(te[cols].to_numpy("float32"))[:, 1]
    te["alarm"] = te["score"] > thr
    rows = []
    for fid, g in te.groupby("flight_id", sort=False):
        g = g.sort_values("window_start_s")
        attacked = g["attack_type"].iloc[0] != "none"
        hit = g[g["alarm"]]
        after = hit[hit["seconds_since_onset_end"] >= 0] if attacked else hit.iloc[:0]
        rows.append({
            "flight_id": fid, "mission_type": g["mission_type"].iloc[0],
            "attacked": attacked, "windows": len(g), "alarms": int(g["alarm"].sum()),
            "detected": bool(len(after)) if attacked else False,
            "delay_s": (
                float(after["seconds_since_onset_end"].iloc[0])
                if len(after) else np.nan
            ),
            "false_alarm_before_onset": bool(
                attacked and (hit["seconds_since_onset_end"] < 0).any()
            ),
        })  # fmt: skip
    flights = pd.DataFrame(rows)
    normal = te[te["attack_type"] == "none"]
    rate = float(normal["alarm"].sum() / hours(normal)) if len(normal) else np.nan
    return flights, rate


def run(alarms_per_hour=1.0, out_dir=None):
    out = Path(out_dir) if out_dir else OUT_DIR
    orig = pd.read_parquet(out / "windows_orig.parquet")
    syn_path = out / "windows_syn.parquet"
    syn = pd.read_parquet(syn_path) if syn_path.is_file() else None
    print("=" * 70)
    print("AEROVIGIL - MODULE 05 v2 RE-SCORING (method, features, data)")
    print("=" * 70)
    print(
        f"Original windows {len(orig):,}"
        + (f"   synthetic {len(syn):,}" if syn is not None else "")
    )
    print(
        f"Alarm budget: {alarms_per_hour:g} per hour   "
        f"window {SECONDS:g} s, hop {STEP_S:g} s\n"
    )
    window_rows, flight_rows, fitted = [], [], {}
    source_sets = TRAIN_SOURCES if syn is not None else ("orig",)
    for task in TASKS:
        for fs in FEATURE_SETS:
            for src in source_sets:
                for name, model in models().items():
                    r = fit_and_score(orig, syn, task, fs, name, model, src)
                    window_rows.append(r)
                    fitted[(task, fs, src, name)] = (model, summary_columns(orig, fs))
                    print(
                        f"  fitted {task:6s} {fs:3s} {src:8s} {name:20s} "
                        f"auc_orig={r['auc_orig']:.3f}"
                    )
    wtab = pd.DataFrame(window_rows)
    wtab.to_csv(out / "rescore_window.csv", index=False)
    for (task, fs, src, name), (model, cols) in fitted.items():
        thr, n_pool, pool_h, allowed = latch_threshold(
            model, orig, cols, alarms_per_hour
        )
        flights, rate = latch_evaluate(model, orig, cols, thr)
        flights.insert(0, "sources", src)
        flights.insert(0, "features", fs)
        flights.insert(0, "model", name)
        flights.insert(0, "task", task)
        flight_rows.append(flights)
        att = flights[flights["attacked"]]
        print(
            f"  latched {task:6s} {fs:3s} {src:8s} {name:20s} thr={thr:.3f} "
            f"({n_pool:,} validation normal windows, {pool_h:.1f} h, "
            f"{allowed} alarms allowed) -> detected {att['detected'].mean():.2f}, "
            f"{rate:.2f} alarms/h on test normals"
        )
    ftab = pd.concat(flight_rows, ignore_index=True)
    ftab.to_csv(out / "rescore_flight.csv", index=False)
    report(wtab, ftab, orig, syn, out)
    return wtab, ftab


def share(frame, key, values, agg, digits=3):
    table = frame.pivot_table(
        index=key, columns="mission_type", values=values, aggfunc=agg
    )
    return table.round(digits).to_string()


def report(wtab, ftab, orig, syn, out):
    pd.set_option("display.width", 220)
    print("\nWINDOW-LEVEL ROC-AUC ON THE TEST SPLIT")
    cols = [
        c
        for c in ("auc_orig", "auc_syn_position_only", "auc_syn_consistent")
        if c in wtab
    ]
    print(
        wtab.pivot_table(
            index=["task", "sources", "model"], columns="features", values=cols
        )
        .round(3)
        .to_string()
    )
    print(
        "\n  task 'window' is the Module 05 label; 'event' is ramp and snapback only."
    )
    print("  features 'gps' is the Module 05 set; 'all' adds the 14 IMU features.")
    if syn is not None:
        print("  syn columns use the main set only and leave out abrupt_jump,")
        print("  where velocity_mode is a no-op; the drift sweep is excluded.")
    att = ftab[ftab["attacked"]]
    key = ["task", "features", "sources", "model"]
    print("\nFLIGHT-LEVEL DETECTION AFTER LATCHING (share of attacked test flights)")
    print(share(att, key, "detected", "mean"))
    print("\nMEDIAN SECONDS FROM ATTACK ONSET TO THE FIRST ALARM")
    print(share(att, key, "delay_s", "median", 1))
    print("\nALARMS ON THE NORMAL TEST FLIGHTS (per hour) AND PRE-ONSET FALSE ALARMS")
    agg = ftab[~ftab["attacked"]].groupby(key)[["alarms", "windows"]].sum()
    rate = agg["alarms"] / (agg["windows"] * STEP_S / 3600.0)
    early = att.groupby(key)["false_alarm_before_onset"].mean()
    print(
        pd.DataFrame(
            {"alarms_per_hour": rate.round(2), "pre_onset_share": early.round(3)}
        ).to_string()
    )
    print(f"\nTables: {out / 'rescore_window.csv'} and {out / 'rescore_flight.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarms-per-hour", type=float, default=1.0)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.alarms_per_hour, a.output_dir)


if __name__ == "__main__":
    main()
