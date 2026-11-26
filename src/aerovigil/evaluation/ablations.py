r"""Plan M9 ablations for the classical models: sensors and window length.

What each input is worth, measured by taking it away. The model, its tuned setting and
the scoring are those of all_models_v2.py (event labels, latch, 1 alarm per flight-hour,
10 s refractory), on the seen split and on the three held-out missions. Unseen missions
are reported first, as the plan asks.

    full          all 216 window features, 5 s windows (the reference)
    no_baro       without the seven barometer features (GPS altitude minus barometric
                  altitude: res_altitude and its abs, rate and rolling statistics)
    no_mag        without magnetometer features. None exists: Module 03 lists the
                  magnetometer-heading residual as "not generated", so this row is the
                  reference by construction and is checked, not re-run
    window_2s     2 s windows, 1 s hop (windows rebuilt from the same features)
    window_10s    10 s windows, 5 s hop

Models: Naive Bayes and logistic regression (deterministic, one run each) and the
random forest (five seeds). Each
keeps the setting Step B tuned for that protocol on the 5 s windows, so a change here is
the input's effect, not a re-tuning. LSTM model size is ablated in lstm_protocols.py.

Writes data/processed/evaluation/ablation_runs.csv and reports/tables/C_ablations.csv

Usage:
    python src\aerovigil\evaluation\ablations.py
    python src\aerovigil\evaluation\ablations.py --models naive_bayes --quick
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import aerovigil.baselines.all_models_v2 as am
import aerovigil.baselines.build_window_tables_v2 as bw
from aerovigil.baselines.build_window_tables_v2 import OUT_DIR as M05_DIR

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "processed" / "evaluation"
TABLE = ROOT / "reports" / "tables" / "C_ablations.csv"
FEATURE_META = ROOT / "data" / "processed" / "module03_v2" / "feature_metadata.csv"
WINDOWS = {"window_2s": (2.0, 1.0), "window_10s": (10.0, 5.0)}
BARO_FEATURE = "res_altitude"
MODELS = ("naive_bayes", "logistic_regression", "random_forest")


def magnetometer_features():
    """Generated features whose inputs include the magnetometer (expected: none)."""
    meta = pd.read_csv(FEATURE_META)
    uses = meta["source_columns"].fillna("").str.contains("magnetic")
    made = ~meta["formula"].fillna("").str.contains("not generated")
    return meta.loc[uses & made, "feature"].tolist()


def drop_family(cols, base):
    """Window columns built from one base feature (any suffix, any statistic)."""
    return [c for c in cols if c.split("__")[0].startswith(base)]


def windows_for(name):
    """Window tables for one ablation, rebuilding them for a new length once."""
    if name not in WINDOWS:
        return M05_DIR, am.STEP_S
    seconds, hop = WINDOWS[name]
    folder = OUT / f"windows_{name}"
    if not (folder / "windows_syn.parquet").is_file():
        saved = bw.SECONDS, bw.STEP_S
        bw.SECONDS, bw.STEP_S = seconds, hop
        try:
            bw.run(("orig", "syn"), output_dir=folder)
        finally:
            bw.SECONDS, bw.STEP_S = saved
    return folder, hop


def tuned_params():
    """Step B's chosen setting per model and protocol."""
    t = pd.read_csv(M05_DIR / "all_models_tuning.csv")
    best = t.loc[t.groupby(["protocol", "model"])["val_auc"].idxmax()]
    # Step B named the held-out missions unseen_<mission>; here they are mission_<m>
    name = {f"unseen_{m}": f"mission_{m}" for m in am.MISSIONS}
    return {(name.get(r["protocol"], r["protocol"]), r["model"]): json.loads(r["params"])
            for _, r in best.iterrows()}  # fmt: skip


def run(models=MODELS, quick=False):
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mags = magnetometer_features()
    print("=" * 78)
    print("AEROVIGIL - PLAN M9 ABLATIONS (sensors, window length)")
    print("=" * 78)
    print(f"Magnetometer features in use: {mags if mags else 'none'}")
    params = tuned_params()
    hours = pd.read_csv(am.SIGNALS_INDEX).set_index("flight_id")["vel_hours"]
    protocols = [("seen", None)] + ([] if quick else
                                    [("mission", m) for m in am.MISSIONS])  # fmt: skip
    configs = ["full", "no_baro", *WINDOWS]
    if mags:
        configs.insert(2, "no_mag")
    wins, flights = [], []
    saved_hop = am.STEP_S
    try:
        for config in configs:
            folder, hop = windows_for(config)
            am.STEP_S = hop  # the clean pool's hours: one hop per window
            orig, syn = am.load(folder)
            cols = am.feature_columns(orig)
            if config == "no_baro":
                cols = [c for c in cols if c not in drop_family(cols, BARO_FEATURE)]
            if config == "no_mag":
                cols = [c for c in cols if c.split("__")[0] not in mags]
            for name in models:
                seeds = am.SEEDS if name in am.STOCHASTIC else (am.SEED,)
                if quick:
                    seeds = seeds[:1]
                for protocol, held in protocols:
                    key = (am.protocol_name(protocol, held), name)
                    sets = am.protocol_sets(orig, syn, held, protocol)
                    for seed in seeds:
                        w, f = am.evaluate_seed(name, params[key], sets, cols, hours,
                                                protocol, held, seed)  # fmt: skip
                        tag = {"config": config, "features": len(cols),
                               "window_s": WINDOWS.get(config, (5.0,))[0]}  # fmt: skip
                        wins.append({**tag, **w})
                        flights.append(f.assign(**tag))
                    print(f"  {config:11s} {name:14s} {key[0]:16s} seeds {len(seeds)}"
                          f"   [{time.time() - started:.0f} s]", flush=True)  # fmt: skip
    finally:
        am.STEP_S = saved_hop
    windows = pd.DataFrame(wins)
    flights = pd.concat(flights, ignore_index=True)
    if not quick:  # keep earlier full runs of the models not re-run this time
        windows, flights = keep_others(windows, flights, models)
    windows.to_csv(OUT / "ablation_runs.csv", index=False)
    flights.to_csv(OUT / "ablation_flights.csv", index=False)
    table = summarise(windows, flights, mags)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print(f"\nElapsed {time.time() - started:.0f} s   saved {TABLE}")
    return table


def keep_others(windows, flights, models):
    """Add the saved rows of every model that was not re-run (a model's rows are
    replaced as a whole, so nothing is mixed between runs)."""
    runs, fl = OUT / "ablation_runs.csv", OUT / "ablation_flights.csv"
    if not (runs.is_file() and fl.is_file()):
        return windows, flights
    old_w, old_f = pd.read_csv(runs), pd.read_csv(fl)
    old_w = old_w[~old_w["model"].isin(models)]
    old_f = old_f[~old_f["model"].isin(models)]
    return (pd.concat([old_w, windows], ignore_index=True),
            pd.concat([old_f, flights], ignore_index=True))  # fmt: skip


def summarise(windows, flights, mags):
    rows = []
    for config, w in windows.groupby("config", sort=False):
        f = flights[flights["config"] == config]
        table, _ = am.summary(
            w.drop(columns=["config", "features", "window_s"]),
            f.drop(columns=["config", "features", "window_s"]),
        )
        table.insert(0, "features", int(w["features"].iloc[0]))
        table.insert(0, "window_s", float(w["window_s"].iloc[0]))
        table.insert(0, "config", config)
        rows.append(table)
    out = pd.concat(rows, ignore_index=True)
    if not mags:  # the reference by construction: no feature uses the magnetometer
        ref = out[out["config"] == "full"].assign(config="no_mag (none in use)")
        out = pd.concat([out, ref], ignore_index=True)
    keep = ["config", "window_s", "features", "protocol", "model", "seeds", "f1",
            "f1_sd", "roc_auc", "auc_syn_consistent", "real_detected",
            "real_delay_s", "consistent_detected", "fa_alarms", "fa_hours",
            "fa_per_hour", "fa_per_hour_sd"]  # fmt: skip
    order = {"unseen missions": 0, "seen": 1}
    out = out.sort_values(
        ["protocol", "model", "config"],
        key=lambda s: s.map(order) if s.name == "protocol" else s,
    )
    return out[[c for c in keep if c in out]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--quick", action="store_true", help="seen only, one seed")
    a = ap.parse_args()
    run(tuple(a.models.split(",")), a.quick)


if __name__ == "__main__":
    main()
