r"""Rebuild every results table of the project from the saved per-flight results.

Nothing is retrained or recalibrated here: each step reads the per-flight (and
per-window) results its module saved and rebuilds its table, so the tables in reports/
always match the data behind them. The heavy runs that produce those results are:

    python src\aerovigil\baselines\all_models_v2.py            (Step B, ~12 min)
    python src\aerovigil\baselines\all_models_v2.py --full     (Step C, classical)
    python src\aerovigil\evaluation\physics_protocols.py       (Step C, physics)
    python src\aerovigil\evaluation\lstm_protocols.py          (Step C, LSTMs)
    python src\aerovigil\evaluation\ablations.py               (Step C, ablations)
    python src\aerovigil\stealth\frontier.py                   (Step D, ~30 min)
    python src\aerovigil\stealth\adaptive.py --detector <name> (Step D, per detector)
    python src\aerovigil\stealth\adaptive.py --summarise       (Step D, exam)
    python src\aerovigil\stealth\whitebox.py                   (Step D, ~2 h)
    python src\aerovigil\stealth\harden.py --model <name>      (Step D, hardening)
    python src\aerovigil\explain\run.py                        (Step E, ~30 min)
    python src\aerovigil\edge\benchmark.py                     (Step F, ~30 min;
                                                                its F_*.csv tables are
                                                                timings, so it writes
                                                                them itself)

Tables written to reports/tables/
    A_old_vs_new.csv           results before and after the Module 04 regeneration
    B_classical_models.csv     six classical models, seen and unseen missions
    C_classical_models.csv     the same with five seeds, unseen types, CV false alarms
    C_physics.csv              Kalman, velocity, fused, CUSUM, consistency
    C_lstm.csv                 LSTM autoencoder and supervised LSTM, sizes included
    C_ablations.csv            sensors and window length
    C_all_detectors.csv        every detector on one table, unseen missions first
    threat_model_table.csv     (in data/processed/module10) the attacker x detector table
    D_frontier_fixed.csv       Stealth Frontier, fixed drift sweep, every detector
    D_frontier_adaptive.csv    Stealth Frontier, adaptive black-box attacker
    D_whitebox.csv             white-box worst case
    D_hardening.csv            random forest and supervised LSTM before / after hardening
    reports/figures/stealth/   D1-D3 figures
    E_*.csv                    alarm reasons, example messages, SHAP, K-Means
    reports/figures/explain/   E1-E2 figures
    F_*.csv                    edge benchmark: size, one-thread latency, int8, streaming

Usage:
    python src\aerovigil\evaluation\run_all.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import aerovigil.baselines.all_models_v2 as am
from aerovigil.evaluation import ablations, compare_runs, lstm_protocols
from aerovigil.evaluation import physics_protocols as phys

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "reports" / "tables"
M05 = ROOT / "data" / "processed" / "module05_v2"
ORDER = {"unseen missions": 0, "unseen attack types": 1, "seen": 2,
         "cross-validated false alarms": 3}  # fmt: skip
FAMILY = {"logistic_regression": "classical", "naive_bayes": "classical",
          "knn": "classical", "decision_tree": "classical",
          "random_forest": "classical", "isolation_forest": "classical",
          "lstm_ae": "learned sequence", "lstm_supervised": "learned sequence",
          "kalman": "physics", "velocity": "physics", "fused": "physics",
          "cusum": "physics", "consistency": "physics"}  # fmt: skip
COLUMNS = ["protocol", "detector", "family", "seeds", "detected", "detected_spread",
           "real_detected", "real_delay_s", "consistent_detected", "f1", "roc_auc",
           "fa_alarms", "fa_hours", "fa_per_hour", "fa_upper_95"]  # fmt: skip


def read(path):
    if not path.is_file():
        print(f"  missing {path.relative_to(ROOT)}: run its module first")
        return None
    return pd.read_csv(path)


def classical(full):
    suffix = "_full" if full else ""
    w = read(M05 / f"all_models_window{suffix}.csv")
    f = read(M05 / f"all_models_flight{suffix}.csv")
    if w is None or f is None:
        return None
    table, _ = am.summary(w, f)
    table.to_csv(TABLES / ("C_classical_models.csv" if full else
                           "B_classical_models.csv"), index=False)  # fmt: skip
    return table


def physics():
    f = read(phys.OUT / "physics_flights.csv")
    if f is None:
        return None
    table = phys.summarise(f)
    table.to_csv(phys.TABLE, index=False)
    return table


def lstm():
    f = read(lstm_protocols.PDIR / "flights.csv")
    w = read(lstm_protocols.PDIR / "windows.csv")
    if f is None or w is None:
        return None
    table, _ = lstm_protocols.summarise(f, w)
    table.to_csv(lstm_protocols.TABLE, index=False)
    return table


def ablation():
    w = read(ablations.OUT / "ablation_runs.csv")
    f = read(ablations.OUT / "ablation_flights.csv")
    if w is None or f is None:
        return None
    table = ablations.summarise(w, f, ablations.magnetometer_features())
    table.to_csv(ablations.TABLE, index=False)
    return table


def stealth():
    """Step D: frontier, adaptive, white-box and hardening tables, and the figures."""
    from aerovigil.stealth import adaptive, figures, harden, whitebox
    from aerovigil.stealth import detectors as sd
    from aerovigil.stealth import frontier

    fixed = read(sd.OUT / "fixed_flights.csv")
    if fixed is None:
        return
    frontier.finish(fixed, time.time())
    exam = read(sd.OUT / "adaptive_exam.csv")
    if exam is not None:
        adaptive.summarise(exam).to_csv(adaptive.TABLE, index=False)
    wb, free = read(sd.OUT / "whitebox_flights.csv"), read(sd.OUT / "whitebox_free.csv")
    if wb is not None:
        whitebox.finish(wb, free if free is not None else pd.DataFrame())
    if (harden.HDIR / "random_forest" / "fixed_flights.csv").is_file():
        harden.summarise()
    figures.run()


def explain():
    """Step E: reason tables, clustering and figures from the saved alarm file."""
    from aerovigil.explain import reasons
    from aerovigil.explain import run as explain_run

    if (reasons.OUT / "alarms.csv").is_file():
        explain_run.summarise(with_shap=False)  # SHAP table is reused, not recomputed


def combine(cls, phy, lst):
    """Every detector on one table, the same columns, unseen missions first."""
    parts = []
    if cls is not None:
        c = cls.rename(columns={"model": "detector"})
        c["detected_spread"] = [
            f"sd {s:.3f}" if np.isfinite(s) else ""
            for s in c.get("detected_sd", np.nan)
        ]
        parts.append(c)
    if phy is not None:
        p = phy.copy()
        p["seeds"] = 1
        p["detected_spread"] = [
            f"95% CI {a:.3f}-{b:.3f}" if np.isfinite(a) else ""
            for a, b in zip(p["detected_ci_low"], p["detected_ci_high"])
        ]
        parts.append(p)
    if lst is not None:
        q = lst[~lst["protocol"].str.startswith("size")].copy()
        q.columns = [c[:-5] if c.endswith("_mean") else c for c in q.columns]
        q["detected"] = q["syn_detected"]
        q["detected_spread"] = [
            f"sd {s:.3f}" if np.isfinite(s) else "" for s in q["syn_detected_std"]
        ]
        q["fa_upper_95"] = np.nan  # per-seed bounds do not average; see C_lstm.csv
        parts.append(q)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out["family"] = out["detector"].map(FAMILY)
    out = out[[c for c in COLUMNS if c in out]]
    return out.sort_values(
        ["protocol", "family", "detector"],
        key=lambda s: s.map(ORDER) if s.name == "protocol" else s,
    ).reset_index(drop=True)


def run():
    started = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("AEROVIGIL - REBUILD EVERY TABLE FROM SAVED RESULTS")
    print("=" * 78)
    compare_runs.run()
    classical(full=False)
    cls, phy, lst = classical(full=True), physics(), lstm()
    ablation()
    combined = combine(cls, phy, lst)
    if combined is not None:
        combined.to_csv(TABLES / "C_all_detectors.csv", index=False)
        pd.set_option("display.width", 250)
        print("\nEVERY DETECTOR (unseen missions first)")
        print(combined.round(3).to_string(index=False))
    from aerovigil.fusion import threat_model

    threat_model.run()
    stealth()
    explain()
    print(f"\nAll tables rebuilt in {time.time() - started:.0f} s -> {TABLES}")
    return combined


def main():
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
