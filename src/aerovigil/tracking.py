r"""Plan M14 (Step I): MLflow tracking for every model-training run.

One local MLflow store for the whole project: sqlite:///<repo>/mlflow.db (gitignored).
Training code calls `log_run(experiment, run_name, params, metrics, tags)`; nothing
breaks when MLflow is missing, and setting AEROVIGIL_NO_MLFLOW=1 switches it off (the
test suite does).

    aerovigil-m5-classical     Module 05 / Steps B-C: each classical model, protocol
                               and seed (all_models_v2.py)
    aerovigil-m7-lstm          Module 07 / Step C: each LSTM job (lstm_protocols.py)
    aerovigil-m8-hardening     Step D: hardened random forest and supervised LSTM
    aerovigil-module07*        the older Module 07 scripts (train_models, tune_lstm)

`python -m aerovigil.tracking --backfill` logs the results the project already saved
(Step C per-seed tables, Step D hardening table) as runs tagged source=saved, so the
store is complete without retraining. Browse it with:

    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
_mlflow = None


def enabled():
    return os.environ.get("AEROVIGIL_NO_MLFLOW", "") not in ("1", "true", "yes")


def client():
    """The mlflow module pointed at the project store, or None."""
    global _mlflow
    if not enabled():
        return None
    if _mlflow is None:
        try:
            import logging

            import mlflow

            logging.getLogger("mlflow").setLevel(logging.ERROR)
            logging.getLogger("alembic").setLevel(logging.ERROR)
            mlflow.set_tracking_uri(STORE)
            _mlflow = mlflow
        except Exception as e:  # noqa: BLE001
            print(f"(MLflow disabled: {e})")
            return None
    return _mlflow


def _clean_metrics(metrics):
    out = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[k] = f
    return out


def log_run(experiment, run_name, params=None, metrics=None, tags=None):
    """Log one finished run. Returns the run id, or None when tracking is off."""
    m = client()
    if m is None:
        return None
    try:
        m.set_experiment(experiment)
        with m.start_run(run_name=run_name) as run:
            if params:
                m.log_params({k: str(v)[:500] for k, v in params.items()})
            if metrics:
                m.log_metrics(_clean_metrics(metrics))
            if tags:
                m.set_tags({k: str(v) for k, v in tags.items()})
            return run.info.run_id
    except Exception as e:  # noqa: BLE001  tracking must never stop a run
        print(f"(MLflow logging failed: {e})")
        return None


class Tracker:
    """The Module 07 scripts' interface, on the shared store."""

    def __init__(self, enabled_, experiment):
        self.experiment = experiment if enabled_ else None

    def log(self, name, params, metrics):
        if self.experiment:
            log_run(self.experiment, name, params, metrics)


# ---------------------------------------------------------------- backfill --


def backfill():
    """Log the saved Step C / D results as runs (source=saved)."""
    import pandas as pd

    import aerovigil.baselines.all_models_v2 as am
    from aerovigil.evaluation import lstm_protocols as lp
    from aerovigil.evaluation import store

    n = 0
    m05 = am.OUT_DIR
    w, f = m05 / "all_models_window_full.csv", m05 / "all_models_flight_full.csv"
    if w.is_file() and store.path_of(f) is not None:
        _, per = am.summary(pd.read_csv(w), store.read_csv(f))
        for _, r in per.iterrows():
            metrics = r.drop(["protocol", "model", "seed"]).to_dict()
            log_run("aerovigil-m5-classical", f"{r['model']} {r['protocol']} s{r['seed']}",
                    {"model": r["model"], "protocol": r["protocol"],
                     "seed": r["seed"]}, metrics,
                    {"source": "saved", "step": "C"})  # fmt: skip
            n += 1
    per = lp.PDIR / "per_seed.csv"
    if per.is_file():
        for _, r in pd.read_csv(per).iterrows():
            metrics = r.drop(["protocol", "detector", "seed"]).to_dict()
            log_run("aerovigil-m7-lstm",
                    f"{r['detector']} {r['protocol']} s{r['seed']}",
                    {"detector": r["detector"], "protocol": r["protocol"],
                     "seed": r["seed"]}, metrics,
                    {"source": "saved", "step": "C"})  # fmt: skip
            n += 1
    hard = ROOT / "reports" / "tables" / "D_hardening.csv"
    if hard.is_file():
        for i, r in pd.read_csv(hard).iterrows():
            params = {k: r[k] for k in r.index if isinstance(r[k], str)}
            metrics = {k: r[k] for k in r.index if not isinstance(r[k], str)}
            log_run("aerovigil-m8-hardening",
                    " ".join(str(v) for v in params.values()) or f"row {i}",
                    params, metrics, {"source": "saved", "step": "D"})  # fmt: skip
            n += 1
    print(f"Logged {n} saved runs to {STORE}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    if ap.parse_args().backfill:
        backfill()


if __name__ == "__main__":
    main()
