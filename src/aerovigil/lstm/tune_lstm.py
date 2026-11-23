r"""Step 6: small Optuna search for the AE and the supervised LSTM (validation flights only).

Both models are tuned on validation AUC: normal validation windows against the validation
attack windows (real and synthetic). The AE still trains on normal windows only; attacks are
used here to pick hyperparameters, never to fit weights. Test flights are not touched.

Usage:
    python src\aerovigil\lstm\tune_lstm.py
    python src\aerovigil\lstm\tune_lstm.py --trials 4 --output-dir data\interim\module07\smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.lstm.common import (
    Tracker,
    load_scaled,
    load_yaml,
    out_dir,  # noqa: E402
    set_threads,
)
from aerovigil.lstm.datasets import ae_masks, clf_masks  # noqa: E402
from aerovigil.lstm.models import build_model, fit, predict  # noqa: E402


def suggest(trial, kind):
    p = {
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
        "layers": trial.suggest_categorical("layers", [1, 2]),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
    }
    if kind == "ae":
        p["latent"] = trial.suggest_categorical("latent", [8, 16, 32])
    return p


def run(config, trials=None, output_dir=None):
    cfg = load_yaml(config)
    out = out_dir(cfg, output_dir)
    set_threads(cfg)
    X, _, meta, _ = load_scaled(out)
    t, tun = cfg["training"], cfg["tuning"]
    n_trials = trials or tun["n_trials"]
    tracker = Tracker(tun["use_mlflow"], "aerovigil-module07-tuning")
    val_attack = (
        (meta["split"] == "validation")
        & (meta["y"] == 1)
        & meta["origin"].isin(["orig", "syn"])
    )
    ae_tr, ae_va = ae_masks(meta)
    clf_tr, clf_va = clf_masks(meta)
    print("=" * 65)
    print("AEROVIGIL - LSTM TUNING (validation flights only)")
    print("=" * 65)
    print(
        f"AE train {int(ae_tr.sum()):,} / val {int(ae_va.sum()):,} normal windows | "
        f"classifier train {int(clf_tr.sum()):,} / val {int(clf_va.sum()):,} windows | "
        f"validation attack windows {int(val_attack.sum()):,}"
    )
    results = {}
    for kind in ("ae", "clf"):
        started = time.time()

        def objective(trial, kind=kind):
            p = suggest(trial, kind)
            full = {
                **p,
                "batch_size": t["batch_size"],
                "weight_decay": t["weight_decay"],
                "grad_clip": t["grad_clip"],
            }
            model = build_model(kind, X.shape[-1], full, t["seed"])
            if kind == "ae":
                model, h = fit(
                    "ae",
                    model,
                    X[ae_tr],
                    None,
                    X[ae_va],
                    None,
                    full,
                    t["seed"],
                    tun["max_epochs"],
                    tun["patience"],
                )
                neg, pos = ae_va.to_numpy(), val_attack.to_numpy()
                score = predict("ae", model, X[neg | pos])[0]
                y = pos[neg | pos].astype(int)
            else:
                model, h = fit(
                    "clf",
                    model,
                    X[clf_tr],
                    meta.loc[clf_tr, "y"].to_numpy(),
                    X[clf_va],
                    meta.loc[clf_va, "y"].to_numpy(),
                    full,
                    t["seed"],
                    tun["max_epochs"],
                    tun["patience"],
                )
                score = predict("clf", model, X[clf_va])[0]
                y = meta.loc[clf_va, "y"].to_numpy()
            auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else 0.5
            tracker.log(
                f"{kind}-trial{trial.number}",
                p,
                {"val_auc": auc, "val_loss": h["best_val"]},
            )
            return auc

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=t["seed"])
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        results[kind] = {
            "params": study.best_params,
            "val_auc": float(study.best_value),
            "trials": n_trials,
        }
        print(
            f"\n{kind.upper()}: best validation AUC {study.best_value:.3f} after {n_trials} trials "
            f"({time.time() - started:.0f} s)"
        )
        print(f"  params: {study.best_params}")
    (out / "best_params.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out / 'best_params.json'}")
    print("=" * 65)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.trials, a.output_dir)


if __name__ == "__main__":
    main()
