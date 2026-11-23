r"""Step 7: train the final models and score every window.

  ae                 LSTM autoencoder, normal windows only            (n_final_seeds)
  clf_seen           supervised LSTM, all attack families in training  (n_final_seeds)
  clf_unseen_<type>  supervised LSTM with ONE attack family held out   (n_unseen_seeds each)

Every model scores ALL windows, so alarms and metrics can be computed afterwards without
retraining. Scores are saved to scores/<name>.npy; scores/index.csv lists the runs.

Usage:
    python src\aerovigil\lstm\train_models.py
    python src\aerovigil\lstm\train_models.py --quick      (1 seed, short training: smoke test)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.lstm.common import (
    Tracker,
    load_scaled,
    load_yaml,
    model_params,  # noqa: E402
    out_dir,
    set_threads,
)
from aerovigil.lstm.datasets import HELD_OUT, ae_masks, clf_masks  # noqa: E402
from aerovigil.lstm.models import build_model, fit, predict  # noqa: E402


def run(config, output_dir=None, quick=False):
    cfg = load_yaml(config)
    out = out_dir(cfg, output_dir)
    set_threads(cfg)
    X, _, meta, _ = load_scaled(out)
    t = cfg["training"]
    n_final = 1 if quick else t["n_final_seeds"]
    n_unseen = 1 if quick else t["n_unseen_seeds"]
    max_epochs = 3 if quick else t["max_epochs"]
    tracker = Tracker(cfg["tuning"]["use_mlflow"] and not quick, "aerovigil-module07")
    sdir = out / "scores"
    sdir.mkdir(exist_ok=True)
    for f in sdir.glob("*.npy"):
        f.unlink()
    p_ae, p_clf = model_params(cfg, out, "ae"), model_params(cfg, out, "clf")

    jobs = [("ae", None, s) for s in range(n_final)] + [
        ("clf", None, s) for s in range(n_final)
    ]
    jobs += [("clf", h, s) for h in HELD_OUT for s in range(n_unseen)]
    print("=" * 65)
    print("AEROVIGIL - TRAIN AND SCORE")
    print("=" * 65)
    print(
        f"AE params {p_ae}\nClassifier params {p_clf}\nTraining runs: {len(jobs)} "
        f"({'quick' if quick else 'full'})"
    )
    rows, started = [], time.time()
    for n, (kind, held, s) in enumerate(jobs, start=1):
        seed = t["seed"] + s
        name = (
            "ae"
            if kind == "ae"
            else "clf_seen" if held is None else f"clf_unseen_{held}"
        ) + f"_seed{seed}"
        t0 = time.time()
        if kind == "ae":
            tr, va = ae_masks(meta)
            model = build_model("ae", X.shape[-1], p_ae, seed)
            model, h = fit(
                "ae",
                model,
                X[tr],
                None,
                X[va],
                None,
                p_ae,
                seed,
                max_epochs,
                t["patience"],
            )
            score, ferr = predict("ae", model, X)
            if s == 0:
                np.save(
                    sdir / f"ae_feature_errors_seed{seed}.npy", ferr.astype(np.float16)
                )
            params = p_ae
        else:
            tr, va = clf_masks(meta, held)
            if not meta.loc[tr, "y"].eq(1).any():
                print(f"  skipped {name}: no positive training windows")
                continue
            model = build_model("clf", X.shape[-1], p_clf, seed)
            model, h = fit(
                "clf",
                model,
                X[tr],
                meta.loc[tr, "y"].to_numpy(),
                X[va],
                meta.loc[va, "y"].to_numpy(),
                p_clf,
                seed,
                max_epochs,
                t["patience"],
            )
            score, _ = predict("clf", model, X)
            params = p_clf
        np.save(sdir / f"{name}.npy", score.astype(np.float32))
        rows.append(
            {
                "name": name,
                "kind": kind,
                "protocol": "seen" if held is None else "unseen",
                "held_out": held or "",
                "seed": seed,
                "best_epoch": h["best_epoch"],
                "best_val_loss": h["best_val"],
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "seconds": round(time.time() - t0, 1),
            }
        )
        tracker.log(
            name,
            params,
            {"best_val_loss": h["best_val"], "best_epoch": h["best_epoch"]},
        )
        print(
            f"  [{n}/{len(jobs)}] {name:<40} epochs {h['best_epoch'] + 1:>3} | val loss "
            f"{h['best_val']:.4f} | {time.time() - t0:.0f} s"
        )
    idx = pd.DataFrame(rows)
    idx.to_csv(sdir / "index.csv", index=False)
    print(f"\nElapsed: {time.time() - started:.0f} s | scores saved to {sdir}")
    print("=" * 65)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(a.config, a.output_dir, a.quick)


if __name__ == "__main__":
    main()
