r"""AEROVIGIL Module 07 v2 step 2: train the LSTM detectors and score every window.

Three detectors, all on the 36 v2 features and the event label:

  lstm_ae              one-class autoencoder, fitted on clean windows of normal
                       training flights only. It never sees an attack, so it is the
                       detector that can meet an unknown attack.
  lstm_supervised      classifier, ramp and snapback against clean, trained on the real
                       attacks and on the synthetic main set.
  lstm_unseen          the same classifier with one attack family held out of training
                       and validation, scored only on that family. This is the number
                       to quote for generalisation.

What changes against Module 07 v1:
  * the classifier's positives are event windows, not every window whose attack
    fraction passes 0.5. v1 asked it to separate the plateau from normal flight, and
    the plateau is where a constant offset looks exactly like an ordinary flight;
  * the autoencoder is scored on the same event windows for the same reason. v1
    reported AUC 0.428 on the real attacks, below chance, which is the signature of
    a plateau that reconstructs better than real normal flight;
  * no scaler is fitted here. The features arrive normalized from Module 03 v2;
  * calibration flights are held out of every fit, so evaluate_v2.py can set the alarm
    threshold on about 2 h of untouched normal flight instead of 0.7 h.

Writes data/processed/module07_v2/scores/<name>.npy (one score per window of the
combined table), scores/index.csv and training_history.json

Usage:
    python src\aerovigil\lstm\train_v2.py
    python src\aerovigil\lstm\train_v2.py --seeds 1 --no-unseen
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aerovigil.lstm.build_sequences_v2 import OUT_DIR, ROOT
from aerovigil.lstm.models import build_model, fit, predict

SEEDS = (42, 43, 44)
UNSEEN_SEEDS = (42, 43)
MAX_EPOCHS = 40
PATIENCE = 6
THREADS = 4
AE_PARAMS = {"hidden": 32, "layers": 1, "latent": 16, "dropout": 0.0, "lr": 1e-3,
             "batch_size": 256, "weight_decay": 1e-5, "grad_clip": 1.0}  # fmt: skip
CLF_PARAMS = {"hidden": 32, "layers": 1, "dropout": 0.1, "lr": 1e-3,
              "batch_size": 256, "weight_decay": 1e-5, "grad_clip": 1.0}  # fmt: skip
FAMILIES = ("abrupt_jump", "constant_offset", "linear_drift", "exponential_ramp",
            "low_amplitude_wander", "original")  # fmt: skip


def load(out):
    """The combined window table: original flights then synthetic ones."""
    xs, metas = [], []
    for name in ("orig", "syn"):
        p = out / f"X_{name}.npy"
        if not p.is_file():
            if name == "orig":
                raise SystemExit(f"Missing {p}. Run build_sequences_v2.py first.")
            continue
        xs.append(np.load(p))
        metas.append(pd.read_parquet(out / f"meta_{name}.parquet"))
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)


def ae_masks(meta):
    """Clean windows of normal flights: fit on the train role, stop on validation."""
    base = (
        (meta["origin"] == "orig")
        & (meta["attack_type"] == "none")
        & (meta["y_event"] == 0)
    )
    return base & (meta["role"] == "train"), base & (meta["role"] == "validation")


def clf_masks(meta, held_out=None):
    """Event windows against clean windows, with one family optionally held out."""
    clean = (meta["origin"] == "orig") & (meta["y_event"] == 0)
    real = (meta["origin"] == "orig") & (meta["y_event"] == 1)
    syn = (
        (meta["origin"] == "syn")
        & (meta["attack_set"] == "main")
        & (meta["y_event"] == 1)
    )
    if held_out == "original":
        real = real & False
    elif held_out is not None:
        syn = syn & (meta["attack_type"] != held_out)
    use = clean | real | syn
    return use & (meta["role"] == "train"), use & (meta["role"] == "validation")


def fit_one(kind, x, meta, params, seed, held_out=None):
    """Fit one model and return it with its training history."""
    tr, va = ae_masks(meta) if kind == "ae" else clf_masks(meta, held_out)
    tr, va = tr.to_numpy(), va.to_numpy()
    if tr.sum() == 0 or va.sum() == 0:
        return None, None
    y = meta["y_event"].to_numpy().astype(np.float32)
    yt, yv = (None, None) if kind == "ae" else (y[tr], y[va])
    if kind == "clf" and (len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2):
        return None, None
    model, hist = fit(
        kind, build_model(kind, x.shape[-1], params, seed), x[tr], yt, x[va], yv,
        params, seed, MAX_EPOCHS, PATIENCE,
    )  # fmt: skip
    hist["n_train"], hist["n_val"] = int(tr.sum()), int(va.sum())
    return model, hist


def run(seeds=None, unseen=True, output_dir=None):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    sdir = out / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    x, meta = load(out)
    seeds = SEEDS[:seeds] if isinstance(seeds, int) else (seeds or SEEDS)
    print("=" * 70)
    print("AEROVIGIL - MODULE 07 v2 TRAINING (event label, v2 features)")
    print("=" * 70)
    print(f"Windows {len(x):,}   shape {x.shape[1:]}   seeds {list(seeds)}")
    jobs = [("ae", s, None) for s in seeds] + [("clf", s, None) for s in seeds]
    if unseen:
        jobs += [("clf", s, f) for f in FAMILIES for s in UNSEEN_SEEDS if s in seeds]
    rows, history, started = [], {}, time.time()
    for kind, seed, held in jobs:
        name = (
            f"{kind}_seed{seed}" if held is None else f"{kind}_unseen_{held}_seed{seed}"
        )
        params = AE_PARAMS if kind == "ae" else CLF_PARAMS
        model, hist = fit_one(kind, x, meta, params, seed, held)
        if model is None:
            print(f"  skipped {name}: not enough windows on both sides")
            continue
        score, _ = predict(kind, model, x)
        np.save(sdir / f"{name}.npy", np.asarray(score, dtype=np.float32))
        history[name] = hist
        rows.append({
            "name": name, "kind": kind, "seed": seed,
            "protocol": "seen" if held is None else "unseen",
            "held_out": held, "n_train": hist["n_train"], "n_val": hist["n_val"],
            "best_epoch": hist["best_epoch"], "best_val": hist["best_val"],
        })  # fmt: skip
        print(
            f"  {name:38s} train {hist['n_train']:6,d}  val {hist['n_val']:6,d}  "
            f"epoch {hist['best_epoch']:2d}  val_loss {hist['best_val']:.5f}  "
            f"({time.time() - started:.0f} s)"
        )
    idx = pd.DataFrame(rows)
    idx.to_csv(sdir / "index.csv", index=False)
    (out / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTrained {len(idx)} models in {time.time() - started:.0f} s -> {sdir}")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="use the first N seeds")
    ap.add_argument("--no-unseen", action="store_true")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.seeds, not a.no_unseen, a.output_dir)


if __name__ == "__main__":
    main()
