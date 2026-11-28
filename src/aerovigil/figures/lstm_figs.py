"""Step K, plan M7: the LSTM autoencoder and the supervised LSTM.

Window scores are Step C's saved seen-protocol scores (seed 42); the ROC AUCs drawn are
checked against Step C's saved values first.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from aerovigil.figures import style as S
from aerovigil.lstm.evaluate_v2 import calibration_pool
from aerovigil.lstm.sequential_v2 import flight_cusum, standardise

ME = "src/aerovigil/figures/lstm_figs.py"
M07 = S.ROOT / "data" / "processed" / "module07_v2"
PDIR = M07 / "protocols"
HISTORY = M07 / "training_history.json"
WINDOWS = PDIR / "windows.csv"
SEQ = M07 / "sequential_thresholds.csv"
KIND = {"ae": "lstm_ae", "clf": "lstm_supervised"}
SEED = 42


def meta():
    return pd.concat([pd.read_parquet(M07 / f"meta_{n}.parquet") for n in ("orig", "syn")],
                     ignore_index=True)  # fmt: skip


def score(kind, seed=SEED):
    return np.load(PDIR / "scores" / f"{kind}_seen_s{seed}.npy")


def masks(m):
    test = (m["role"] == "test").to_numpy()
    real = test & (m["origin"] == "orig").to_numpy() & (m["y_event"] >= 0).to_numpy()
    neg = real & (m["y_event"] == 0).to_numpy()
    cons = test & ((m["origin"] == "syn") & (m["attack_set"] == "main")
                   & (m["y_event"] == 1) & (m["velocity_mode"] == "consistent")
                   & (m["attack_type"] != "abrupt_jump")).to_numpy()  # fmt: skip
    return real, neg, cons


def saved(kind):
    w = pd.read_csv(WINDOWS)
    return w[w["name"] == f"{kind}_seen_s{SEED}"].iloc[0]


def m7_loss():
    h = json.loads(HISTORY.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    titles = {"ae": "LSTM autoencoder (reconstruction loss)",
              "clf": "Supervised LSTM (classification loss)"}  # fmt: skip
    for ax, kind in zip(axes, ("ae", "clf")):
        col = S.colour(KIND[kind])
        for k, name in enumerate(sorted(n for n in h if n.startswith(f"{kind}_seed"))):
            r = h[name]
            ep = np.arange(1, len(r["train"]) + 1)
            seed = name.split("seed")[1]
            alpha = 1.0 if k == 0 else 0.45
            ax.plot(ep, r["train"], color=col, linewidth=1.8, alpha=alpha,
                    label="training" if k == 0 else None)  # fmt: skip
            ax.plot(ep, r["val"], color=S.INK, linewidth=1.6, linestyle="--",
                    alpha=alpha, label="validation" if k == 0 else None)  # fmt: skip
            ax.scatter([r["best_epoch"]], [r["best_val"]], color="#ec4647", zorder=3,
                       s=30, label="kept (best validation)" if k == 0 else None)  # fmt: skip
            ax.annotate(f"seed {seed}", (ep[-1], r["val"][-1]), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color=S.INK2,
                        va="center")  # fmt: skip
        ax.set_title(titles[kind], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Training and validation loss (seen missions, seeds 42-44)")
    fig.tight_layout()
    S.note(fig, "The weights of the epoch with the lowest validation loss are kept "
                "(early stopping). Step C retrains the same configuration for seeds "
                "42-46 with identical results for 42-44.")  # fmt: skip
    return S.save(fig, "M7_training_loss",
                  "Training and validation loss of both LSTMs per epoch, with the "
                  "epoch that was kept.", str(HISTORY.relative_to(S.ROOT)), ME)  # fmt: skip


def m7_error():
    m = meta()
    real, neg, cons = masks(m)
    s = score("ae")
    thr = float(saved("ae")["threshold"])
    pos_real = real & (m["y_event"] == 1).to_numpy()
    groups = [(neg, "Normal windows (real test flights)", "#257de0", "-"),
              (pos_real, "Real attack windows", "#ec4647", "--"),
              (cons, "Velocity-consistent synthetic attack windows", "#f2a600", ":")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 4.4))
    lo, hi = np.percentile(s[real | cons], [0.1, 99.9])
    bins = np.geomspace(max(lo, 1e-2), hi, 70)
    for mk, lab, col, ls in groups:
        ax.hist(s[mk], bins=bins, density=True, histtype="step", linewidth=2.2,
                color=col, linestyle=ls, label=f"{lab} ({mk.sum():,})")  # fmt: skip
    ax.axvline(thr, color=S.INK, linewidth=1.4, linestyle="-.")
    ax.annotate(f"alarm threshold {thr:.1f}", (thr, ax.get_ylim()[1] * 0.45),
                xytext=(4, 0), textcoords="offset points", fontsize=8)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (window score, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("LSTM autoencoder: reconstruction error, normal vs attack")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.16), ncol=1)
    S.note(fig, "Seen missions, seed 42, test windows. The threshold allows 1 false "
                "alarm per flight-hour on the clean calibration flights.")  # fmt: skip
    return S.save(fig, "M7_reconstruction_error",
                  "The autoencoder's reconstruction error on normal and attacked test "
                  "windows: real attacks stand out, velocity-consistent ones mostly "
                  "do not.", [f"data/processed/module07_v2/protocols/scores/"
                               f"ae_seen_s{SEED}.npy", str(WINDOWS.relative_to(S.ROOT))],
                  ME)  # fmt: skip


def m7_roc():
    m = meta()
    real, neg, cons = masks(m)
    y_all = m["y_event"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for kind in ("ae", "clf"):
        s = score(kind)
        w = saved(kind)
        sets = [(y_all[real], s[real], "roc_auc"),
                (np.r_[np.ones(cons.sum()), np.zeros(neg.sum())],
                 np.r_[s[cons], s[neg]], "auc_syn_consistent")]  # fmt: skip
        for ax, (y, sc, col) in zip(axes, sets):
            got = roc_auc_score(y, sc)
            if not np.isclose(got, w[col], atol=1e-6):
                raise SystemExit(
                    f"{kind} {col} {got:.6f} differs from Step C {w[col]:.6f}"
                )
            fpr, tpr, _ = roc_curve(y, sc)
            kw = S.line_kw(KIND[kind], marker=None)
            kw["label"] = f"{S.label(KIND[kind])} (AUC {got:.3f})"
            ax.plot(np.maximum(fpr, 1e-4), tpr, **kw)
    for ax, title in zip(axes, ("Real attacks (test windows)",
                                "Velocity-consistent synthetic attacks")):  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False-positive rate (log scale)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("True-positive rate")
    fig.suptitle("ROC curves, LSTMs, seen missions (seed 42)")
    fig.tight_layout()
    S.note(fig, "Window level, 5 s windows; synthetic panel: velocity-consistent attack "
                "windows (abrupt jumps excluded) against the real normal test windows.")  # fmt: skip
    return S.save(fig, "M7_roc_curves",
                  "ROC curves of the two LSTMs on real and on velocity-consistent "
                  "synthetic attacks.",
                  ["data/processed/module07_v2/protocols/scores/*_seen_s42.npy",
                   str(WINDOWS.relative_to(S.ROOT))], ME)  # fmt: skip


def m7_cusum():
    """CUSUM over the supervised LSTM's window score, no decay vs decay 0.90."""
    m = meta()
    s = np.load(M07 / "scores" / f"clf_seed{SEED}.npy")
    pool = calibration_pool(m)
    z = standardise(s, pool)
    thr = pd.read_csv(SEQ)
    thr = thr[(thr["name"] == f"clf_seed{SEED}") & np.isclose(thr["k"], 0.25)]
    sweep = m[(m["attack_set"] == "sweep") & np.isclose(m["drift_rate_mps"], 0.5)
              & (m["role"] == "test")]  # fmt: skip
    fid = sorted(sweep["flight_id"].unique())[0]
    src = sweep.loc[sweep["flight_id"] == fid, "source_flight_id"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    styles = {1.0: ("no decay", S.INK2, "--"), 0.9: ("decay 0.90", S.colour(
        "lstm_supervised"), "-")}  # fmt: skip
    for ax, (flight, title) in zip(axes, ((src, "Normal flight"),
                                          (fid, "Same flight, 0.5 m/s consistent drift"))):  # fmt: skip
        idx = m.index[m["flight_id"] == flight].to_numpy()
        idx = idx[np.argsort(m.loc[idx, "window_end_s"].to_numpy())]
        t = m.loc[idx, "window_end_s"].to_numpy()
        on = m.loc[idx, "y_event"].to_numpy() == 1
        if on.any():
            ax.axvspan(t[on].min(), t[on].max(), color="#f9c0c0", linewidth=0,
                       zorder=0, label="attack on")  # fmt: skip
        for decay, (lab, col, ls) in styles.items():
            series = flight_cusum(z, m, 0.25, decay)[idx]
            h = float(thr.loc[np.isclose(thr["decay"], decay), "h"].iloc[0])
            ax.plot(t, series, color=col, linestyle=ls, linewidth=2,
                    label=f"{lab} (threshold {h:.0f})")  # fmt: skip
            ax.axhline(h, color=col, linestyle=":", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Window end time (s)")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("CUSUM of the standardised LSTM score")
    fig.suptitle("Supervised LSTM: CUSUM over its window scores (k = 0.25)")
    fig.tight_layout()
    S.note(fig, f"Flight {src} (test). Without decay the sum never forgets, so its "
                f"threshold must sit high enough for flights of 46-265 s; with decay 0.90 "
                f"it forgets after about 10 windows and a lower threshold fits. "
                f"Thresholds: 1 false alarm per flight-hour on calibration flights.")  # fmt: skip
    return S.save(fig, "M7_cusum_lstm",
                  "CUSUM over the supervised LSTM's score on a normal flight and the same "
                  "flight with a slow drift, without and with decay.",
                  [f"data/processed/module07_v2/scores/clf_seed{SEED}.npy",
                   str(SEQ.relative_to(S.ROOT))], ME)  # fmt: skip


def m7_window_auc():
    r = pd.read_csv(M07 / "results_window.csv")
    groups = ["original"] + [f"{f}|{m}" for f in S.FAMILIES for m in S.MODES]
    rows = [
        ("lstm_ae", "seen", "LSTM autoencoder"),
        ("lstm_supervised", "seen", "Supervised LSTM, seen family"),
        ("lstm_supervised_unseen", "unseen", "Supervised LSTM, family held out"),
    ]
    t = pd.DataFrame(
        {
            lab: r[(r["detector"] == d) & (r["protocol"] == p)]
            .set_index("group")
            .reindex(groups)["roc_auc_mean"]
            for d, p, lab in rows
        }
    ).T
    fig, ax = plt.subplots(figsize=(14, 3.8))
    im = ax.imshow(t.to_numpy(float), cmap="Blues", vmin=0.5, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(t.to_numpy(float)):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.85 else S.INK,
            )
    labels = ["Real attacks"] + [
        S.cell_label(f, m) for f in S.FAMILIES for m in S.MODES
    ]
    ax.set_xticks(range(len(groups)), labels, fontsize=7)
    ax.set_yticks(range(len(t)), t.index, fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.02, label="Window ROC AUC")
    ax.set_title("LSTM window AUC per attack family and velocity mode (test windows)")
    S.note(
        fig,
        "Mean over seeds 42-44 (Module 07 runs). 'Family held out': the supervised "
        "LSTM trained without that family. Velocity-consistent drifts are the "
        "hard case for both models.",
    )
    return S.save(
        fig,
        "M7_window_auc",
        "Window-level AUC of both LSTMs for every attack family and velocity "
        "mode, including families the model never saw.",
        "data/processed/module07_v2/results_window.csv",
        ME,
    )


def make():
    m7_loss()
    m7_error()
    m7_roc()
    m7_cusum()
    m7_window_auc()
