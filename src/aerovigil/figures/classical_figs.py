"""Step K, plan M5: the six classical models (ROC, confusion matrices, metrics, attack
types, feature importance).

ROC curves need window scores, which Steps B/C do not keep: each model is refitted once
at seed 42 on the seen protocol exactly as Step C fits it (same tuned parameters, same
training windows), its test scores cached in data/processed/figures/, and its AUC
checked against Step C's saved value before anything is drawn.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from aerovigil.baselines import all_models_v2 as am
from aerovigil.figures import style as S
from aerovigil.stealth import detectors as D

ME = "src/aerovigil/figures/classical_figs.py"
M05 = S.ROOT / "data" / "processed" / "module05_v2"
CACHE = S.ROOT / "data" / "processed" / "figures" / "m5_scores_s42.joblib"
WINDOW = M05 / "all_models_window_full.csv"
TABLE = S.ROOT / "reports" / "tables" / "C_classical_models.csv"
FLIGHTS = M05 / "all_models_flight_full.csv"
SEED = 42
Y = am.Y


def saved_seen():
    w = pd.read_csv(WINDOW)
    return w[(w["protocol"] == "seen") & (w["seed"] == SEED)].set_index("model")


def scores():
    """{model: (real labels, real scores, consistent labels, consistent scores)}."""
    if CACHE.is_file():
        return joblib.load(CACHE)
    orig, syn = am.load(M05)
    cols = am.feature_columns(orig)
    tr, _, _, o_test, s_test = am.protocol_sets(orig, syn, None, "seen")
    lab = (o_test[Y] >= 0).to_numpy()
    neg = (o_test[Y] == 0).to_numpy()
    cons = ((s_test["velocity_mode"] == "consistent") & (s_test[Y] == 1)
            & (s_test["attack_type"] != "abrupt_jump")).to_numpy()  # fmt: skip
    saved = saved_seen()
    out = {}
    for name in S.CLASSICAL:
        print(f"    refit {name} (seed {SEED})", flush=True)
        if name in ("random_forest", "decision_tree"):
            pipe, _, _ = D.fit_classical(name, SEED)
        else:
            pipe = am.fit(name, D.seen_params(name), tr, cols, SEED)
        so = am.score(name, pipe, o_test, cols)
        ss = am.score(name, pipe, s_test, cols)
        y_real, s_real = o_test.loc[lab, Y].to_numpy(), so[lab]
        y_cons = np.r_[np.ones(cons.sum()), np.zeros(neg.sum())]
        s_cons = np.r_[ss[cons], so[neg]]
        for got, col in ((am.auc(y_real, s_real), "roc_auc"),
                         (am.auc(y_cons, s_cons), "auc_syn_consistent")):  # fmt: skip
            want = float(saved.loc[name, col])
            if not np.isclose(got, want, atol=1e-6):
                raise SystemExit(
                    f"{name} {col} {got:.6f} differs from Step C {want:.6f}"
                )
        out[name] = (y_real, s_real, y_cons, s_cons)
        if name == "random_forest":
            out["_importance"] = pd.Series(pipe[-1].feature_importances_, index=cols)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    D.atomic_write(CACHE, lambda p: joblib.dump(out, p))
    return out


def m5_roc():
    sc = scores()
    saved = saved_seen()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    panels = ((0, "Real attacks (test windows)", "roc_auc"),
              (2, "Velocity-consistent synthetic attacks", "auc_syn_consistent"))  # fmt: skip
    for ax, (k, title, col) in zip(axes, panels):
        for name in S.CLASSICAL:
            y, s = sc[name][k], sc[name][k + 1]
            fpr, tpr, _ = roc_curve(y, s)
            fpr = np.maximum(fpr, 1e-4)
            kw = S.line_kw(name, marker=None, linewidth=2)
            kw["label"] = f"{S.label(name)} (AUC {saved.loc[name, col]:.3f})"
            ax.plot(fpr, tpr, **kw)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False-positive rate (log scale)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.16), ncol=2)
    axes[0].set_ylabel("True-positive rate")
    fig.suptitle("ROC curves, classical models, seen missions (seed 42)")
    fig.tight_layout()
    S.note(fig, "Window level, 5 s windows. Synthetic panel: velocity-consistent attack "
                "windows (abrupt jumps excluded) against the real normal test windows. "
                "Models refitted exactly as in Step C; AUCs match the saved values.")  # fmt: skip
    return S.save(fig, "M5_roc_curves",
                  "ROC curves of the six classical models on real and on "
                  "velocity-consistent synthetic attacks.",
                  ["data/processed/module05_v2/windows_{orig,syn}.parquet",
                   "data/processed/module05_v2/all_models_window_full.csv"], ME)  # fmt: skip


def m5_confusion():
    saved = saved_seen()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for ax, name in zip(axes.ravel(), S.CLASSICAL):
        r = saved.loc[name]
        m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        rate = m / m.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(m[i, j]):,}\n({rate[i, j]:.1%})", ha="center",
                        va="center", fontsize=9,
                        color="white" if rate[i, j] > 0.6 else S.INK)  # fmt: skip
        ax.set_xticks([0, 1], ["normal", "attack"])
        ax.set_yticks([0, 1], ["normal", "attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(S.label(name), fontsize=10)
    fig.suptitle("Confusion matrices at the 1 false alarm per hour threshold")
    fig.tight_layout()
    S.note(fig, "Real test windows, seen missions, seed 42; threshold set on clean "
                "validation flights. Row percentages: share of each true class.")  # fmt: skip
    return S.save(fig, "M5_confusion_matrices",
                  "Confusion matrices of the six classical models at the shared alarm "
                  "budget.", "data/processed/module05_v2/all_models_window_full.csv", ME)  # fmt: skip


def m5_metrics():
    t = pd.read_csv(TABLE)
    protos = [("seen", "Seen missions", "#257de0", None),
              ("unseen missions", "Unseen mission", "#f47230", "//")]  # fmt: skip
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    y = np.arange(len(S.CLASSICAL))
    for ax, (m, title) in zip(axes, metrics):
        for i, (p, lab, col, hatch) in enumerate(protos):
            v = t[t["protocol"] == p].set_index("model").reindex(S.CLASSICAL)
            sd = v[f"{m}_sd"].fillna(0).to_numpy()
            ax.barh(y + (0.2 if i else -0.2), v[m], 0.38, xerr=sd, color=col,
                    hatch=hatch, edgecolor=S.SURFACE, label=lab,
                    error_kw={"ecolor": S.INK2, "capsize": 2})  # fmt: skip
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel("Score (0-1)")
    axes[0].set_yticks(y, [S.label(n) for n in S.CLASSICAL])
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(1, 1.0))
    fig.suptitle("Window-level scores, seen vs unseen missions", x=0.4)
    fig.tight_layout()
    S.note(fig, "Real test windows at the 1 false alarm per hour threshold; unseen = "
                "leave-one-mission-out, pooled over the three held-out missions. Error "
                "bars: standard deviation over 5 seeds (models with randomness only).")  # fmt: skip
    return S.save(fig, "M5_metrics_seen_unseen",
                  "Accuracy, precision, recall and F1 of each classical model on seen "
                  "and on unseen missions.", str(TABLE.relative_to(S.ROOT)), ME)  # fmt: skip


def m5_by_attack():
    f = pd.read_csv(FLIGHTS)
    f = f[(f["protocol"] == "seen") & f["attacked"]]
    f["kind"] = np.where(f["origin"] == "orig", "Real attacks",
                         f["attack_type"].map(S.FAMILIES) + ", "
                         + f["velocity_mode"].map({"position_only": "position only",
                                                   "consistent": "consistent"}))  # fmt: skip
    t = f.groupby(["model", "kind"])["detected"].mean().unstack().reindex(S.CLASSICAL)
    cols = ["Real attacks"] + sorted(c for c in t.columns if c != "Real attacks")
    t = t[cols]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(t.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else S.INK)  # fmt: skip
    ax.set_yticks(range(len(t)), [S.label(n) for n in t.index])
    ax.set_xticks(
        range(len(cols)),
        cols,
        fontsize=8,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Attacked test flights caught", fraction=0.025)
    ax.set_title("Detection by attack type (seen missions, mean of seeds)")
    fig.subplots_adjust(bottom=0.32)
    S.note(fig, "A flight counts as caught when the model alarms while the attack is "
                "on (per-flight latch, 10 s refractory).")  # fmt: skip
    return S.save(fig, "M5_detection_by_attack",
                  "Share of attacked test flights each classical model catches, per "
                  "attack family and velocity mode.", str(FLIGHTS.relative_to(S.ROOT)), ME)  # fmt: skip


def pretty_feature(c):
    base, _, stat = c.partition("__")
    return f"{base.replace('res_', '').replace('_', ' ')} ({stat})"


def m5_importance():
    imp = scores()["_importance"].sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh([pretty_feature(c) for c in imp.index], imp.to_numpy(),
            color=S.colour("random_forest"))  # fmt: skip
    for i, v in enumerate(imp.to_numpy()):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random forest: 15 most important window features")
    S.note(fig, "Seen-mission random forest, seed 42. Each feature is a statistic of "
                "one per-row residual over the 5 s window.")  # fmt: skip
    return S.save(fig, "M5_rf_importance",
                  "The window features the random forest relies on most.",
                  "data/processed/stealth/models/rf_seen_s42.joblib", ME)  # fmt: skip


def make():
    m5_roc()
    m5_confusion()
    m5_metrics()
    m5_by_attack()
    m5_importance()
