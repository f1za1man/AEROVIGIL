"""Figures for the Module 05 report (saved to reports/figures/module05)."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import load_yaml, resolve  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures" / "module05"
SHORT = {"logistic_regression": "LogReg", "naive_bayes": "NaiveBayes", "knn": "KNN",
         "decision_tree": "DecTree", "random_forest": "RandForest", "isolation_forest": "IsoForest"}
TYPES = ["abrupt_jump", "constant_offset", "linear_drift", "exponential_ramp", "low_amplitude_wander"]
TYPE_SHORT = {"abrupt_jump": "jump", "constant_offset": "offset", "linear_drift": "drift",
              "exponential_ramp": "exp. ramp", "low_amplitude_wander": "wander"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "baselines" / "baselines.yaml"))
    cfg = load_yaml(ap.parse_args().config)
    out_base = resolve(cfg["paths"]["output_dir"])
    res = out_base / f"results_{'+'.join(cfg['training_sources'])}"
    main_df = pd.read_csv(res / "results_main.csv")
    by_attack = pd.read_csv(res / "results_by_attack.csv")
    scores = np.load(res / "scores_seen.npz")
    models = list(cfg["models"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if cfg["training_sources"] == ["orig", "syn"] else "_" + "+".join(cfg["training_sources"])

    # Figure 1: F1 by model, seen vs unseen missions.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, ts, title in zip(axes, ("orig_test", "syn_all"),
                             ("Original dataset attacks (test flights)",
                              "Synthetic attacks, Module 04 (test flights)")):
        sub = main_df[main_df["test_set"] == ts]
        x = np.arange(len(models)); w = 0.38
        for k, (proto, label, colour) in enumerate((("seen", "seen missions", "tab:blue"),
                                                    ("unseen_mission_mean", "unseen missions (mean of 3 folds)", "tab:orange"))):
            s = sub[sub["protocol"] == proto].set_index("model").reindex(models)
            ax.bar(x + (k - 0.5) * w, s["f1_mean"], w, yerr=s["f1_std"], capsize=3,
                   label=label, color=colour)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in models], rotation=20)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("window-level F1"); axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"M5_fig1_f1_seen_vs_unseen{tag}.png", dpi=150); plt.close(fig)

    # Figure 2: ROC curves, seen protocol.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, ts, title in zip(axes, ("orig_test", "syn_all"),
                             ("Original dataset attacks", "Synthetic attacks")):
        for m in models:
            y, s = scores[f"{m}|{ts}|y"], scores[f"{m}|{ts}|s"]
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.4, label=SHORT[m])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title(f"ROC, seen missions: {title}", fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"M5_fig2_roc_seen{tag}.png", dpi=150); plt.close(fig)

    # Figure 3: detection by attack type and velocity mode (seen protocol).
    cols = [f"{t}|{mode}" for t in TYPES for mode in ("consistent", "position_only")]
    seen = by_attack[(by_attack["protocol"] == "seen")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    for ax, metric, title in zip(axes, ("recall_mean", "roc_auc_mean"),
                                 ("Recall (attack windows detected)", "ROC-AUC")):
        grid = np.full((len(models), len(cols)), np.nan)
        for i, m in enumerate(models):
            for j, c in enumerate(cols):
                r = seen[(seen["model"] == m) & (seen["test_set"] == f"syn:{c}")]
                if len(r):
                    grid[i, j] = r[metric].iloc[0]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(models))); ax.set_yticklabels([SHORT[m] for m in models])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{TYPE_SHORT[c.split('|')[0]]}\n{'C' if c.endswith('consistent') else 'P'}"
                            for c in cols], fontsize=8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[i, j] < 0.6 else "black")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025)
    fig.suptitle("Synthetic attacks by type and velocity mode (seen missions)   "
                 "C = consistent velocity, P = position only", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"M5_fig3_detection_by_attack{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: random-forest feature importance.
    imp = pd.read_csv(res / "rf_importance_seen.csv").head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp["feature"], imp["importance"], color="tab:green")
    ax.set_xlabel("importance (Random Forest, seen missions)")
    ax.set_title("Top 15 window features"); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"M5_fig4_rf_importance{tag}.png", dpi=150); plt.close(fig)

    for f in sorted(FIG_DIR.glob("M5_fig*.png")):
        print("Saved:", f)


if __name__ == "__main__":
    main()