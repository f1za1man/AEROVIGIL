"""Step K, plan M10: explainability (SHAP summary, one alarm's attribution, K-Means
clusters of alarms)."""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aerovigil.evaluation import store
from aerovigil.explain import reasons as R
from aerovigil.figures import style as S

ME = "src/aerovigil/figures/explain_figs.py"
TABLES = S.ROOT / "reports" / "tables"
SHAP = TABLES / "E_shap_global.csv"
KM = TABLES / "E_kmeans.csv"
SIL = TABLES / "E_kmeans_silhouette.csv"
EXP = S.ROOT / "data" / "processed" / "explain"
CLUSTERS = EXP / "alarm_clusters.csv"
ALARMS = EXP / "alarms.csv"
EXAMPLE = ("Curved__Attacked__log_21_04_2023_09_58_46", "kalman", 0)


def m10_shap():
    g = pd.read_csv(SHAP)
    order = (g[(g["model"] == "random_forest") & (g["windows"] == "attack windows")]
             .sort_values("mean_abs_shap")["check"].tolist())  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    y = np.arange(len(order))
    for ax, win in zip(axes, ("attack windows", "clean windows")):
        for i, model in enumerate(("random_forest", "decision_tree")):
            v = (g[(g["model"] == model) & (g["windows"] == win)].set_index("check")
                 .reindex(order)["mean_abs_shap"])  # fmt: skip
            ax.barh(y + (i - 0.5) * 0.38, v, 0.36, color=S.colour(model),
                    hatch=[None, "//"][i], edgecolor=S.SURFACE, label=S.label(model))  # fmt: skip
        n = int(g.loc[g["windows"] == win, "n_windows"].iloc[0])
        ax.set_title(f"{win.capitalize()} ({n:,} test windows)", fontsize=10)
        ax.set_xlabel("Mean |SHAP| (summed over each check's window features)")
    axes[0].set_yticks(y, [R.SHORT[c] for c in order])
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("SHAP: which sensor checks the tree models rely on")
    fig.tight_layout()
    S.note(fig, "TreeSHAP on the seen-mission models (seed 42). Each check groups the "
                "window statistics of one residual; e.g. horizontal speed = GPS position "
                "change vs GPS-reported velocity.")  # fmt: skip
    return S.save(fig, "M10_shap_summary",
                  "SHAP summary: the tree models rely first on horizontal-speed "
                  "consistency, then on course and GPS speed.",
                  str(SHAP.relative_to(S.ROOT)), ME)  # fmt: skip


def m10_attribution():
    a = store.read_csv(ALARMS)
    fid, det, k = EXAMPLE
    r = a[
        (a["flight_id"] == fid) & (a["detector"] == det) & (a["alarm_index"] == k)
    ].iloc[0]
    checks = list(R.SHORT)
    ratio = pd.Series({c: r[f"ratio_{c}"] for c in checks}).astype(float)
    ratio = ratio.sort_values()
    top = {r["reason1"], r["reason2"], r["reason3"]}
    names = {v: kk for kk, v in R.SHORT.items()}
    top = {names.get(t, t) for t in top}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.1, 1]})  # fmt: skip
    cols = [S.colour(det) if c in top else "#cfcfcf" for c in ratio.index]
    a1.barh([R.SHORT[c] for c in ratio.index], np.maximum(ratio.to_numpy(), 1e-2),
            color=cols)  # fmt: skip
    a1.axvline(1.0, color="#ec4647", linestyle="--", linewidth=1.3)
    a1.annotate("normal limit", (1.0, 0), xytext=(4, -2), textcoords="offset points",
                fontsize=8, color="#ec4647")  # fmt: skip
    a1.set_xscale("log")
    a1.set_xlabel("Value at the alarm / normal upper limit (log)")
    a1.set_title(f"Per-check attribution of one {S.label(det)} alarm", fontsize=10)
    a2.axis("off")
    msg = "\n".join(textwrap.wrap(r["message"], 62))
    a2.text(0, 1, "Operator message", fontsize=10, fontweight="bold", va="top")
    a2.text(0, 0.9, msg, fontsize=9, va="top", family="monospace")
    fig.tight_layout()
    S.note(fig, f"Flight {fid} (test), first alarm, {r['alarm_t_s']:.1f} s into the log. "
                f"Normal limit = 95th percentile of the check on clean training flights. "
                f"The three highlighted checks are the reasons in the message.")  # fmt: skip
    return S.save(fig, "M10_alarm_attribution",
                  "How one real alarm is explained: each sensor check against its normal "
                  "limit, and the message the operator sees.",
                  [str(ALARMS.relative_to(S.ROOT)) + ".gz",
                   "data/processed/explain/normals.json"], ME)  # fmt: skip


def m10_clusters():
    a = pd.read_csv(CLUSTERS)
    km = pd.read_csv(KM)
    sil = pd.read_csv(SIL)
    fams = ["real attack", "abrupt_jump", "constant_offset", "linear_drift",
            "exponential_ramp", "low_amplitude_wander"]  # fmt: skip
    tab = pd.crosstab(a["cluster"], a["family"], normalize="columns")
    tab = tab.reindex(columns=[f for f in fams if f in tab.columns])
    counts = a["family"].value_counts()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                 gridspec_kw={"width_ratios": [2.2, 1]})  # fmt: skip
    im = a1.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(tab.to_numpy()):
        a1.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 else S.INK)  # fmt: skip
    a1.set_xticks(range(tab.shape[1]),
                  [S.FAMILIES.get(c, "Real attack").replace(" ", "\n")
                   + f"\n(n={counts[c]})" for c in tab.columns], fontsize=8)  # fmt: skip
    a1.set_yticks(range(tab.shape[0]), [f"cluster {c + 1}" for c in tab.index])
    a1.grid(False)
    k = tab.shape[0]
    row = km[(km["description"] == "random forest, check ratios") & (km["k"] == k)]
    ari = float(row["ari_vs_family"].iloc[0])
    a1.set_title(
        f"K-Means (k = {k}) on the random forest's first alarms: share of "
        f"each family per cluster (ARI {ari:.2f})",
        fontsize=10,
    )
    fig.colorbar(im, ax=a1, fraction=0.03, label="Share of the family's alarms")
    a2.plot(sil["k"], sil["silhouette"], color=S.colour("random_forest"), marker="o",
            linewidth=2)  # fmt: skip
    a2.set_xlabel("Number of clusters k")
    a2.set_ylabel("Silhouette score")
    a2.set_title("Cluster quality by k", fontsize=10)
    a2.set_ylim(0, max(0.5, sil["silhouette"].max() * 1.2))
    fig.tight_layout()
    S.note(fig, "Each alarm is described by its check ratios (value / normal limit). "
                "ARI near 0 means the clusters do not recover the attack families; only "
                "abrupt jumps form their own group. Silhouette peaks at a low k: the "
                "alarms do not fall into clear types.")  # fmt: skip
    return S.save(fig, "M10_alarm_clusters",
                  "Unsupervised clusters of alarms do not recover the attack families "
                  "(ARI 0.09-0.16).",
                  [str(CLUSTERS.relative_to(S.ROOT)), str(KM.relative_to(S.ROOT)),
                   str(SIL.relative_to(S.ROOT))], ME)  # fmt: skip


def make():
    m10_shap()
    m10_attribution()
    m10_clusters()
