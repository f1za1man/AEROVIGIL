import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "module03"
WIN_DIR = OUT_DIR / "windows"
FEAT_DIR = OUT_DIR / "features"
FIG_DIR = ROOT / "reports" / "figures" / "module03"
TS_REPORT = OUT_DIR / "timestamp_report.csv"
SPLITS = ("train", "validation", "test")
SEED = 42


def load_split(name):
    d = np.load(WIN_DIR / f"windows_{name}.npz", allow_pickle=False)
    X = d["X"]
    means = X.mean(axis=1, dtype="float64")  # (windows, features)
    frac = d["attack_fraction"].astype("float64")
    names = [str(s) for s in d["feature_names"]]
    del X
    return means, frac, names


def attack_group(frac):
    g = np.full(len(frac), "normal", dtype=object)
    g[(frac > 0) & (frac < 1)] = "mixed"
    g[frac >= 1] = "attacked"
    return g


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("AEROVIGIL - EXPLORATORY ANALYSIS AND PCA")
    print("=" * 65)

    data, names = {}, None
    for s in SPLITS:
        m, f, names = load_split(s)
        data[s] = (m, f)
        print(f"{s:<11}: {m.shape[0]:,} windows x {m.shape[1]} features")

    # ---- PCA: fit on training windows only ------------------------------
    Xtr = data["train"][0]
    pca = PCA(n_components=min(10, Xtr.shape[1]), random_state=SEED)
    pca.fit(Xtr)
    scores = {s: pca.transform(data[s][0]) for s in SPLITS}

    ev = pca.explained_variance_ratio_
    print("\nEXPLAINED VARIANCE (fitted on training windows)")
    for i, (r, c) in enumerate(zip(ev, np.cumsum(ev)), start=1):
        print(f"  PC{i:<2}: {r * 100:6.2f}%   cumulative {c * 100:6.2f}%")

    print("\nTOP LOADINGS")
    for pc in range(2):
        order = np.argsort(-np.abs(pca.components_[pc]))[:5]
        print(
            f"  PC{pc + 1}: "
            + ", ".join(f"{names[i]} ({pca.components_[pc][i]:+.2f})" for i in order)
        )

    # ---- PCA scatter plots ---------------------------------------------------
    colours = {"normal": "#1f77b4", "mixed": "#ff7f0e", "attacked": "#d62728"}
    allpc = np.vstack([scores[s][:, :2] for s in SPLITS])
    lo, hi = allpc.min(axis=0), allpc.max(axis=0)
    pad = 0.05 * (hi - lo)
    rng = np.random.default_rng(SEED)

    for s in SPLITS:
        sc, frac = scores[s], data[s][1]
        grp = attack_group(frac)
        order = rng.permutation(len(sc))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for label in ("normal", "mixed", "attacked"):
            sel = order[grp[order] == label]
            ax.scatter(
                sc[sel, 0],
                sc[sel, 1],
                s=7,
                alpha=0.45,
                c=colours[label],
                label=f"{label} ({(grp == label).sum():,})",
                linewidths=0,
            )
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% variance)")
        ax.set_title(
            f"PCA of window-mean features - {s} " f"(PCA fitted on training only)"
        )
        ax.legend(markerscale=3, title="Attack content of window")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"pca_{s}.png", dpi=150)
        plt.close(fig)

    # ---- feature correlation (training windows) -----------------------------
    corr = pd.DataFrame(Xtr, columns=names).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Feature correlation (training windows, window-mean features)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)

    # ---- missing values (all flights, physical-unit features) ---------------
    files = sorted(FEAT_DIR.glob("*.parquet"))
    nan_counts = pd.Series(0, index=names, dtype="int64")
    total = 0
    for p in files:
        g = pd.read_parquet(p, columns=names)
        nan_counts += g.isna().sum()
        total += len(g)
    pct = (nan_counts / total * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(pct.index, pct.values, color="#4c72b0")
    ax.set_xlabel("Missing values (% of 7,893,049 rows)")
    ax.set_title("Missing (NaN) values per engineered feature")
    ax.grid(axis="x", alpha=0.3)
    for y_pos, v in enumerate(pct.values):
        ax.text(v, y_pos, f" {v:.2f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    # ---- sampling intervals (from timestamp report) -------------------------
    ts = pd.read_csv(TS_REPORT)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ms = ts["median_interval_s"] * 1000
    axes[0].hist(ms, bins=40, color="#4c72b0")
    axes[0].set_xlabel("Median sampling interval (ms)")
    axes[0].set_ylabel("Flights")
    axes[0].set_title("Median interval per flight")
    rate = ts["estimated_hz"].round(0).astype(int).value_counts().sort_index()
    axes[1].bar(rate.index.astype(str), rate.values, color="#55a868")
    axes[1].set_xlabel("Estimated rate (Hz)")
    axes[1].set_ylabel("Flights")
    axes[1].set_title("Sampling rate per flight")
    for i, v in enumerate(rate.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    gap = ts.loc[ts["large_gaps"] > 0, "max_gap_s"]
    axes[2].hist(gap, bins=25, color="#c44e52")
    axes[2].set_xlabel("Largest gap in flight (s)")
    axes[2].set_ylabel("Flights")
    axes[2].set_title(f"Dropouts ({len(gap)} flights with large gaps)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sampling_intervals.png", dpi=150)
    plt.close(fig)

    # ---- save PCA summary ----------------------------------------------------
    summary = {
        "fitted_on": "train windows only",
        "n_training_windows": int(Xtr.shape[0]),
        "input": "mean of each normalized feature over each 5 s window",
        "features": names,
        "explained_variance_ratio": [float(v) for v in ev],
        "cumulative_explained_variance": [float(v) for v in np.cumsum(ev)],
    }
    with (OUT_DIR / "pca_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    expected = [
        "pca_train.png",
        "pca_validation.png",
        "pca_test.png",
        "feature_correlation.png",
        "missing_values.png",
        "sampling_intervals.png",
    ]
    leaked = [
        n for n in names if any(w in n.lower() for w in ("attack", "label", "flight"))
    ]
    checks = {
        "PCA fitted on training windows only (11,579)": pca.n_samples_
        == data["train"][0].shape[0],
        "No NaN in PCA input": all(np.isfinite(data[s][0]).all() for s in SPLITS),
        "No label/ID column in PCA input": not leaked,
        "All six figures created": all((FIG_DIR / n).exists() for n in expected),
        "Explained variance recorded": (OUT_DIR / "pca_summary.json").exists(),
    }
    print("\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nFigures: {FIG_DIR}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("PCA checks FAILED")


if __name__ == "__main__":
    main()
