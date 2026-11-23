r"""Step 9: figures for the report.

  M7_fig1_window_auc.png        ROC-AUC per attack family: AE vs supervised (seen / unseen)
  M7_fig2_ae_timeline.png       AE score, threshold and per-feature error for one attacked flight
  M7_fig3_detection_by_attack   detection rate per attack, with the Module 06 detectors alongside
  M7_fig4_stealth_preview.png   displacement before the alarm against drift rate

Usage:
    python src\aerovigil\lstm\plot_lstm.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.lstm.common import load_yaml, out_dir, resolve  # noqa: E402
from aerovigil.lstm.datasets import ATTACK_TYPES  # noqa: E402

COLORS = {
    "lstm_ae": "tab:blue",
    "lstm_supervised": "tab:orange",
    "lstm_supervised_unseen": "tab:red",
    "cusum": "tab:green",
    "nis_test": "tab:gray",
}
LABELS = {
    "lstm_ae": "LSTM autoencoder (normal only)",
    "lstm_supervised": "Supervised LSTM (seen)",
    "lstm_supervised_unseen": "Supervised LSTM (family held out)",
    "cusum": "Module 06 CUSUM",
    "nis_test": "Module 06 NIS test",
}


def fig_window_auc(res, path):
    groups = [g for g in ["original"] + ATTACK_TYPES if g in set(res["group"])]
    dets = ["lstm_ae", "lstm_supervised", "lstm_supervised_unseen"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.26
    for i, d in enumerate(dets):
        sub = res[res["detector"] == d].set_index("group")
        vals = [sub["roc_auc_mean"].get(g, np.nan) for g in groups]
        err = [np.nan_to_num(sub["roc_auc_std"].get(g, 0.0)) for g in groups]
        ax.bar(
            np.arange(len(groups)) + (i - 1) * w,
            vals,
            w,
            yerr=err,
            capsize=2,
            color=COLORS[d],
            label=LABELS[d],
        )
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("ROC-AUC on test windows (mean +- sd over seeds)")
    ax.set_title(
        "Window-level detection: one-class vs supervised, seen and unseen attack families"
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_timeline(out, meta, flights, features, path):
    thr = json.loads((out / "thresholds.json").read_text())
    idx = pd.read_csv(out / "scores" / "index.csv")
    ae = idx[(idx["kind"] == "ae")].iloc[0]
    cand = flights[
        (flights["origin"] == "syn")
        & (flights["split"] == "test")
        & (flights["attack_type"] == "linear_drift")
        & (flights["velocity_mode"] == "position_only")
    ]
    if cand.empty:
        cand = flights[(flights["origin"] == "syn") & (flights["split"] == "test")]
    if cand.empty:
        return
    fl = cand.iloc[len(cand) // 2]
    ii = np.flatnonzero((meta["flight_id"] == fl["flight_id"]).to_numpy())
    score = np.load(out / "scores" / f"{ae['name']}.npy")[ii]
    ferr = np.load(out / "scores" / f"ae_feature_errors_seed{ae['seed']}.npy")[
        ii
    ].astype(float)
    t = meta["window_end_s"].to_numpy()[ii]
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    axes[0].plot(t, score, color="tab:blue", lw=1.2, label="reconstruction error")
    axes[0].axhline(
        thr[ae["name"]]["threshold"],
        color="k",
        ls="--",
        lw=0.9,
        label="threshold (1 alarm/h budget)",
    )
    axes[0].set_ylabel("AE window score")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title(
        f"{fl['attack_type']} ({fl['velocity_mode']}) - {ae['name']}", fontsize=10
    )
    im = axes[1].imshow(
        np.log10(ferr.T + 1e-4),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[t[0], t[-1], -0.5, ferr.shape[1] - 0.5],
    )
    axes[1].set_yticks(range(len(features)))
    axes[1].set_yticklabels(features, fontsize=6)
    axes[1].set_xlabel("window end time (s)")
    axes[1].set_ylabel("feature")
    fig.colorbar(im, ax=axes.tolist(), label="log10 reconstruction error")
    for a in axes:
        a.axvspan(fl["attack_start_s"], fl["attack_end_s"], color="tab:red", alpha=0.10)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_detection(summary, k6, path):
    syn = summary[summary["set"] == "syn"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, mode in zip(axes, ("consistent", "position_only")):
        w, types = 0.16, ATTACK_TYPES
        series = []
        for d in ("lstm_ae", "lstm_supervised", "lstm_supervised_unseen"):
            s = syn[(syn["detector"] == d) & (syn["velocity_mode"] == mode)].set_index(
                "attack_type"
            )
            series.append((d, [s["detection_rate"].get(t, np.nan) for t in types]))
        if k6 is not None:
            for d in ("cusum", "nis_test"):
                s = k6[
                    (k6["set"] == "syn")
                    & (k6["detector"] == d)
                    & (k6["velocity_mode"] == mode)
                ].set_index("attack_type")
                series.append((d, [s["detection_rate"].get(t, np.nan) for t in types]))
        for i, (d, vals) in enumerate(series):
            ax.bar(
                np.arange(len(types)) + (i - len(series) / 2 + 0.5) * w,
                vals,
                w,
                color=COLORS[d],
                label=LABELS[d],
            )
        ax.set_xticks(np.arange(len(types)))
        ax.set_xticklabels(types, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{mode} velocity")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("detection rate (test flights)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Detection at 1 false alarm per flight-hour (each model calibrated on its own normal data)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_sweep(frontier, k6, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for d in ("lstm_ae", "lstm_supervised"):
        s = frontier[frontier["detector"] == d].sort_values("drift_rate_mps")
        if len(s):
            ax.plot(
                s["drift_rate_mps"],
                s["median_displacement_m"].clip(lower=0.01),
                "o-",
                color=COLORS[d],
                label=LABELS[d],
            )
    if k6 is not None:
        for d in ("cusum", "nis_test"):
            s = k6[k6["detector"] == d].sort_values("drift_rate_mps")
            ax.plot(
                s["drift_rate_mps"],
                s["median_displacement_m"].clip(lower=0.01),
                "s--",
                color=COLORS[d],
                label=LABELS[d],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("attacker drift rate (m/s)")
    ax.set_ylabel("median displacement before the alarm (m)")
    ax.set_title("Preliminary stealth curve (consistent drift, test flights)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config, output_dir=None, fig_dir=None):
    cfg = load_yaml(config)
    out = out_dir(cfg, output_dir)
    fig_dir = Path(fig_dir) if fig_dir else ROOT / "reports" / "figures" / "module07"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(out / "results_window.csv")
    summary = pd.read_csv(out / "summary_by_attack.csv")
    frontier = pd.read_csv(out / "sweep_frontier.csv")
    meta = pd.read_parquet(out / "windows_meta.parquet")
    flights = pd.read_parquet(out / "flights.parquet").set_index(
        "flight_id", drop=False
    )
    features = json.loads((out / "scaler.json").read_text())["features"]
    k_dir = resolve(cfg["paths"]["kalman_dir"])
    k_sum = (
        pd.read_csv(k_dir / "summary_by_attack.csv")
        if (k_dir / "summary_by_attack.csv").exists()
        else None
    )
    k_sw = (
        pd.read_csv(k_dir / "sweep_frontier.csv")
        if (k_dir / "sweep_frontier.csv").exists()
        else None
    )
    fig_window_auc(res, fig_dir / "M7_fig1_window_auc.png")
    fig_timeline(out, meta, flights, features, fig_dir / "M7_fig2_ae_timeline.png")
    fig_detection(summary, k_sum, fig_dir / "M7_fig3_detection_by_attack.png")
    fig_sweep(frontier, k_sw, fig_dir / "M7_fig4_stealth_preview.png")
    print(
        f"Figures saved to {fig_dir}"
        + ("" if k_sum is not None else " (Module 06 tables not found: omitted)")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--fig-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.output_dir, a.fig_dir)


if __name__ == "__main__":
    main()
