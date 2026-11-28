"""Step K, plan M11: size and latency before vs after export / quantization, and the
smaller random forests."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aerovigil.figures import style as S

ME = "src/aerovigil/figures/edge_figs.py"
TABLES = S.ROOT / "reports" / "tables"
EDGE = TABLES / "F_edge_models.csv"
FORESTS = TABLES / "F_compact_forests.csv"
FORMS = [("pytorch", "PyTorch"), ("sklearn", "scikit-learn"), ("onnx_fp32", "ONNX float32"),
         ("onnx_int8", "ONNX int8")]  # fmt: skip
HATCH = {"pytorch": None, "sklearn": None, "onnx_fp32": "//", "onnx_int8": ".."}


def m11_size_latency():
    t = pd.read_csv(EDGE)
    t = t[t["form"].isin([f for f, _ in FORMS]) & t["size_kb"].notna()]
    dets = ["lstm_ae", "lstm_supervised", "random_forest"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    labels, sizes, lats, cols, hatches, catch = [], [], [], [], [], []
    for d in dets:
        for f, flab in FORMS:
            r = t[(t["detector"] == d) & (t["form"] == f)]
            if r.empty:
                continue
            r = r.iloc[0]
            labels.append(f"{S.label(d)}\n{flab}")
            sizes.append(r["size_kb"])
            lats.append(r["latency_ms_median"])
            cols.append(S.colour(d))
            hatches.append(HATCH[f])
            catch.append(r["consistent_detected"])
    y = np.arange(len(labels))
    for ax, vals, xl, fmt in ((axes[0], sizes, "Model file size (KB, log)", "{:,.0f} KB"),
                              (axes[1], lats, "Time per window, one CPU thread (ms, log)",
                               "{:.2f} ms"),
                              (axes[2], catch, "Velocity-consistent attacks caught",
                               "{:.0%}")):  # fmt: skip
        ax.barh(y, vals, color=cols, hatch=hatches, edgecolor=S.SURFACE)
        for yi, v in zip(y, vals):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7)
        ax.set_xlabel(xl)
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(y), fontsize=8)
        ax.invert_yaxis()
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[2].set_xlim(0, 1.15)
    fig.suptitle("Edge deployment: size, speed and detection before and after export "
                 "and int8 quantization")  # fmt: skip
    fig.tight_layout()
    S.note(fig, "Mean of 5 seeds; latency = median over test windows on one thread of an "
                "i7-7600U laptop (timings vary about 10% between runs). int8 = ONNX "
                "Runtime dynamic quantization (uint8 weights, chosen on validation "
                "windows); it does not apply to trees. Hatching: plain = original "
                "framework, // = ONNX float32, dots = ONNX int8.")  # fmt: skip
    return S.save(fig, "M11_size_latency",
                  "Model size, time per window and detection before and after ONNX "
                  "export and int8 quantization: the LSTMs shrink 2.5x with the same "
                  "detections.", str(EDGE.relative_to(S.ROOT)), ME)  # fmt: skip


def m11_forests():
    t = pd.read_csv(FORESTS)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t["onnx_kb"], t["consistent_detected"], color=S.colour("random_forest"),
            marker="P", markersize=8, linewidth=2, label="Velocity-consistent caught")  # fmt: skip
    ax.plot(t["onnx_kb"], t["syn_detected"], color=S.colour("random_forest"),
            marker="o", markersize=6, linewidth=1.5, linestyle="--",
            label="All synthetic attacks caught")  # fmt: skip
    ax.plot(t["onnx_kb"], t["real_detected"], color=S.INK, marker="s", markersize=5,
            linewidth=1.2, linestyle=":", label="Dataset's own attacks caught")  # fmt: skip
    for _, r in t.iterrows():
        depth = "full" if str(r["max_depth"]) == "none" else r["max_depth"]
        ax.annotate(f"{r['trees']} trees\ndepth {depth}", (r["onnx_kb"], 0.61),
                    ha="center", va="bottom", fontsize=7, color=S.INK2)  # fmt: skip
        ax.axvline(r["onnx_kb"], color=S.GRID, linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0.6, 1.03)
    ax.set_xlabel("ONNX file size (KB, log)")
    ax.set_ylabel("Share of attacked test flights caught")
    ax.set_title("Smaller random forests: size against detection")
    ax.legend(loc="center right", fontsize=8)
    S.note(fig, "Each forest retrained on the seen-mission training windows (5 seeds) "
                "with its own threshold at 1 false alarm per flight-hour. A 20-tree, "
                "depth-8 forest is 100 KB and still catches every real attack.")  # fmt: skip
    return S.save(fig, "M11_compact_forests",
                  "How much detection smaller random forests give up for a smaller file.",
                  str(FORESTS.relative_to(S.ROOT)), ME)  # fmt: skip


def make():
    m11_size_latency()
    m11_forests()
