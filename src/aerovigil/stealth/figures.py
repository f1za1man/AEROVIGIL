r"""Plan M8 (Step D) figures: the Stealth Frontier and hardening before/after.

D1_stealth_frontier.png  one panel per detector: distance the drone is moved before the
                         first alarm against the attacker's mean drift speed, for the
                         fixed drift (Module 04 sweep), the black-box adaptive attacker
                         and the white-box worst case, with bootstrap 95% bands.
D2_detection.png         the same layout, share of flights detected.
D3_hardening.png         the random forest and the supervised LSTM against the adaptive
                         attacker, before and after hardening (replay of the old
                         winners, and a fresh search), five seeds, 95% bands.

All at the same budget of 1 false alarm per flight-hour. The dashed grey line is the
whole drift (never caught): a curve on it means the detector saw nothing.

Usage:
    python src\aerovigil\stealth\figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from aerovigil.stealth import detectors as D  # noqa: E402

FIG = D.ROOT / "reports" / "figures" / "stealth"
TABLES = D.ROOT / "reports" / "tables"
# categorical slots 1-3 of the validated reference palette (all-pairs safe)
SERIES = ("#257de0", "#f47230", "#00b894")
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
MARKERS = ("o", "s", "^")
TITLES = {"kalman": "Kalman chi-square (M6)", "cusum": "CUSUM (M6)",
          "velocity": "Velocity channel (M10)", "fused": "Fused (M10)",
          "consistency": "Consistency (M9)", "random_forest": "Random forest (M5)",
          "lstm_ae": "LSTM autoencoder (M7)",
          "lstm_supervised": "Supervised LSTM (M7)"}  # fmt: skip
ORDER = ("kalman", "cusum", "velocity", "fused", "consistency", "random_forest",
         "lstm_ae", "lstm_supervised")  # fmt: skip


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })  # fmt: skip


def plain(v, _pos=None):
    """0.01, 0.1, 1, 10, 100 rather than powers of ten."""
    return f"{v:g}"


def log_axes(ax, log_y=True):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))


def read(name):
    p = TABLES / name
    return pd.read_csv(p) if p.is_file() else None


def curves():
    """(label, frame with mean_rate_mps, value and band columns) per attacker."""
    out = []
    fixed = read("D_frontier_fixed.csv")
    if fixed is not None:
        out.append(
            (
                "Fixed drift (sweep)",
                fixed.rename(columns={"drift_rate_mps": "mean_rate_mps"}),
            )
        )
    adaptive = read("D_frontier_adaptive.csv")
    if adaptive is not None:
        a = adaptive[adaptive["mean_rate_mps"].notna()]
        if "tag" in a:
            a = a[a["tag"] == "before"]
        out.append(("Adaptive, black-box", a))
    white = read("D_whitebox.csv")
    if white is not None:
        out.append(("White-box worst case", white))
    return out


def never_caught():
    """Median whole drift per speed (the attacker's distance when never caught)."""
    fixed = read("D_frontier_fixed.csv")
    if fixed is None:
        return None
    return fixed.groupby("drift_rate_mps")["median_final_offset_m"].median()


def panel(ax, det, sets, value, lo, hi, log_y):
    for (label, frame), color, marker in zip(sets, SERIES, MARKERS):
        g = frame[frame["detector"] == det].sort_values("mean_rate_mps")
        if not len(g):
            continue
        x = g["mean_rate_mps"].to_numpy(float)
        ax.fill_between(x, g[lo], g[hi], color=color, alpha=0.15, lw=0)
        ax.plot(x, g[value], color=color, lw=2, marker=marker, ms=5,
                mec="white", mew=1, label=label)  # fmt: skip
    log_axes(ax, log_y)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(TITLES.get(det, det), loc="left")


def frontier_figure(value="D_m", lo="D_ci_low", hi="D_ci_high", log_y=True,
                    ylabel="metres moved before the first alarm",
                    name="D1_stealth_frontier.png", title=None):  # fmt: skip
    sets = curves()
    if not sets:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True)
    ref = never_caught() if log_y else None
    for ax, det in zip(axes.ravel(), ORDER):
        if ref is not None:
            ax.plot(ref.index, ref.values, color=INK2, lw=1, ls="--",
                    label="Whole drift (never caught)")  # fmt: skip
        panel(ax, det, sets, value, lo, hi, log_y)
        if not log_y:
            ax.set_ylim(-0.03, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("attacker's mean drift speed (m/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(
        0.01,
        0.005,
        "Test flights, budget 1 false alarm per flight-hour; bands: "
        "bootstrap 95% over flights; learned models averaged over 5 seeds. "
        "Stealth Frontier is this project's own metric.",
        color=INK2,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    return FIG / name


def hardening_figure():
    t = read("D_hardening.csv")
    if t is None:
        return None
    a = t[(t["attack"] == "adaptive") & t["mean_rate_mps"].notna()]
    tags = (("before", "Before hardening"),
            ("after_replay", "After: replay of the old winners"),
            ("after_research", "After: attacker searches again"))  # fmt: skip
    models = [
        m for m in ("random_forest", "lstm_supervised") if m in set(a["detector"])
    ]
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4),
                             sharey=True, squeeze=False)  # fmt: skip
    ref = never_caught()
    for ax, model in zip(axes[0], models):
        if ref is not None:
            ax.plot(ref.index, ref.values, color=INK2, lw=1, ls="--",
                    label="Whole drift (never caught)")  # fmt: skip
        for (tag, label), color, marker in zip(tags, SERIES, MARKERS):
            g = a[(a["detector"] == model) & (a["tag"] == tag)]
            g = g.sort_values("mean_rate_mps")
            if not len(g):
                continue
            x = g["mean_rate_mps"].to_numpy(float)
            ax.fill_between(x, g["D_ci_low"], g["D_ci_high"], color=color,
                            alpha=0.15, lw=0)  # fmt: skip
            ax.plot(x, g["D_m"], color=color, lw=2, marker=marker, ms=5,
                    mec="white", mew=1, label=label)  # fmt: skip
        log_axes(ax)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(TITLES[model], loc="left")
        ax.set_xlabel("attacker's mean drift speed (m/s)")
    axes[0, 0].set_ylabel("metres moved before the first alarm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "21 test flights x 5 seeds, budget 1 false alarm per "
             "flight-hour, thresholds recalibrated after hardening; bands: bootstrap "
             "95% over flights.", color=INK2, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(FIG / "D3_hardening.png", dpi=160)
    plt.close(fig)
    return FIG / "D3_hardening.png"


def run():
    style()
    out = [
        frontier_figure(
            title="Stealth Frontier: how far each detector lets the "
            "attacker move the drone"
        ),  # fmt: skip
        frontier_figure(
            "detected",
            "detected_ci_low",
            "detected_ci_high",
            False,
            "share of flights detected",
            "D2_detection.png",
            "Share of attacked test flights detected",
        ),  # fmt: skip
        hardening_figure(),
    ]
    for p in out:
        if p:
            print(f"saved {p}")


if __name__ == "__main__":
    run()
