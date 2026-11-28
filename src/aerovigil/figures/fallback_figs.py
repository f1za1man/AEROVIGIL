"""Step K, plan M12: position error with vs without the dead-reckoning fallback, and how
long dead reckoning stays within 10 m and 25 m (Kaplan-Meier)."""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aerovigil.figures import style as S
from aerovigil.mitigation import fallback as fb
from aerovigil.mitigation import run as MR

ME = "src/aerovigil/figures/fallback_figs.py"
OUT = S.ROOT / "data" / "processed" / "mitigation"
WITH = (S.INK, "-", "With fallback (dead reckoning)")
WITHOUT = ("#9c9c9c", "--", "Without fallback (trusts the spoofed GPS)")


def band(ax, curves, colour, ls, label):
    c = np.asarray(curves, float)
    n = np.sum(np.isfinite(c), axis=0)
    ok = n >= max(3, 0.2 * len(c))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns with no runs left
        med = np.nanmedian(c, axis=0)
        lo, hi = np.nanpercentile(c, 25, axis=0), np.nanpercentile(c, 75, axis=0)
    ax.fill_between(
        MR.GRID_S[ok], lo[ok], hi[ok], color=colour, alpha=0.15, linewidth=0
    )
    ax.plot(
        MR.GRID_S[ok], med[ok], color=colour, linestyle=ls, linewidth=2, label=label
    )


def choice():
    var, rb = (OUT / "choice.txt").read_text().split()
    return var, float(rb)


def m12_error():
    curves = MR.load_curves()
    var, rb = choice()
    panels = [("real", "na", "Dataset's own attacks"),
              ("synthetic", "position_only", "Synthetic, position only"),
              ("synthetic", "consistent", "Synthetic, velocity-consistent")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (kind, mode, title) in zip(axes, panels):
        c = curves[f"{kind}|{mode}|{var}|{rb:g}"]
        band(ax, c[:, 1], *WITHOUT)
        band(ax, c[:, 0], *WITH)
        for lim in MR.LIMITS:
            ax.axhline(lim, color=S.INK2, linewidth=0.8, linestyle=":")
            ax.text(119, lim * 1.1, f"{lim:g} m", ha="right", color=S.INK2, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 1000)
        ax.set_xlim(0, 120)
        ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
        ax.set_title(f"{title} (n = {len(c)})", fontsize=10)
        ax.set_xlabel("Seconds since the alarm")
    axes[0].set_ylabel("Horizontal position error (m, log)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Position error after a spoofing alarm, with and without the fallback")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    S.note(fig, f"Fused detector raises the alarm; the fallback then drops GPS and "
                f"navigates on the IMU (learned bias) and barometer ({var} variant, "
                f"roll back {rb:g} s, chosen on validation flights). Line: median over "
                f"runs; band: middle half. Truth = the unspoofed GPS (about 2 m from "
                f"ground truth).")  # fmt: skip
    return S.save(fig, "M12_fallback_error",
                  "Position error with and without the dead-reckoning fallback after "
                  "an alarm: the fallback helps for about 15-20 s, then drifts.",
                  ["data/processed/mitigation/curves.npz",
                   "data/processed/mitigation/choice.txt"], ME)  # fmt: skip


def m12_within():
    var, _ = choice()
    runs = pd.read_csv(OUT / "coast_runs.csv")
    runs = runs[(runs["set"] == "test") & (runs["variant"] == var)]
    grid = np.arange(0, 120.5, 0.5)
    styles = {10: (S.INK, "-"), 25: ("#9c9c9c", "--")}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, mission in zip(axes, ["all", "curved", "random", "straight"]):
        g = runs if mission == "all" else runs[runs["mission_type"] == mission]
        for lim, (col, ls) in styles.items():
            s = fb.kaplan_meier(g[f"t{lim:g}_s"], g[f"t{lim:g}_hit"].astype(bool), grid)
            ax.step(grid, 100 * s, where="post", color=col, linestyle=ls, linewidth=2,
                    label=f"within {lim:g} m")  # fmt: skip
        ax.set_title(f"{mission.capitalize()} ({len(g)} runs, "
                     f"{g['flight_id'].nunique()} flights)", fontsize=10)  # fmt: skip
        ax.set_xlabel("Seconds without GPS")
        if not g["t10_hit"].astype(bool).any():
            ax.text(60, 50, "flights too short:\nno run drifted 10 m\nbefore it ended",
                    ha="center", va="center", color=S.INK2, fontsize=8)  # fmt: skip
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Runs still within the limit (%)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "How long dead reckoning stays close to the truth (normal test flights)"
    )
    fig.tight_layout()
    S.note(fig, "GPS is dropped at many start points on every normal test flight. "
                "Kaplan-Meier estimate: runs that reach the end of the flight first are "
                "counted as still within the limit up to that point.")  # fmt: skip
    return S.save(fig, "M12_within_limits",
                  "Share of dead-reckoning runs still within 10 m and 25 m of the truth "
                  "over time (median 20 s and 30 s).",
                  "data/processed/mitigation/coast_runs.csv", ME)  # fmt: skip


def make():
    m12_error()
    m12_within()
