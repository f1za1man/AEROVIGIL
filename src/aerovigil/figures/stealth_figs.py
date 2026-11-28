"""Step K, plan M8: the adaptive attacker (Module 08 search, oracle self-test) and the
Stealth Frontier (Step D: fixed, adaptive and white-box attackers, hardening).

Inside a panel the detector keeps its own colour; the attacker is told apart by grey /
black and by line style, so colours still mean the same detector in every figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

from aerovigil.figures import style as S

ME = "src/aerovigil/figures/stealth_figs.py"
TABLES = S.ROOT / "reports" / "tables"
M08 = S.ROOT / "data" / "processed" / "module08"
SEARCH = M08 / "adaptive_search.csv"
SELFTEST = M08 / "adaptive_selftest.csv"
ORDER = ["kalman", "velocity", "fused", "cusum", "consistency", "random_forest",
         "lstm_ae", "lstm_supervised"]  # fmt: skip


def table(name):
    return pd.read_csv(TABLES / name)


def plain(v, _pos=None):
    return f"{v:g}"


def log_axes(ax, log_y=True):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))


def m8_hill_climber():
    s = pd.read_csv(SEARCH)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    searches = list(dict.fromkeys(s["search"]))
    rates = [x for x in searches if x.startswith("mean_rate")]
    ramp = plt.get_cmap("Blues")
    for i, name in enumerate(searches):
        g = s[s["search"] == name].sort_values("eval")
        best = np.maximum.accumulate(g["score"].to_numpy())
        if name in rates:
            col = ramp(0.35 + 0.65 * rates.index(name) / max(1, len(rates) - 1))
            lab, ls = name.replace("mean_rate=", "") + " m/s", "-"
        else:
            col, lab, ls = S.INK, "no speed limit", "--"
        ax.plot(g["eval"], np.maximum(best, 1e-2), color=col, linewidth=2,
                linestyle=ls)  # fmt: skip
        ax.annotate(lab, (g["eval"].iloc[-1], max(best[-1], 1e-2)), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, va="center")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel(
        "Evaluations (one evaluation = one attack tried on 5 practice flights)"
    )
    ax.set_ylabel("Best score so far (m moved unseen, log)")
    ax.set_title("Module 08 black-box attacker: hill-climber progress per speed")
    ax.set_xlim(right=s["eval"].max() * 1.12)
    S.note(fig, "Attacking the Kalman chi-square test, which it sees only as alarms. "
                "Score = distance the drone is moved on the practice (validation) "
                "flights before an alarm, with the measured mean drift speed held to "
                "its target.")  # fmt: skip
    return S.save(fig, "M8_hill_climber",
                  "How the black-box attacker's best attack improves with each try, "
                  "one curve per mean drift speed.", str(SEARCH.relative_to(S.ROOT)), ME)  # fmt: skip


def m8_selftest():
    t = pd.read_csv(SELFTEST)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    rows = [("module06", "Module 06's original frontier", "#ec4647", "o", "-"),
            ("module04", "Re-injected with the old onset (oracle check)", S.INK, "x",
             ":"),
            ("eased", "Same drift, eased in over 20 s", "#257de0", "s", "--")]  # fmt: skip
    for col, lab, c, mk, ls in rows:
        ax.plot(t["drift_rate_mps"], t[col], color=c, marker=mk, linestyle=ls,
                linewidth=2, markersize=7, label=lab)  # fmt: skip
    log_axes(ax, log_y=False)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Drift speed (m/s)")
    ax.set_ylabel("Share of test flights detected")
    ax.set_title("Oracle self-test: the old frontier was the switch-on")
    ax.legend(loc="lower right", fontsize=8)
    S.note(fig, "Kalman chi-square test on the 21 test flights. With the old onset the "
                "attacker's oracle reproduces Module 06's frontier closely (43% vs 57% "
                "at 0.25 m/s, the same elsewhere); easing the same drift in removes the "
                "detections at 0.25-0.5 m/s. From 1 m/s the drift itself is caught.")  # fmt: skip
    return S.save(fig, "M8_oracle_selftest",
                  "The attacker's oracle reproduces the old frontier, and an eased "
                  "onset removes its 0.25-0.5 m/s detections: those came from the "
                  "switch-on, not the drift.",
                  str(SELFTEST.relative_to(S.ROOT)), ME)  # fmt: skip


ATTACKERS = [("fixed", "Fixed drift (sweep)", S.NEUTRAL, "o", "--"),
             ("adaptive", "Adaptive black-box attacker", None, "s", "-"),
             ("white", "White-box worst case", S.INK, "^", ":")]  # fmt: skip


def frontier_sets():
    fixed = table("D_frontier_fixed.csv").rename(columns={"drift_rate_mps":
                                                          "mean_rate_mps"})  # fmt: skip
    a = table("D_frontier_adaptive.csv")
    a = a[a["mean_rate_mps"].notna() & (a["tag"] == "before")]
    return {"fixed": fixed, "adaptive": a, "white": table("D_whitebox.csv")}


def whole_drift():
    f = table("D_frontier_fixed.csv")
    return f.groupby("drift_rate_mps")["median_final_offset_m"].median()


def frontier(value, lo, hi, log_y, ylabel, name, title, caption):
    sets = frontier_sets()
    ref = whole_drift()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharex=True, sharey=True)
    for ax, det in zip(axes.ravel(), ORDER):
        if log_y:
            ax.plot(
                ref.index, ref.values, color=S.INK2, linewidth=1, linestyle=(0, (1, 3))
            )
        for key, _, col, mk, ls in ATTACKERS:
            g = sets[key]
            g = g[g["detector"] == det].sort_values("mean_rate_mps")
            if g.empty:
                continue
            c = col or S.colour(det)
            x = g["mean_rate_mps"].to_numpy(float)
            ax.fill_between(x, g[lo], g[hi], color=c, alpha=0.13, linewidth=0)
            ax.plot(x, g[value], color=c, marker=mk, linestyle=ls, linewidth=2,
                    markersize=5, markeredgecolor="white")  # fmt: skip
        log_axes(ax, log_y)
        if not log_y:
            ax.set_ylim(-0.03, 1.03)
        ax.set_title(S.label(det), fontsize=10, color=S.colour(det), loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Attacker's mean drift speed (m/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=c or S.INK2, marker=mk, linestyle=ls, linewidth=2,
                      label=lab + (" (detector colour)" if c is None else ""))
               for _, lab, c, mk, ls in ATTACKERS]  # fmt: skip
    if log_y:
        handles.append(Line2D([], [], color=S.INK2, linestyle=(0, (1, 3)),
                              label="Whole drift (never caught)"))  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    S.note(fig, "21 test flights, budget 1 false alarm per flight-hour for every "
                "detector; learned models averaged over 5 seeds; bands: bootstrap 95% "
                "over flights. Seen missions only. The Stealth Frontier is this "
                "project's own measure.")  # fmt: skip
    return S.save(fig, name, caption,
                  ["reports/tables/D_frontier_fixed.csv",
                   "reports/tables/D_frontier_adaptive.csv",
                   "reports/tables/D_whitebox.csv"], ME)  # fmt: skip


def m8_frontier():
    return frontier("D_m", "D_ci_low", "D_ci_high", True,
                    "Metres moved before the first alarm", "M8_stealth_frontier",
                    "Stealth Frontier: how far each attacker moves the drone before "
                    "the first alarm",
                    "Headline: distance the drone is moved before the first alarm vs "
                    "the attacker's speed, for the fixed, adaptive and white-box "
                    "attackers, per detector.")  # fmt: skip


def m8_detection():
    return frontier("detected", "detected_ci_low", "detected_ci_high", False,
                    "Share of test flights detected", "M8_detection_by_attacker",
                    "Share of attacked test flights detected, per attacker",
                    "Share of flights each detector catches against the fixed, adaptive "
                    "and white-box attackers, per speed.")  # fmt: skip


def m8_hardening():
    t = table("D_hardening.csv")
    a = t[(t["attack"] == "adaptive") & t["mean_rate_mps"].notna()]
    ref = whole_drift()
    tags = [("before", "Before hardening", None, "o", "-"),
            ("after_replay", "After: old winning attacks replayed", S.NEUTRAL, "s", "--"),
            ("after_research", "After: attacker searches again", S.INK, "^", ":")]  # fmt: skip
    fa = t[t["fa_per_hour"].notna()].groupby(["detector", "tag"])["fa_per_hour"].first()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, det in zip(axes, ("random_forest", "lstm_supervised")):
        ax.plot(ref.index, ref.values, color=S.INK2, linewidth=1, linestyle=(0, (1, 3)))
        for tag, _, col, mk, ls in tags:
            g = a[(a["detector"] == det) & (a["tag"] == tag)].sort_values(
                "mean_rate_mps"
            )
            c = col or S.colour(det)
            x = g["mean_rate_mps"].to_numpy(float)
            ax.fill_between(x, g["D_ci_low"], g["D_ci_high"], color=c, alpha=0.13,
                            linewidth=0)  # fmt: skip
            ax.plot(x, g["D_m"], color=c, marker=mk, linestyle=ls, linewidth=2,
                    markersize=5, markeredgecolor="white")  # fmt: skip
        log_axes(ax)
        ax.set_title(f"{S.label(det)}: false alarms {fa[(det, 'before')]:.2f} -> "
                     f"{fa[(det, 'after')]:.2f} per hour", fontsize=10,
                     color=S.colour(det), loc="left")  # fmt: skip
        ax.set_xlabel("Attacker's mean drift speed (m/s)")
    axes[0].set_ylabel("Metres moved before the first alarm")
    handles = [Line2D([], [], color=c or S.INK2, marker=mk, linestyle=ls, linewidth=2,
                      label=lab + (" (detector colour)" if c is None else ""))
               for _, lab, c, mk, ls in tags]  # fmt: skip
    handles.append(Line2D([], [], color=S.INK2, linestyle=(0, (1, 3)),
                          label="Whole drift (never caught)"))  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Stealth Frontier before and after adversarial hardening")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.note(fig, "Hardening = retraining with the attacks that got through added to the "
                "training data, threshold recalibrated to the same budget. 21 test "
                "flights x 5 seeds; bands: bootstrap 95% over flights.")  # fmt: skip
    return S.save(fig, "M8_hardening",
                  "Headline: the Stealth Frontier of the random forest and the "
                  "supervised LSTM before and after hardening, with 95% bands and the "
                  "false-alarm cost.", "reports/tables/D_hardening.csv", ME)  # fmt: skip


def make():
    m8_hill_climber()
    m8_selftest()
    m8_frontier()
    m8_detection()
    m8_hardening()
