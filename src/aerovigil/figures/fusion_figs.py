"""Step K, repo Module 10 (fused detector) and the shared comparison: threat-model
heatmap, false alarms per flight-hour with 95% bounds, detection-delay distributions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

from aerovigil.evaluation import store
from aerovigil.figures import style as S

ME = "src/aerovigil/figures/fusion_figs.py"
M10 = S.ROOT / "data" / "processed" / "module10"
THREAT = M10 / "threat_model.csv"
ALL = S.ROOT / "reports" / "tables" / "C_all_detectors.csv"
PHYS = S.ROOT / "data" / "processed" / "evaluation" / "physics_flights.csv"
RF = S.ROOT / "data" / "processed" / "module05_v2" / "all_models_flight_full.csv"
LSTM = S.ROOT / "data" / "processed" / "module07_v2" / "protocols" / "flights.csv"

ROWS = {"M05 RF": "random_forest", "M06 chi2": "kalman", "M07 AE": "lstm_ae",
        "M07 sup": "lstm_supervised", "M09 consist": "consistency",
        "M10 kalman": "kalman", "M10 velocity": "velocity", "M10 fused": "fused"}  # fmt: skip
COLS = [("real", "Dataset's own\nattacks"), ("position-only", "Synthetic,\nposition only"),
        ("consistent", "Synthetic,\nvelocity-consistent"),
        ("drift slow", "Consistent drift\n0.05-0.1 m/s"),
        ("drift medium", "Consistent drift\n0.25 m/s"),
        ("drift fast", "Consistent drift\n0.5-5 m/s"),
        ("adaptive", "Adaptive,\nspeed-limited"),
        ("adaptive free", "Adaptive,\nno speed limit")]  # fmt: skip


def r10_threat():
    t = pd.read_csv(THREAT).pivot(
        index="detector", columns="attacker", values="detected"
    )
    t = t.reindex(index=list(ROWS), columns=[c for c, _ in COLS])
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    v = t.to_numpy(float)
    im = ax.imshow(np.ma.masked_invalid(v), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_facecolor("#f3f3f3")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            txt = "n/a" if np.isnan(v[i, j]) else f"{v[i, j]:.0%}"
            col = S.INK2 if np.isnan(v[i, j]) else ("white" if v[i, j] > 0.6 else S.INK)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
    labels = [f"{S.label(ROWS[r])} ({r.split()[0]})" for r in t.index]
    ax.set_yticks(range(len(t)), labels, fontsize=9)
    ax.set_xticks(range(len(COLS)), [c[1] for c in COLS], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Share of attacked test flights detected",
                 fraction=0.025)  # fmt: skip
    ax.set_title("Threat model: who catches which attacker (test flights)")
    S.note(fig, "Budget 1 false alarm per flight-hour for every detector. n/a = not "
                "measured for that detector in the threat-model run (the Stealth "
                "Frontier figures cover all eight detectors against the adaptive and "
                "white-box attackers). Slow-drift rows near 5-10% are chance alarms, "
                "not detection.")  # fmt: skip
    return S.save(fig, "R10_threat_model_heatmap",
                  "The threat-model table as a heatmap: every detector catches crude "
                  "attacks; little catches smooth or adaptive drifts.",
                  str(THREAT.relative_to(S.ROOT)), ME)  # fmt: skip


def bounds(k, h):
    """Exact Poisson 95% interval on k alarms in h hours (k may be a seed mean)."""
    lo = 0.0 if k <= 0 else chi2.ppf(0.025, 2 * k) / 2 / h
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2 / h
    return lo, hi


def r10_false_alarms():
    t = pd.read_csv(ALL)
    protos = [("seen", "Seen missions (0.75 h)", "#a3a3a3", "o"),
              ("unseen missions", "Unseen mission (5.25 h)", "#5a5a5a", "s"),
              ("cross-validated false alarms", "Cross-validated, all 144 normal "
               "flights (5.25 h)", S.INK, "D")]  # fmt: skip
    dets = S.DETECTORS
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(dets))
    for i, (p, lab, col, mk) in enumerate(protos):
        g = t[t["protocol"] == p].set_index("detector").reindex(dets)
        off = (i - 1) * 0.25
        for j, d in enumerate(dets):
            k, h = g.loc[d, "fa_alarms"], g.loc[d, "fa_hours"]
            if not np.isfinite(k):
                continue
            lo, hi = bounds(k, h)
            ax.plot([max(lo, 0.03), hi], [y[j] + off] * 2, color=col, linewidth=1.6)
            ax.scatter([max(k / h, 0.03)], [y[j] + off], color=col, marker=mk, s=40,
                       zorder=3, label=lab if j == 0 else None,
                       facecolors=col if k > 0 else "white")  # fmt: skip
    ax.axvline(1.0, color="#ec4647", linestyle="--", linewidth=1.4)
    ax.annotate("budget: 1 per flight-hour", (1.0, len(dets) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color="#ec4647")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.025, 60)
    ax.set_yticks(y, [S.label(d) for d in dets])
    for tick, d in zip(ax.get_yticklabels(), dets):
        tick.set_color(S.colour(d))
    ax.invert_yaxis()
    ax.set_xlabel("False alarms per flight-hour (log scale; open marker = none seen)")
    ax.set_title("False alarms per flight-hour with exact 95% Poisson bounds")
    ax.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.45, -0.1))
    S.note(fig, "Point = alarms / normal flight-hours; line = exact Poisson 95% "
                "interval. Learned models: mean of 5 seeds. The seen rates rest on only "
                "45 minutes of flight, so their bounds are wide; the cross-validated "
                "rate is the one to quote.")  # fmt: skip
    return S.save(fig, "R10_false_alarms",
                  "False alarms per flight-hour of every detector, seen vs unseen "
                  "missions vs cross-validated, with 95% bounds.",
                  str(ALL.relative_to(S.ROOT)), ME)  # fmt: skip


def delays():
    """Per-flight delay on caught attacks, seen protocol, one model per detector."""
    p = pd.read_csv(PHYS)
    p = p[(p["protocol"] == "seen") & p["attacked"] & p["detected"]]
    p["kind"] = np.where(p["set"] == "orig", "real",
                         np.where(p["set"] == "main", p["velocity_mode"], "sweep"))  # fmt: skip
    parts = [p[["detector", "kind", "delay_s"]]]
    rf = store.read_csv(RF)
    rf = rf[(rf["protocol"] == "seen") & (rf["model"] == "random_forest")
            & (rf["seed"] == 42) & rf["attacked"] & rf["detected"]]  # fmt: skip
    rf["kind"] = np.where(rf["origin"] == "orig", "real", rf["velocity_mode"])
    parts.append(rf.assign(detector="random_forest")[["detector", "kind", "delay_s"]])
    ls = store.read_csv(LSTM)
    ls = ls[ls["name"].isin(["ae_seen_s42", "clf_seen_s42"]) & ls["attacked"]
            & ls["detected"]]  # fmt: skip
    ls = ls[ls["attack_set"].isin(["orig", "main"]) | (ls["origin"] == "orig")]
    ls["kind"] = np.where(ls["origin"] == "orig", "real", ls["velocity_mode"])
    ls["detector"] = ls["name"].map({"ae_seen_s42": "lstm_ae",
                                     "clf_seen_s42": "lstm_supervised"})  # fmt: skip
    parts.append(ls[["detector", "kind", "delay_s"]])
    return pd.concat(parts, ignore_index=True)


def r10_delays():
    d = delays()
    kinds = [("real", "Dataset's own attacks"), ("position_only", "Synthetic, position only"),
             ("consistent", "Synthetic, velocity-consistent")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (k, title) in zip(axes, kinds):
        data, labels = [], []
        for det in S.DETECTORS:
            v = d.loc[(d["detector"] == det) & (d["kind"] == k), "delay_s"].dropna()
            data.append(np.maximum(v.to_numpy(), 0.02))
            labels.append(det)
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                        showfliers=True, flierprops={"markersize": 3})  # fmt: skip
        for patch, det in zip(bp["boxes"], labels):
            patch.set_facecolor(S.colour(det))
            patch.set_alpha(0.55)
            patch.set_edgecolor(S.colour(det))
        for med in bp["medians"]:
            med.set_color(S.INK)
        for j, v in enumerate(data):
            ax.text(200, j + 1, f"n={len(v)}", va="center", fontsize=7, color=S.INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 600)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Delay from attack onset to first alarm (s, log)")
    axes[0].set_yticks(
        range(1, len(S.DETECTORS) + 1), [S.label(x) for x in S.DETECTORS]
    )
    for tick, det in zip(axes[0].get_yticklabels(), S.DETECTORS):
        tick.set_color(S.colour(det))
    axes[0].invert_yaxis()
    fig.suptitle("Detection delay per detector (caught attacked test flights)")
    fig.tight_layout()
    S.note(fig, "Seen missions; learned models seed 42; n = flights caught. Boxes: "
                "median and quartiles; whiskers 1.5 x IQR.")  # fmt: skip
    return S.save(fig, "R10_detection_delay",
                  "How long each detector takes to alarm after the attack starts, for "
                  "real, position-only and velocity-consistent attacks.",
                  [str(PHYS.relative_to(S.ROOT)), str(RF.relative_to(S.ROOT)) + ".gz",
                   str(LSTM.relative_to(S.ROOT)) + ".gz"], ME)  # fmt: skip


def make():
    r10_threat()
    r10_false_alarms()
    r10_delays()
