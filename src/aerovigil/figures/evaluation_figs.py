"""Step K, plan M9: the evaluation framework (seen vs unseen missions vs unseen attack
types, seeds with intervals, ablations and model size)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aerovigil.figures import style as S

ME = "src/aerovigil/figures/evaluation_figs.py"
TABLES = S.ROOT / "reports" / "tables"
ALL = TABLES / "C_all_detectors.csv"
ABL = TABLES / "C_ablations.csv"
CLSTM = TABLES / "C_lstm.csv"
SEED_CL = S.ROOT / "data" / "processed" / "module05_v2" / "all_models_per_seed_full.csv"
SEED_LS = S.ROOT / "data" / "processed" / "module07_v2" / "protocols" / "per_seed.csv"
PROTOS = [("seen", "Seen missions", "#a3a3a3", "o"),
          ("unseen missions", "Unseen mission", "#5a5a5a", "s"),
          ("unseen attack types", "Unseen attack type", S.INK, "^")]  # fmt: skip


def colour_ticks(ax, names, axis="y"):
    ticks = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for t, n in zip(ticks, names):
        t.set_color(S.colour(n))


def m9_protocols():
    t = pd.read_csv(ALL)
    dets = S.DETECTORS
    metrics = [("detected", "All attacked test flights caught"),
               ("consistent_detected", "Velocity-consistent attacks caught")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    y = np.arange(len(dets))
    for ax, (m, title) in zip(axes, metrics):
        for i, (p, lab, col, mk) in enumerate(PROTOS):
            g = t[t["protocol"] == p].set_index("detector").reindex(dets)
            ax.scatter(g[m], y + (i - 1) * 0.22, color=col, marker=mk, s=45,
                       label=lab, zorder=3)  # fmt: skip
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel("Share of attacked test flights caught")
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(y, [S.label(d) for d in dets])
    colour_ticks(axes[0], dets)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Does it still work on a mission or attack type it never saw?")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    S.note(fig, "Budget 1 false alarm per flight-hour, recalibrated per protocol. Unseen "
                "mission: leave one of the three missions out of training and test on "
                "it (pooled). Unseen attack type: leave one attack family out. Learned "
                "models: mean of 5 seeds. False alarms per protocol: R10_false_alarms. "
                "Missing points: not run for that detector (the autoencoder never sees "
                "attacks, so it has no unseen-attack-type run).")  # fmt: skip
    return S.save(fig, "M9_protocols",
                  "Detection per detector on seen missions, an unseen mission and an "
                  "unseen attack type.", str(ALL.relative_to(S.ROOT)), ME)  # fmt: skip


def seeds():
    cl = pd.read_csv(SEED_CL)
    cl = cl[cl["model"] == "random_forest"].rename(columns={"model": "detector"})
    ls = pd.read_csv(SEED_LS)
    cols = ["protocol", "detector", "seed", "consistent_detected", "fa_per_hour", "f1"]
    return pd.concat([cl[cols], ls[cols]], ignore_index=True)


def m9_seeds():
    d = seeds()
    dets = ["random_forest", "lstm_ae", "lstm_supervised"]
    panels = [("seen", "consistent_detected", "Velocity-consistent caught, seen", False),
              ("unseen missions", "consistent_detected",
               "Velocity-consistent caught, unseen mission", False),
              ("seen", "fa_per_hour", "False alarms / h, seen", True),
              ("unseen missions", "fa_per_hour", "False alarms / h, unseen mission",
               True)]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    rng = np.random.default_rng(1)
    for ax, (p, m, title, logy) in zip(axes, panels):
        for i, det in enumerate(dets):
            v = d.loc[(d["protocol"] == p) & (d["detector"] == det), m].dropna()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), np.maximum(v, 0.05) if logy
                       else v, color=S.colour(det), marker=S.MODELS[det][2], s=36,
                       alpha=0.8, zorder=3)  # fmt: skip
            if len(v):
                mu, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
                c = max(mu, 0.05) if logy else mu
                lo = c - max(mu - sd, 0.05) if logy else min(sd, mu)
                ax.errorbar(i + 0.3, c, yerr=[[max(lo, 0)], [sd]], color=S.INK,
                            capsize=4, marker="_", markersize=14)  # fmt: skip
        ax.set_xticks(range(len(dets)), [S.label(x) for x in dets], fontsize=8,
                      rotation=15)  # fmt: skip
        colour_ticks(ax, dets, "x")
        ax.set_title(title, fontsize=10)
        if logy:
            ax.set_yscale("log")
            ax.axhline(1.0, color="#ec4647", linestyle="--", linewidth=1.2)
        else:
            ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Share of flights / alarms per hour")
    fig.suptitle("Spread over the 5 training seeds (42-46)")
    fig.tight_layout()
    S.note(fig, "Points: one seed each; black bar: mean +/- 1 standard deviation. The "
                "dashed red line is the 1 false alarm per hour budget. Zero false alarms "
                "are drawn at 0.05. Physics detectors have no training seed.")  # fmt: skip
    return S.save(fig, "M9_seeds",
                  "How much the learned detectors' results move between training seeds.",
                  [str(SEED_CL.relative_to(S.ROOT)), str(SEED_LS.relative_to(S.ROOT))],
                  ME)  # fmt: skip


CONFIG = {"full": "All features,\n5 s windows", "no_baro": "No\nbarometer",
          "no_mag (none in use)": "No\nmagnetometer*", "window_2s": "2 s\nwindows",
          "window_10s": "10 s\nwindows"}  # fmt: skip


def m9_ablations():
    a = pd.read_csv(ABL)
    models = ["random_forest", "logistic_regression", "naive_bayes"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True)
    x = np.arange(len(CONFIG))
    for r, (p, ptitle) in enumerate((("seen", "Seen missions"),
                                     ("unseen missions", "Unseen mission"))):  # fmt: skip
        for c, (m, mtitle, logy) in enumerate((("consistent_detected",
                                                "Velocity-consistent caught", False),
                                               ("fa_per_hour", "False alarms / h", True))):  # fmt: skip
            ax = axes[r, c]
            for i, mod in enumerate(models):
                g = a[(a["protocol"] == p) & (a["model"] == mod)].set_index("config")
                v = g.reindex(list(CONFIG))[m].to_numpy(float)
                err = None
                if m == "fa_per_hour":
                    err = g.reindex(list(CONFIG))["fa_per_hour_sd"].fillna(0).to_numpy()
                    v = np.maximum(v, 0.05)
                ax.bar(x + (i - 1) * 0.27, v, 0.25, color=S.colour(mod), yerr=err,
                       label=S.label(mod), error_kw={"ecolor": S.INK2, "capsize": 2},
                       hatch=[None, "//", ".."][i], edgecolor=S.SURFACE)  # fmt: skip
            if logy:
                ax.set_yscale("log")
                ax.axhline(1.0, color="#ec4647", linestyle="--", linewidth=1.2)
            else:
                ax.set_ylim(0, 1.05)
            ax.set_title(f"{mtitle}, {ptitle.lower()}", fontsize=10)
    for ax in axes[1]:
        ax.set_xticks(x, list(CONFIG.values()), fontsize=9)
    axes[0, 0].set_ylabel("Share of flights")
    axes[1, 0].set_ylabel("Share of flights")
    axes[0, 1].set_ylabel("Alarms per flight-hour (log)")
    axes[1, 1].set_ylabel("Alarms per flight-hour (log)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ablations: sensors and window length (classical models)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    S.note(fig, "* No feature uses the magnetometer, so removing it changes nothing. "
                "Random forest: mean of 5 seeds (error bar: seed sd); the others are "
                "deterministic. 2 s windows bring the unseen-mission false alarms of the "
                "random forest from 22 to 0.8 per hour. Zero false alarms drawn at "
                "0.05; red line = budget.")  # fmt: skip
    return S.save(fig, "M9_ablations",
                  "What happens without the barometer or magnetometer and with 2 s or "
                  "10 s windows, on seen and unseen missions.",
                  str(ABL.relative_to(S.ROOT)), ME)  # fmt: skip


def m9_model_size():
    c = pd.read_csv(CLSTM)
    sizes = [("seen", "32 x 1 (used)"), ("size h32 l2", "32 x 2"),
             ("size h64 l1", "64 x 1"), ("size h64 l2", "64 x 2")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(sizes))
    for i, det in enumerate(("lstm_ae", "lstm_supervised")):
        g = c[c["detector"] == det].set_index("protocol").reindex([s for s, _ in sizes])
        off = (i - 0.5) * 0.36
        axes[0].bar(x + off, g["auc_syn_consistent_mean"], 0.34,
                    yerr=g["auc_syn_consistent_std"], color=S.colour(det),
                    label=S.label(det), error_kw={"capsize": 2})  # fmt: skip
        fa, sd = g["fa_per_hour_mean"].to_numpy(), g["fa_per_hour_std"].to_numpy()
        axes[1].bar(x + off, fa, 0.34, yerr=[np.minimum(sd, fa), sd],
                    color=S.colour(det), label=S.label(det), error_kw={"capsize": 2})  # fmt: skip
    axes[0].set_ylim(0.5, 1)
    axes[0].set_ylabel("Window AUC, velocity-consistent attacks")
    axes[1].set_ylabel("False alarms per flight-hour")
    axes[1].axhline(1.0, color="#ec4647", linestyle="--", linewidth=1.2)
    for ax in axes:
        ax.set_xticks(x, [s[1] for s in sizes])
        ax.set_xlabel("LSTM size (hidden units x layers)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Ablation: LSTM model size (seen missions, 5 seeds)")
    fig.tight_layout()
    S.note(fig, "Bars: mean of 5 seeds; error bars: seed sd. Bigger models do not help: "
                "the differences are within the seed spread.")  # fmt: skip
    return S.save(fig, "M9_model_size",
                  "Larger LSTMs do not detect velocity-consistent attacks better.",
                  str(CLSTM.relative_to(S.ROOT)), ME)  # fmt: skip


def m9_by_attack():
    from aerovigil.evaluation import store

    ph = pd.read_csv(
        S.ROOT / "data" / "processed" / "evaluation" / "physics_flights.csv"
    )
    ph = ph[
        (ph["protocol"] == "seen") & ph["attacked"] & ph["set"].isin(["orig", "main"])
    ]
    ph["kind"] = np.where(
        ph["set"] == "orig", "real", ph["attack_type"] + "|" + ph["velocity_mode"]
    )
    parts = [ph.groupby(["detector", "kind"])["detected"].mean()]
    rf = store.read_csv(
        S.ROOT / "data" / "processed" / "module05_v2" / "all_models_flight_full.csv"
    )
    rf = rf[
        (rf["protocol"] == "seen") & (rf["model"] == "random_forest") & rf["attacked"]
    ]
    rf["kind"] = np.where(
        rf["origin"] == "orig", "real", rf["attack_type"] + "|" + rf["velocity_mode"]
    )
    parts.append(
        rf.assign(detector="random_forest")
        .groupby(["detector", "kind"])["detected"]
        .mean()
    )
    ls = store.read_csv(
        S.ROOT / "data" / "processed" / "module07_v2" / "protocols" / "flights.csv"
    )
    ls = ls[
        (ls["protocol"] == "seen")
        & ls["attacked"]
        & ((ls["origin"] == "orig") | (ls["attack_set"] == "main"))
    ]
    ls["detector"] = np.where(ls["kind"] == "ae", "lstm_ae", "lstm_supervised")
    ls["kind"] = np.where(
        ls["origin"] == "orig", "real", ls["attack_type"] + "|" + ls["velocity_mode"]
    )
    parts.append(ls.groupby(["detector", "kind"])["detected"].mean())
    t = pd.concat(parts).unstack()
    cols = ["real"] + [f"{f}|{m}" for f in S.FAMILIES for m in S.MODES]
    t = t.reindex(index=S.DETECTORS, columns=cols)
    fig, ax = plt.subplots(figsize=(14, 5.2))
    im = ax.imshow(t.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(t.to_numpy(float)):
        ax.text(
            j,
            i,
            f"{v:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if v > 0.6 else S.INK,
        )
    labels = ["Real attacks"] + [
        S.cell_label(f, m) for f in S.FAMILIES for m in S.MODES
    ]
    ax.set_xticks(range(len(cols)), labels, fontsize=7)
    ax.set_yticks(range(len(t)), [S.label(d) for d in t.index])
    colour_ticks(ax, list(t.index))
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.02, label="Attacked test flights caught")
    ax.set_title(
        "Detection by attack family and velocity mode, every detector "
        "(seen missions)"
    )
    S.note(
        fig,
        "Budget 1 false alarm per flight-hour; learned models: mean of 5 seeds. "
        "A flight counts as caught when the detector alarms while the attack is "
        "on.",
    )
    return S.save(
        fig,
        "M9_detection_by_attack",
        "Every detector against every attack family in both velocity modes.",
        [
            "data/processed/evaluation/physics_flights.csv",
            "data/processed/module05_v2/all_models_flight_full.csv.gz",
            "data/processed/module07_v2/protocols/flights.csv.gz",
        ],
        ME,
    )


def make():
    m9_protocols()
    m9_by_attack()
    m9_seeds()
    m9_ablations()
    m9_model_size()
