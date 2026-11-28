"""Step K, plan M4: the synthetic attacks (families, velocity modes, drift sweep, onset)."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aerovigil.attacks.inject_v2 import KEEP, inject
from aerovigil.figures import style as S

ME = "src/aerovigil/figures/attack_figs.py"
MAN = S.ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"
ATT = S.ROOT / "data" / "processed" / "module04_v2" / "base"
BASE = S.ROOT / "data" / "processed" / "module03_v2" / "base"
SHORT = {"position_only": "po", "consistent": "cv"}


def manifest():
    return pd.read_csv(MAN)


def source_flight(man):
    main = man[man["attack_set"] == "main"]
    full = main.groupby("source_flight_id").size()
    sweep = set(man.loc[man["attack_set"] == "sweep", "source_flight_id"])
    return sorted(f for f in full.index if full[f] == full.max() and f in sweep)[0]


def velocity_error(att, src):
    m = src.set_index("t_s")[["velocity_north", "velocity_east"]]
    a = att.set_index("t_s")[["velocity_north", "velocity_east"]]
    m = m.reindex(a.index)
    return np.hypot(a["velocity_north"] - m["velocity_north"],
                    a["velocity_east"] - m["velocity_east"]).to_numpy()  # fmt: skip


def m4_families():
    man = manifest()
    fid = source_flight(man)
    src = pd.read_parquet(BASE / f"{fid}.parquet", columns=["t_s", "velocity_north",
                                                            "velocity_east"])  # fmt: skip
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex="col")
    used = []
    for k, (typ, name) in enumerate(S.FAMILIES.items()):
        for mode in ("position_only", "consistent"):
            att_id = f"{fid}__syn_{typ}_00_{SHORT[mode]}"
            row = man[man["attacked_flight_id"] == att_id]
            if row.empty:
                continue
            path = ATT / row["file"].iloc[0]
            used.append(path)
            a = pd.read_parquet(path, columns=["t_s", "offset_m", "velocity_north",
                                               "velocity_east"])  # fmt: skip
            t0 = float(row["start_s"].iloc[0])
            tt = a["t_s"].to_numpy() - t0
            col, ls = S.MODE_STYLE[mode]
            axes[0, k].plot(tt, a["offset_m"], color=col, linestyle=ls, linewidth=1.8,
                            label=S.MODES[mode])  # fmt: skip
            axes[1, k].plot(tt, velocity_error(a, src), color=col, linestyle=ls,
                            linewidth=1.8, label=S.MODES[mode])  # fmt: skip
        axes[0, k].set_title(name)
        axes[1, k].set_xlabel("Time since attack start (s)")
    axes[0, 0].set_ylabel("Position offset (m)")
    axes[1, 0].set_ylabel("GPS velocity changed by (m/s)")
    axes[0, 0].legend(loc="center right", fontsize=8)
    fig.suptitle("The five attack families in both velocity modes (one source flight)")
    fig.tight_layout()
    S.note(fig, f"Source flight {fid}, draw 0 of each family. Position-only attacks "
                f"move the GPS position but leave the GPS velocity alone; "
                f"velocity-consistent attacks also change the velocity to match.")  # fmt: skip
    return S.save(fig, "M4_attack_families",
                  "Offset over time for each attack family, and how much each velocity "
                  "mode changes the reported GPS velocity.",
                  "data/interim/module04_v2/attack_manifest_v2.csv + "
                  "data/processed/module04_v2/base/*.parquet", ME)  # fmt: skip


def m4_sweep():
    man = manifest()
    fid = source_flight(man)
    sw = man[(man["attack_set"] == "sweep") & (man["source_flight_id"] == fid)]
    sw = sw.sort_values("drift_rate_mps")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ramp = plt.get_cmap("Blues")
    rates = sw["drift_rate_mps"].to_numpy()
    for i, (_, r) in enumerate(sw.iterrows()):
        a = pd.read_parquet(ATT / r["file"], columns=["t_s", "offset_m"])
        tt = a["t_s"] - r["start_s"]
        col = ramp(0.35 + 0.65 * i / max(1, len(rates) - 1))
        ax.plot(tt, a["offset_m"], color=col, linewidth=2)
        ax.annotate(f"{r['drift_rate_mps']:g} m/s", (tt.iloc[-1], a["offset_m"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=S.INK)  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylim(0.05, None)
    ax.set_xlim(left=-5)
    ax.set_xlabel("Time since attack start (s)")
    ax.set_ylabel("Position offset (m, log scale)")
    ax.set_title("Drift sweep: one velocity-consistent drift per speed")
    S.note(fig, f"Source flight {fid}. The sweep is made on test flights only "
                f"({sw['drift_rate_mps'].nunique()} speeds x 21 flights).")  # fmt: skip
    return S.save(fig, "M4_drift_sweep",
                  "The drift-sweep attacks: the same flight drifted at 0.05 to 5 m/s.",
                  "data/interim/module04_v2/attack_manifest_v2.csv + "
                  "data/processed/module04_v2/base/*sweep*.parquet", ME)  # fmt: skip


def m4_onset():
    man = manifest()
    fid = source_flight(man)
    r = man[(man["attack_set"] == "sweep") & (man["source_flight_id"] == fid)
            & np.isclose(man["drift_rate_mps"], 0.5)].iloc[0]  # fmt: skip
    g = pd.read_parquet(BASE / f"{fid}.parquet", columns=KEEP)
    spec = {"attack_type": "linear_drift", "start_s": float(r["start_s"]),
            "duration_s": float(r["duration_s"]),
            "params": json.loads(r["params_json"])}  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {"scale": ("Old onset (scale): velocity overshoots", "#ec4647", "--"),
              "warp": ("New onset (warp): no overshoot", "#257de0", "-")}  # fmt: skip
    peaks = {}
    for onset, (lab, col, ls) in styles.items():
        out, _ = inject(g, spec, "consistent", "x", onset=onset)
        tt = out["t_s"].to_numpy() - spec["start_s"]
        m = (tt >= -2) & (tt <= 40)
        v = out["offset_rate_mps"].to_numpy()
        peaks[onset] = float(np.nanmax(v[m]))
        ax.plot(tt[m], v[m], color=col, linestyle=ls, linewidth=2.2, label=lab)
    rate = spec["params"]["drift_rate_mps"]
    ax.axhline(rate, color=S.INK2, linewidth=1, linestyle=":")
    ax.annotate(f"target drift {rate:g} m/s", (38, rate), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=S.INK2)  # fmt: skip
    over = peaks["scale"] / rate - 1
    ax.text(0.98, 0.95, f"old onset peaks at {peaks['scale']:.2f} m/s (+{over:.0%})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#ec4647")  # fmt: skip
    ax.set_xlabel("Time since attack start (s)")
    ax.set_ylabel("Velocity added to the GPS (m/s)")
    ax.set_title("Velocity-consistent onset, old vs new")
    ax.legend(loc="lower right")
    S.note(fig, "Recomputed with the Module 04 injector on one sweep flight (0.5 m/s "
                "drift). The old overshoot is what made slow drifts look detectable.")  # fmt: skip
    return S.save(fig, "M4_onset_old_vs_new",
                  f"The old velocity-consistent onset overshot the drift speed by "
                  f"{over:.0%}; the new one eases in without overshoot.",
                  "data/interim/module04_v2/attack_manifest_v2.csv + "
                  "src/aerovigil/attacks/inject_v2.py", ME)  # fmt: skip


def m4_coverage():
    man = manifest()
    main = man[
        (man["attack_set"] == "main") & (man["velocity_mode"] == "position_only")
    ]
    fams = list(S.FAMILIES)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    data = [main.loc[main["attack_type"] == f, "peak_offset_m"] for f in fams]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.6)
    for b in bp["boxes"]:
        b.set_facecolor("#9dc8f7")
    for m in bp["medians"]:
        m.set_color(S.INK)
    axes[0].set_xticks(range(1, len(fams) + 1), [S.FAMILIES[f].replace(" ", "\n")
                                                  for f in fams], fontsize=8)  # fmt: skip
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Peak spoofed offset (m, log)")
    axes[0].set_title("Peak offset per family", fontsize=10)
    axes[1].hist(main["duration_s"], bins=25, color="#257de0", edgecolor=S.SURFACE)
    axes[1].set_xlabel("Attack duration (s)")
    axes[1].set_ylabel("Attacks")
    axes[1].set_title("Attack duration", fontsize=10)
    axes[2].hist(main["start_s"] / main["flight_duration_s"], bins=20, color="#257de0",
                 edgecolor=S.SURFACE)  # fmt: skip
    axes[2].set_xlabel("Attack start (fraction of the flight)")
    axes[2].set_title("Attack start", fontsize=10)
    fig.suptitle(f"Coverage of the synthetic attack set ({len(main)} attacks per velocity "
                 f"mode, all splits)")  # fmt: skip
    fig.tight_layout()
    S.note(fig, "Main set, position-only copies (the velocity-consistent copies are "
                "nearly the same; their eased onset trims a little). 41 position-only and "
                "45 velocity-consistent attacks never move the GPS more than 2 m but are "
                "still labelled attacks.")  # fmt: skip
    return S.save(fig, "M4_parameter_coverage",
                  "How large, long and where in the flight the synthetic attacks are.",
                  str(MAN.relative_to(S.ROOT)), ME)  # fmt: skip


def make():
    m4_families()
    m4_sweep()
    m4_onset()
    m4_coverage()
