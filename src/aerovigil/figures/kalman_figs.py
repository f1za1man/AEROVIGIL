"""Step K, plan M6: the Kalman filter, its chi-square (NIS) test and its CUSUM."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aerovigil.figures import style as S
from aerovigil.kalman.detectors import W_COLS, cusum_alarms, nis_alarms

ME = "src/aerovigil/figures/kalman_figs.py"
M06 = S.ROOT / "data" / "processed" / "module06"
INNOV = M06 / "innovations"
CAL = M06 / "detector_calibration.json"
SPLITS = S.ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
MAN = S.ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"
REFRACTORY_S = 10.0


def cal():
    return json.loads(CAL.read_text())


def test_ids(kind):
    s = pd.read_csv(SPLITS)
    ids = s.loc[(s["split"] == "test") & s["flight_id"].str.contains(kind), "flight_id"]
    return sorted(ids)


def examples():
    """(title, path) for a normal, a real attacked and a synthetic drift flight."""
    normal = next(f for f in test_ids("Normal") if f.startswith("Curved"))
    real = next(f for f in test_ids("Attacked") if f.startswith("Curved"))
    man = pd.read_csv(MAN)
    syn = man[(man["split"] == "test") & (man["attack_set"] == "main")
              & (man["attack_type"] == "linear_drift")
              & (man["velocity_mode"] == "position_only")
              & (man["mission_type"] == "curved")].sort_values("attacked_flight_id")  # fmt: skip
    syn_id = syn["attacked_flight_id"].iloc[0]
    return [("Normal flight", INNOV / "orig" / f"{normal}.parquet"),
            ("Real attack (dataset)", INNOV / "orig" / f"{real}.parquet"),
            ("Synthetic linear drift, position only",
             INNOV / "syn" / f"{syn_id}.parquet")]  # fmt: skip


def attack_span(df):
    on = df["label"].to_numpy() == 1
    if not on.any():
        return None
    t = df["t_s"].to_numpy()
    return float(t[on].min()), float(t[on].max())


def cusum_trace(z, valid, t, k, h):
    """The two-sided CUSUM statistic (max over components), reset as the detector is."""
    sp, sn = np.zeros(z.shape[1]), np.zeros(z.shape[1])
    out, skip = np.full(len(z), np.nan), -np.inf
    for i in range(len(z)):
        if not valid[i] or t[i] < skip:
            sp[:], sn[:] = 0.0, 0.0
            continue
        sp = np.maximum(0.0, sp + z[i] - k)
        sn = np.maximum(0.0, sn - z[i] - k)
        out[i] = max(sp.max(), sn.max())
        if out[i] > h:
            sp[:], sn[:] = 0.0, 0.0
            skip = t[i] + REFRACTORY_S
    return out


def shade(ax, span):
    if span:
        ax.axvspan(*span, color="#f9c0c0", linewidth=0, zorder=0, label="attack on")


def m6_innovation_nis():
    c = cal()
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex="col")
    used = []
    for j, (title, path) in enumerate(examples()):
        used.append(str(path.relative_to(S.ROOT)))
        df = pd.read_parquet(path)
        t, v = df["t_s"].to_numpy(), df["valid"].to_numpy()
        span = attack_span(df)
        pos = np.hypot(df["nu_pos_e"], df["nu_pos_n"])
        vel = np.hypot(df["nu_vel_e"], df["nu_vel_n"])
        a = axes[0, j]
        shade(a, span)
        a.plot(t, pos, color="#257de0", linewidth=1.3, label="position innovation (m)")
        a.plot(t, vel, color="#f47230", linewidth=1.3, linestyle="--",
               label="velocity innovation (m/s)")  # fmt: skip
        a.set_title(title)
        a.set_yscale("log")
        b = axes[1, j]
        shade(b, span)
        nis = df["nis_h"].to_numpy(float)
        b.plot(t, nis, color=S.colour("kalman"), linewidth=1.2, label="NIS")
        b.axhline(c["nis_threshold"], color=S.INK, linestyle=":", linewidth=1.4,
                  label=f"threshold {c['nis_threshold']:.2f}")  # fmt: skip
        al = nis_alarms(nis, v, t, c["nis_threshold"], REFRACTORY_S)
        b.scatter(t[al], nis[al], marker="v", s=60, color="#ec4647", zorder=3,
                  label="alarm")  # fmt: skip
        b.set_yscale("log")
        b.set_xlabel("Time since start of log (s)")
        b.text(0.02, 0.95, f"{len(al)} alarms", transform=b.transAxes, va="top",
               fontsize=9, color="#ec4647")  # fmt: skip
    axes[0, 0].set_ylabel("Innovation size (log)")
    axes[1, 0].set_ylabel("NIS, horizontal (log)")
    handles = (
        axes[0, 0].get_lines()[:2]
        + axes[1, 0].get_lines()[:2]
        + [axes[1, 1].collections[-1], Patch(color="#f9c0c0", label="attack on")]
    )
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Kalman filter innovations and the chi-square (NIS) test")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    S.note(fig, "Innovation = GPS minus the filter's IMU-based prediction (east/north "
                "combined). NIS = squared innovation weighted by its predicted "
                "covariance (4 degrees of freedom); threshold set for 1 false alarm per "
                "flight-hour on training and validation flights. Test flights.")  # fmt: skip
    return S.save(fig, "M6_innovation_nis",
                  "Filter innovations and the NIS test over time for a normal, a real "
                  "attacked and a synthetic drift flight, with alarms marked.",
                  used + [str(CAL.relative_to(S.ROOT))], ME)  # fmt: skip


def m6_whitened():
    c = cal()
    mu, sd = np.array(c["mu"]), np.array(c["sd"])
    parts = []
    for fid in test_ids("Normal"):
        df = pd.read_parquet(INNOV / "orig" / f"{fid}.parquet")
        ok = df["valid"].to_numpy()
        parts.append(((df[W_COLS].to_numpy(float) - mu) / sd)[ok])
    z = np.vstack(parts)
    names = {"w_pos_e": "position, east", "w_vel_e": "velocity, east",
             "w_pos_n": "position, north", "w_vel_n": "velocity, north"}  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    x = np.linspace(-6, 6, 400)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    bins = np.linspace(-6, 6, 61)
    for k, (ax, col) in enumerate(zip(axes, W_COLS)):
        ax.hist(np.clip(z[:, k], -6, 6), bins=bins, density=True, color="#9dc8f7",
                edgecolor=S.SURFACE, label="normal test flights")  # fmt: skip
        ax.plot(x, pdf, color=S.INK, linewidth=1.8, label="standard normal")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("Whitened innovation (standard units)")
        ax.set_title(names[col], fontsize=10)
        ax.text(0.03, 0.95, f"sd {np.std(z[:, k]):.2f}", transform=ax.transAxes,
                fontsize=8, va="top")  # fmt: skip
    axes[0].set_ylabel("Density (log)")
    axes[0].legend(loc="lower center", fontsize=7)
    fig.suptitle(f"Whitened innovations against a standard normal "
                 f"({len(z):,} filter ticks, 21 normal test flights)")  # fmt: skip
    fig.tight_layout()
    S.note(fig, "Each component is centred and scaled with the median / MAD from "
                "training and validation flights. Velocity innovations match a standard "
                "normal; the east position innovation is wider (a slow bias builds up "
                "over long flights), which the budget-based thresholds absorb.")  # fmt: skip
    return S.save(fig, "M6_whitened_innovations",
                  "The filter's whitened innovations on normal test flights against a "
                  "standard normal: velocity fits, east position is wider.",
                  ["data/processed/module06/innovations/orig/*Normal*.parquet",
                   str(CAL.relative_to(S.ROOT))], ME)  # fmt: skip


def m6_cusum():
    c = cal()
    mu, sd = np.array(c["mu"]), np.array(c["sd"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    used = []
    for ax, (title, path) in zip(axes, examples()):
        used.append(str(path.relative_to(S.ROOT)))
        df = pd.read_parquet(path)
        t, v = df["t_s"].to_numpy(), df["valid"].to_numpy()
        z = (df[W_COLS].to_numpy(float) - mu) / sd
        tr = cusum_trace(z, v, t, c["k"], c["h"])
        shade(ax, attack_span(df))
        ax.plot(t, np.maximum(tr, 1e-2), color=S.colour("cusum"), linewidth=1.4,
                label="CUSUM statistic")  # fmt: skip
        ax.axhline(c["h"], color=S.INK, linestyle=":", linewidth=1.4,
                   label=f"threshold h = {c['h']:.0f}")  # fmt: skip
        al = cusum_alarms(z, v, t, c["k"], c["h"], REFRACTORY_S)
        ax.scatter(t[al], np.full(len(al), c["h"]), marker="v", s=60,
                   color="#ec4647", zorder=3, label="alarm")  # fmt: skip
        ax.text(0.02, 0.95, f"{len(al)} alarms", transform=ax.transAxes, va="top",
                fontsize=9, color="#ec4647")  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(1e-1, c["h"] * 5)
        ax.set_title(title)
        ax.set_xlabel("Time since start of log (s)")
    axes[0].set_ylabel("CUSUM statistic (log)")
    handles = axes[0].get_lines()[:2] + [
        axes[1].collections[-1],
        Patch(color="#f9c0c0", label="attack on"),
    ]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(f"Kalman CUSUM over the whitened innovations (k = {c['k']:g})")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    S.note(fig, "Two-sided CUSUM per innovation component; the largest of the eight "
                "sums is shown. It resets after an alarm and over data gaps. On long "
                "flights a slow bias in the position innovation makes it climb even "
                "without an attack, which is why its threshold is so high.")  # fmt: skip
    return S.save(fig, "M6_cusum_trace",
                  "The CUSUM statistic over time for the same three flights, with its "
                  "threshold and alarms.", used + [str(CAL.relative_to(S.ROOT))], ME)  # fmt: skip


def m6_absorption():
    man = pd.read_csv(MAN)
    sw = man[man["attack_set"] == "sweep"]
    src = sorted(sw["source_flight_id"].unique())[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    used = []
    for ax, rate in zip(axes, (0.25, 2.0)):
        r = sw[(sw["source_flight_id"] == src) & np.isclose(sw["drift_rate_mps"], rate)]
        path = INNOV / "sweep" / f"{r['attacked_flight_id'].iloc[0]}.parquet"
        used.append(str(path.relative_to(S.ROOT)))
        d = pd.read_parquet(path)
        true_e, true_n = d["gps_e"] - d["offset_e"], d["gps_n"] - d["offset_n"]
        err = np.hypot(d["est_e"] - true_e, d["est_n"] - true_n)
        shade(ax, attack_span(d))
        ax.plot(d["t_s"], d["offset_m"], color="#ec4647", linewidth=2, linestyle="--",
                label="GPS pushed away from the truth")  # fmt: skip
        ax.plot(d["t_s"], err, color=S.colour("kalman"), linewidth=1.6,
                label="Filter position error")  # fmt: skip
        ax.set_title(f"Velocity-consistent drift at {rate:g} m/s", fontsize=10)
        ax.set_xlabel("Time since start of log (s)")
        ax.set_ylabel("Metres")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("The filter follows a smooth drift: its own estimate is dragged along")
    fig.tight_layout()
    S.note(fig, f"Source flight {src} (test), drift-sweep copies. The filter position "
                f"error tracks the GPS offset, so the innovation stays small and the "
                f"chi-square test sees little once the drift is under way. Fast drifts are "
                f"mostly caught early, while the drift builds up; slow ones are not.")  # fmt: skip
    return S.save(fig, "M6_drift_absorption",
                  "The Kalman filter absorbs a smooth velocity-consistent drift into its "
                  "own position estimate.", used, ME)  # fmt: skip


def make():
    m6_innovation_nis()
    m6_whitened()
    m6_cusum()
    m6_absorption()
