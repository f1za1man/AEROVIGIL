"""Step K, repo Module 09: the consistency detector (velocity and course channels), why
the course check fails on real flights, and the tilt-vs-speed check."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aerovigil.consistency.detector import CHANNELS, per_channel_scores
from aerovigil.consistency.evaluate import prepared
from aerovigil.figures import style as S

ME = "src/aerovigil/figures/consistency_figs.py"
M09 = S.ROOT / "data" / "processed" / "module09"
ALARMS = M09 / "alarms.csv"
CAL = M09 / "consistency_calibration.json"
BASE = S.ROOT / "data" / "processed" / "module03_v2" / "base"
SPLITS = S.ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
NAMES = {CHANNELS[0]: "Course channel (GPS course vs heading)",
         CHANNELS[1]: "Velocity channel (GPS position change vs GPS velocity)"}  # fmt: skip
PEAK = {CHANNELS[0]: "peak_course_heading", CHANNELS[1]: "peak_horizontal_velocity"}


def normal_test_flights():
    s = pd.read_csv(SPLITS)
    return sorted(s.loc[(s["split"] == "test") & s["flight_id"].str.contains("Normal"),
                        "flight_id"])  # fmt: skip


def r09_channels():
    cal = json.loads(CAL.read_text())
    mu, sd, thr = np.array(cal["mu"]), np.array(cal["sd"]), cal["threshold"]
    normal = {c: [] for c in CHANNELS}
    for fid in normal_test_flights():
        p = prepared(BASE / f"{fid}.parquet")
        for c, z in per_channel_scores(p["values"], mu, sd).items():
            normal[c].append(float(np.nanmax(z)) if np.isfinite(z).any() else np.nan)
    a = pd.read_csv(ALARMS)
    groups = [("Normal flights\n(whole flight)", None, "#257de0", "o"),
              ("Real attacks", (a["set"] == "orig") & a["attacked"], "#ec4647", "^"),
              ("Synthetic,\nposition only", (a["set"] == "main")
               & (a["velocity_mode"] == "position_only"), "#f2a600", "s"),
              ("Synthetic,\nvelocity-consistent", (a["set"] == "main")
               & (a["velocity_mode"] == "consistent"), "#6a4ce2", "D")]  # fmt: skip
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, c in zip(axes, CHANNELS):
        for i, (lab, mask, col, mk) in enumerate(groups):
            v = np.array(normal[c]) if mask is None else a.loc[mask, PEAK[c]].to_numpy()
            v = v[np.isfinite(v)]
            x = i + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, np.maximum(v, 1e-2), s=14, color=col, marker=mk, alpha=0.6,
                       edgecolors="none")  # fmt: skip
            caught = (v > thr).mean() if len(v) else np.nan
            ax.text(i, 2e5, f"{caught:.0%} above", ha="center", fontsize=8, color=S.INK)
        ax.axhline(thr, color=S.INK, linestyle=":", linewidth=1.4)
        ax.annotate(f"threshold {thr:.1f}", (3.45, thr), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8)  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 6e5)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        ax.set_title(NAMES[c], fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Peak standardised channel value per flight (log)")
    fig.suptitle(
        "Consistency detector: each channel on clean and attacked test flights"
    )
    fig.tight_layout()
    S.note(fig, "Peak of |value - median| / (1.4826 MAD) per flight: during the attack "
                "for attacked flights, over the whole flight for normal ones. The "
                "detector alarms when either channel passes the threshold (1 false "
                "alarm per flight-hour on calibration flights). The course channel "
                "never gets there.")  # fmt: skip
    return S.save(fig, "R09_consistency_channels",
                  "The velocity channel separates attacks from clean flight; the course "
                  "channel never reaches the threshold.",
                  [str(ALARMS.relative_to(S.ROOT)), str(CAL.relative_to(S.ROOT)),
                   "data/processed/module03_v2/base/*Normal*.parquet"], ME)  # fmt: skip


def attitude(g):
    x, y, z, w = (g[f"orientation_{k}"].to_numpy(float) for k in "xyzw")
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x**2 + y**2), -1, 1)))
    return yaw, tilt


def normal_rows():
    cols = ["velocity_north", "velocity_east", "orientation_x", "orientation_y",
            "orientation_z", "orientation_w", "mission_type", "aligned_all"]  # fmt: skip
    parts = []
    for fid in normal_test_flights():
        g = pd.read_parquet(BASE / f"{fid}.parquet", columns=cols)
        g = g[g["aligned_all"] == 1].iloc[::10]
        yaw, tilt = attitude(g)
        course = np.arctan2(g["velocity_north"], g["velocity_east"])
        diff = np.degrees(np.angle(np.exp(1j * (course - yaw))))
        parts.append(pd.DataFrame({"speed": np.hypot(g["velocity_north"],
                                                     g["velocity_east"]),
                                   "diff": diff, "tilt": tilt,
                                   "mission": g["mission_type"].to_numpy()}))  # fmt: skip
    return pd.concat(parts, ignore_index=True)


MISSION_C = {"curved": ("#257de0", "o"), "random": ("#f47230", "s"),
             "straight": ("#00b894", "^")}  # fmt: skip


def r09_course():
    d = normal_rows()
    d = d[d["speed"] > 2.0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                 gridspec_kw={"width_ratios": [1.4, 1]})  # fmt: skip
    for m, (col, mk) in MISSION_C.items():
        g = d[d["mission"] == m]
        a1.scatter(g["speed"], g["diff"], s=4, color=col, marker=mk, alpha=0.35,
                   edgecolors="none", label=m.capitalize())  # fmt: skip
        a2.hist(g["diff"], bins=np.linspace(-180, 180, 73), histtype="step",
                linewidth=2, color=col, density=True, label=m.capitalize())  # fmt: skip
    a1.set_xlabel("Horizontal speed (m/s)")
    a1.set_ylabel("GPS course minus heading (deg)")
    a1.set_ylim(-185, 185)
    a1.set_title("Course vs heading on normal flights (speed > 2 m/s)")
    a1.legend(loc="lower right", markerscale=4)
    a2.set_xlabel("GPS course minus heading (deg)")
    a2.set_ylabel("Density")
    a2.set_title("Distribution per mission")
    p90 = np.percentile(np.abs(d["diff"]), 90)
    a2.text(0.02, 0.95, f"90% of samples within +/-{p90:.0f} deg",
            transform=a2.transAxes, va="top", fontsize=8)  # fmt: skip
    fig.tight_layout()
    S.note(fig, "21 normal test flights, every 10th complete log row. A multirotor does "
                "not have to point where it flies: below about 5 m/s and in turns its "
                "course and heading differ by 20-100 deg with honest GPS. A slow drift "
                "turns the GPS course by only a few degrees, which hides in that spread, "
                "so the course channel never separates attacks (see "
                "R09_consistency_channels).")  # fmt: skip
    return S.save(fig, "R09_course_vs_heading",
                  "Why the course check fails: with honest GPS the heading already "
                  "differs from the GPS course by 20-100 deg at low speed and in turns.",
                  "data/processed/module03_v2/base/*Normal*.parquet (test split)", ME)  # fmt: skip


def r09_tilt():
    d = normal_rows()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, (col, mk) in MISSION_C.items():
        g = d[d["mission"] == m]
        ax.scatter(g["speed"], g["tilt"], s=4, color=col, marker=mk, alpha=0.3,
                   edgecolors="none", label=m.capitalize())  # fmt: skip
    r = np.corrcoef(d["speed"], d["tilt"])[0, 1]
    ax.set_xlabel("Horizontal speed from GPS (m/s)")
    ax.set_ylabel("Tilt from vertical (deg)")
    ax.set_ylim(0, np.percentile(d["tilt"], 99.8) * 1.1)
    ax.set_title(f"Tilt vs speed on normal flights (R^2 = {r**2:.2f})")
    ax.legend(loc="upper left", markerscale=4)
    S.note(fig, "21 normal test flights, every 10th complete log row. Tilt follows "
                "speed only loosely: at one speed it spans about 0-35 deg depending on "
                "whether the drone is speeding up or slowing down, so a drift that adds "
                "under 1 m/s to a consistent GPS velocity cannot be checked against tilt.")  # fmt: skip
    return S.save(fig, "R09_tilt_vs_speed",
                  "Tilt follows speed only loosely (it mostly follows acceleration), so it "
                  "cannot expose a smooth velocity-consistent drift.",
                  "data/processed/module03_v2/base/*Normal*.parquet (test split)", ME)  # fmt: skip


def make():
    r09_channels()
    r09_course()
    r09_tilt()
