"""Step K, plan M2 and M3: the dataset, its phases, features, windows and splits."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aerovigil.figures import style as S

ME = "src/aerovigil/figures/data_figs.py"
M03 = S.ROOT / "data" / "interim" / "module03"
V2 = S.ROOT / "data" / "processed" / "module03_v2"
COVERAGE = M03 / "raw_coverage.csv"
SPLITS = M03 / "flight_splits.csv"
MISSIONS = ["curved", "random", "straight"]
COND = {"normal": ("Normal", "#257de0"), "attacked": ("Attacked", "#f47230")}
SPLIT_C = {"train": "#257de0", "validation": "#f47230", "test": "#00b894"}


def coverage():
    return pd.read_csv(COVERAGE)


def m2_flights():
    r = coverage()
    n = r.groupby(["mission", "condition"]).size().unstack()[["normal", "attacked"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(MISSIONS))
    for i, c in enumerate(["normal", "attacked"]):
        v = n.loc[MISSIONS, c].to_numpy()
        b = ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=COND[c][1], label=COND[c][0])
        ax.bar_label(b, padding=2, fontsize=9)
    ax.set_xticks(x, [m.capitalize() for m in MISSIONS])
    ax.set_ylabel("Flights")
    ax.set_xlabel("Mission")
    ax.set_title(f"Flights per mission and label ({len(r)} flights)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, n.to_numpy().max() * 1.18)
    return S.save(fig, "M2_flights_per_mission",
                  "How many normal and attacked flights each of the three missions has.",
                  str(COVERAGE.relative_to(S.ROOT)), ME)  # fmt: skip


def m2_durations():
    r = coverage()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    bins = np.arange(0, 310, 10)
    for ax, m in zip(axes, MISSIONS):
        for c in ["normal", "attacked"]:
            v = r.loc[(r["mission"] == m) & (r["condition"] == c), "merged_duration_s"]
            ax.hist(v, bins=bins, histtype="step", linewidth=2, color=COND[c][1],
                    label=f"{COND[c][0]} (median {v.median():.0f} s)")  # fmt: skip
        ax.set_title(m.capitalize())
        ax.set_xlabel("Flight duration (s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Flights")
    fig.suptitle("Flight durations per mission", y=1.03)
    return S.save(fig, "M2_flight_durations",
                  "Length of every flight, per mission: straight flights last about "
                  "40 s, random about 110 s and curved about 230 s.",
                  str(COVERAGE.relative_to(S.ROOT)), ME)  # fmt: skip


def m2_rates():
    r = coverage()
    r["hz"] = (1.0 / r["step_merged_s"]).round().astype(int)
    t = r.groupby(["mission", "hz"]).size().unstack(fill_value=0).loc[MISSIONS]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(MISSIONS))
    cols = {125: "#257de0", 250: "#f47230"}
    for i, hz in enumerate(sorted(t.columns)):
        b = ax.bar(x + (i - 0.5) * 0.38, t[hz], 0.36, color=cols.get(hz, S.NEUTRAL),
                   label=f"{hz} Hz")  # fmt: skip
        ax.bar_label(b, padding=2, fontsize=9)
    ax.set_xticks(x, [m.capitalize() for m in MISSIONS])
    ax.set_xlabel("Mission")
    ax.set_ylabel("Flights")
    ax.set_ylim(0, t.to_numpy().max() * 1.18)
    rows_s = (r["rows_merged"] / r["merged_duration_s"]).mean()
    ax.set_title("Logging rate of the merged sensor table per flight")
    ax.legend(title="Most common time step", loc="upper left")
    S.note(fig, f"Rate = 1 / most common time step between merged rows. Mean over all "
                f"flights: {rows_s:.0f} rows per second, so time always comes from "
                f"timestamps, never from row counts.")  # fmt: skip
    return S.save(fig, "M2_logging_rate",
                  "Flights log at about 125 or 250 Hz; the rate differs between "
                  "flights of the same mission.",
                  str(COVERAGE.relative_to(S.ROOT)), ME)  # fmt: skip


def local_m(lat, lon, lat0, lon0):
    r = 6371000.0
    e = np.radians(lon - lon0) * r * np.cos(np.radians(lat0))
    n = np.radians(lat - lat0) * r
    return e, n


def example_flight_id():
    s = pd.read_csv(SPLITS)
    test = s.loc[(s["split"] == "test") & s["flight_id"].str.contains("Attacked"),
                 "flight_id"]  # fmt: skip
    return sorted(f for f in test if f.startswith("Curved"))[0]


def shade_phases(ax, t, phase):
    edges = np.flatnonzero(np.diff(phase)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(phase)]
    for a, b in zip(starts, ends):
        col = S.PHASES[int(phase[a])][1]
        if col:
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=col, linewidth=0, zorder=0)


def m3_example():
    fid = example_flight_id()
    path = V2 / "base" / f"{fid}.parquet"
    g = pd.read_parquet(path, columns=["t_s", "offset_m", "phase", "latitude_deg",
                                       "longitude_deg", "raw_latitude_deg",
                                       "raw_longitude_deg"])  # fmt: skip
    g = g.dropna(subset=["latitude_deg", "raw_latitude_deg"])
    t, ph = g["t_s"].to_numpy(), g["phase"].to_numpy()
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    shade_phases(a1, t, ph)
    a1.plot(t, g["offset_m"], color=S.INK, linewidth=1.6, label="GPS offset from truth")
    a1.set_xlabel("Time since start of log (s)")
    a1.set_ylabel("Offset of reported GPS from true position (m)")
    a1.set_title("Phases of one attacked flight")
    present = sorted(set(int(p) for p in ph))
    handles = [Patch(color=S.PHASES[p][1] or S.SURFACE, label=S.PHASES[p][0],
                     ec=S.GRID) for p in present]  # fmt: skip
    a1.legend(
        handles=handles + a1.get_lines(),
        loc="center right",
        fontsize=8,
        framealpha=0.9,
        frameon=True,
    )
    lat0, lon0 = g["raw_latitude_deg"].iloc[0], g["raw_longitude_deg"].iloc[0]
    e, n = local_m(g["raw_latitude_deg"], g["raw_longitude_deg"], lat0, lon0)
    es, ns = local_m(g["latitude_deg"], g["longitude_deg"], lat0, lon0)
    a2.plot(e, n, color="#257de0", linewidth=2, label="True path (raw GPS)")
    a2.plot(es, ns, color="#ec4647", linewidth=1.4, linestyle="--",
            label="Reported (spoofed) GPS")  # fmt: skip
    a2.set_aspect("equal", adjustable="datalim")
    a2.set_xlabel("East (m)")
    a2.set_ylabel("North (m)")
    a2.set_title("Path seen from above")
    a2.legend(loc="best", fontsize=8)
    S.note(fig, f"Flight {fid} (test split). Phases are measured from the offset, not "
                f"taken from the attack flag.")  # fmt: skip
    return S.save(fig, "M3_example_flight_phases",
                  "One real attacked flight: the GPS offset with its measured phases "
                  "(pre, ramp, plateau, post) and the true vs reported path.",
                  str(path.relative_to(S.ROOT)), ME)  # fmt: skip


KEY_FEATURES = {
    "res_horizontal_velocity": "GPS vs IMU horizontal velocity residual (m/s)",
    "res_course_heading_abs": "|sin(GPS course - heading)|, speed-weighted",
    "res_altitude_abs": "|GPS - barometer altitude| (m)",
    "res_vertical_velocity_abs": "|GPS vs barometer climb rate| (m/s)",
    "res_accel_horizontal": "GPS speed change vs accelerometer (m/s^2)",
    "gps_speed_from_position": "Speed from GPS positions (m/s)",
}


def test_flights():
    s = pd.read_csv(SPLITS)
    return s.loc[s["split"] == "test", "flight_id"].tolist()


def m3_features():
    cols = list(KEY_FEATURES) + ["phase"]
    parts = []
    for fid in test_flights():
        d = pd.read_parquet(V2 / "features" / f"{fid}.parquet", columns=cols)
        d["normal_flight"] = "Normal" in fid
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    normal = d[d["normal_flight"] & (d["phase"] == 0)]
    attack = d[~d["normal_flight"] & d["phase"].isin([2, 3])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (f, name) in zip(axes.ravel(), KEY_FEATURES.items()):
        a, b = normal[f].dropna(), attack[f].dropna()
        lo, hi = np.nanpercentile(np.r_[a, b], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins=bins, density=True, histtype="step", linewidth=2,
                color="#257de0", label="Normal flights")  # fmt: skip
        ax.hist(b, bins=bins, density=True, histtype="step", linewidth=2,
                color="#f47230", linestyle="--", label="Attack (ramp + plateau)")  # fmt: skip
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("Density")
        ax.set_yscale("log")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Key features, normal vs attacked samples (54 test flights)")
    fig.tight_layout()
    S.note(fig, "Each sample is one log row. Log scale on the y axis; the 0.5-99.5% "
                "range is shown. Real attacks in this dataset are position jumps, so "
                "they show mainly in the velocity and position-change features.")  # fmt: skip
    return S.save(fig, "M3_feature_distributions",
                  "How six key consistency features differ between normal flight and "
                  "the dataset's own attacks.",
                  "data/processed/module03_v2/features/*.parquet", ME)  # fmt: skip


def window_means(split):
    d = np.load(V2 / "windows" / f"windows_{split}.npz", allow_pickle=False)
    x = np.nanmean(d["X"], axis=1)
    names = np.array([str(n) for n in d["class_names"]])
    return x, names[d["window_class"]], [str(n) for n in d["feature_names"]]


def m3_pca():
    xtr, ctr, _ = window_means("train")
    xte, cte, _ = window_means("test")
    keep = np.isfinite(xtr).all(axis=1)
    xtr = xtr[keep]
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0) + 1e-9
    z = (xtr - mu) / sd
    _, s, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    share = s**2 / np.sum(s**2)
    ok = np.isfinite(xte).all(axis=1)
    p = ((xte[ok] - mu) / sd - z.mean(axis=0)) @ vt[:2].T
    c = cte[ok]
    fig, ax = plt.subplots(figsize=(6.6, 5))
    groups = [("normal", "Normal", "#257de0", "o", 6),
              ("plateau", "Attack plateau", "#ec4647", "^", 10),
              ("ramp", "Attack ramp", "#f2a600", "s", 10)]  # fmt: skip
    for key, lab, col, mk, sz in groups:
        m = c == key
        ax.scatter(p[m, 0], p[m, 1], s=sz, c=col, marker=mk, alpha=0.55,
                   edgecolors="none", label=f"{lab} ({m.sum()})")  # fmt: skip
    ax.set_xlabel(f"PC 1 ({share[0]:.0%} of variance)")
    ax.set_ylabel(f"PC 2 ({share[1]:.0%} of variance)")
    lo = np.percentile(p, [0.5, 99.5], axis=0)
    pad = 0.1 * (lo[1] - lo[0])
    ax.set_xlim(lo[0, 0] - pad[0], lo[1, 0] + pad[0])
    ax.set_ylim(lo[0, 1] - pad[1], lo[1, 1] + pad[1])
    ax.set_title("PCA of 5 s windows (test flights)")
    ax.legend(loc="best", markerscale=2)
    S.note(fig, "Each point is one 5 s window, averaged over time; PCA fitted on "
                "training windows only. Pre-attack, snapback and post-attack windows "
                "not shown.")  # fmt: skip
    return S.save(fig, "M3_pca_windows",
                  "Normal and attacked 5 s windows on the first two principal "
                  "components: ramp windows stand apart, plateau windows overlap "
                  "normal flight.",
                  "data/processed/module03_v2/windows/windows_{train,test}.npz", ME)  # fmt: skip


def m3_splits():
    s = pd.read_csv(SPLITS)
    s["condition"] = np.where(
        s["flight_id"].str.contains("Attacked"), "attacked", "normal"
    )
    s["mission"] = s["flight_id"].str.split("__").str[0].str.lower()
    order = ["train", "validation", "test"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    t = s.groupby(["split", "condition"]).size().unstack().loc[order]
    bottom = np.zeros(len(order))
    for c in ["normal", "attacked"]:
        b = a1.bar(order, t[c], bottom=bottom, color=COND[c][1], label=COND[c][0],
                   edgecolor=S.SURFACE, linewidth=2)  # fmt: skip
        a1.bar_label(b, label_type="center", color="white", fontsize=9)
        bottom += t[c].to_numpy()
    a1.set_ylabel("Flights")
    a1.set_title("Flights per split (seed 42)")
    a1.legend(loc="upper right")
    wins = {sp: pd.read_csv(V2 / "windows" / f"window_manifest_{sp}.csv")
            for sp in order}  # fmt: skip
    w = pd.DataFrame({sp: wins[sp]["window_class"].value_counts() for sp in order}).T
    cls = [
        c for c in ["normal", "pre", "ramp", "plateau", "snapback", "post"] if c in w
    ]
    cols = ["#257de0", "#9dc8f7", "#f2a600", "#ec4647", "#6a4ce2", S.NEUTRAL]
    bottom = np.zeros(len(order))
    for c, col in zip(cls, cols):
        v = w[c].fillna(0).to_numpy()
        a2.bar(order, v, bottom=bottom, color=col, label=c, edgecolor=S.SURFACE,
               linewidth=1)  # fmt: skip
        bottom += v
    for i, tot in enumerate(bottom):
        a2.text(i, tot, f"{int(tot):,}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("5 s windows")
    a2.set_title("Windows per split and class")
    a2.legend(loc="upper right", fontsize=8, ncol=2)
    a2.set_ylim(0, bottom.max() * 1.15)
    S.note(fig, "Splits are by flight, so no flight and no window crosses splits.")
    return S.save(fig, "M3_split_sizes",
                  "Train, validation and test sizes in flights and in 5 s windows.",
                  ["data/interim/module03/flight_splits.csv",
                   "data/processed/module03_v2/windows/window_manifest_*.csv"], ME)  # fmt: skip


SENSORS = {"GPS (reported)": "latitude_deg", "GPS (raw)": "raw_latitude_deg",
           "IMU": "linear_acceleration_x", "Barometer": "pressure_altitude",
           "Magnetometer": "magnetic_field_x", "Ground truth": "gt_latitude_deg"}  # fmt: skip


def m3_missing():
    rows = []
    for p in sorted((V2 / "base").glob("*.parquet")):
        g = pd.read_parquet(p, columns=list(SENSORS.values()) + ["aligned_all"])
        r = {"mission": p.name.split("__")[0].lower(), "rows": len(g),
             "Incomplete row": float((g["aligned_all"] != 1).mean())}  # fmt: skip
        r.update({k: float(g[c].isna().mean()) for k, c in SENSORS.items()})
        rows.append(r)
    d = pd.DataFrame(rows)
    cols = list(SENSORS) + ["Incomplete row"]
    w = d.groupby("mission").apply(lambda g: pd.Series(
        {c: np.average(g[c], weights=g["rows"]) for c in cols}), include_groups=False)  # fmt: skip
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(cols))
    mc = {"curved": ("#257de0", None), "random": ("#f47230", "//"),
          "straight": ("#00b894", "..")}  # fmt: skip
    for i, m in enumerate(MISSIONS):
        col, hatch = mc[m]
        b = ax.bar(x + (i - 1) * 0.27, 100 * w.loc[m, cols], 0.25, color=col, hatch=hatch,
                   edgecolor=S.SURFACE, label=m.capitalize())  # fmt: skip
        ax.bar_label(b, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x, cols)
    ax.set_ylabel("Rows with no value (%)")
    ax.set_title("Missing values in the merged sensor table, per mission")
    ax.legend(loc="upper left")
    S.note(fig, "All 360 flights, weighted by rows. IMU / barometer dropouts (5-10% of "
                "rows) are kept as gaps (aligned_all = 0) and dropped before any running "
                "sum; an incomplete row lacks at least one sensor.")  # fmt: skip
    return S.save(fig, "M3_missing_values",
                  "Share of rows with a missing value per sensor and mission.",
                  "data/processed/module03_v2/base/*.parquet", ME)  # fmt: skip


def m3_correlation():
    meta = {"flight_id", "mission_type", "time_usec", "attack_label", "after_gap", "t_s",
            "offset_m", "spoof_truth", "phase", "seconds_since_onset", "aligned_all"}  # fmt: skip
    s = pd.read_csv(SPLITS)
    parts = []
    for fid in s.loc[s["split"] == "train", "flight_id"]:
        d = pd.read_parquet(V2 / "features" / f"{fid}.parquet")
        parts.append(d[[c for c in d.columns if c not in meta]].iloc[::25])
    d = pd.concat(parts, ignore_index=True)
    c = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    names = [n.replace("res_", "").replace("_", " ") for n in c.columns]
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman correlation")
    ax.set_title(f"Correlation of the {len(names)} per-row features (training flights)")
    S.note(fig, "Every 25th row of the 252 training flights. Blocks of strongly "
                "correlated features are the rolling statistics of the same residual.")  # fmt: skip
    return S.save(fig, "M3_feature_correlation",
                  "How the per-row features correlate: they group by the residual they "
                  "are computed from.", "data/processed/module03_v2/features/*.parquet", ME)  # fmt: skip


def make():
    for f in (m2_flights, m2_durations, m2_rates, m3_example, m3_features, m3_pca,
              m3_splits, m3_missing, m3_correlation):  # fmt: skip
        f()
