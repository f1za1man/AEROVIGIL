"""Step 7: innovation plot, drift-absorption plot, detection and frontier figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.kalman.common import load_yaml, out_dir  # noqa: E402
from aerovigil.kalman.detectors import W_COLS, attack_window  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures" / "module06"
COLOURS = {"cusum": "tab:red", "nis_test": "tab:blue"}
LABELS = {"cusum": "CUSUM", "nis_test": "chi-square (NIS) test"}
TYPE_SHORT = {"abrupt_jump": "jump", "constant_offset": "offset", "linear_drift": "drift",
              "exponential_ramp": "exp. ramp", "low_amplitude_wander": "wander"}


def cusum_path(z, valid, t, k, h, refractory_s):
    sp = np.zeros(z.shape[1]); sn = np.zeros(z.shape[1])
    stat = np.zeros(len(z)); alarms, skip = [], -np.inf
    for i in range(len(z)):
        if not valid[i] or t[i] < skip:
            sp[:] = 0.0; sn[:] = 0.0
            continue
        sp = np.maximum(0.0, sp + z[i] - k); sn = np.maximum(0.0, sn - z[i] - k)
        stat[i] = max(sp.max(), sn.max())
        if stat[i] > h:
            alarms.append(i); sp[:] = 0.0; sn[:] = 0.0; skip = t[i] + refractory_s
    return stat, alarms


def pick(res, set_name, rate=None, attack_type=None, mode=None):
    q = res[(res["set"] == set_name) & res["attacked"] & (res["detector"] == "cusum")]
    if rate is not None:
        q = q[np.isclose(q["drift_rate_mps"], rate)]
    if attack_type:
        q = q[(q["attack_type"] == attack_type) & (q["velocity_mode"] == mode)]
    return q.sort_values("flight_id").iloc[0] if len(q) else None


def main():
    cfg = load_yaml(ROOT / "configs" / "kalman" / "kalman.yaml")
    base = out_dir(cfg)
    cal = json.load((base / "detector_calibration.json").open())
    res = pd.read_csv(base / "alarms.csv")
    summary = pd.read_csv(base / "summary_by_attack.csv")
    frontier = pd.read_csv(base / "sweep_frontier.csv")
    rate_hz, refr = cfg["filter"]["rate_hz"], cfg["detector"]["refractory_s"]
    mu, sd = np.array(cal["mu"]), np.array(cal["sd"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def load(row):
        sub = "sweep" if row["set"] == "sweep" else "syn"
        man = pd.read_csv(ROOT / cfg["paths"]["attack_manifest"])
        file = man.loc[man["attacked_flight_id"] == row["flight_id"], "file"].iloc[0]
        return pd.read_parquet(base / "innovations" / sub / file)

    # ---- Figure 1: innovation plot (CUSUM sees what the single-step test does not) ----
    rates = sorted(res.loc[res["set"] == "sweep", "drift_rate_mps"].dropna().unique())
    mid = rates[np.argmin(np.abs(np.array(rates) - 0.5))]
    row = pick(res, "sweep", rate=mid)
    df = load(row)
    t, v = df["t_s"].to_numpy(), df["valid"].to_numpy()
    z = (df[W_COLS].to_numpy(float) - mu) / sd
    stat, alarms = cusum_path(z, v, t, cal["k"], cal["h"], refr)
    win = attack_window(df)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for j, c in enumerate(W_COLS):
        axes[0].plot(t, df[c], lw=0.7, label=c.replace("w_", "").replace("_", " "))
    axes[0].set_ylabel("whitened innovation"); axes[0].legend(ncol=4, fontsize=8, loc="upper left")
    axes[1].plot(t, df["nis_h"], lw=0.7, color="tab:blue")
    axes[1].axhline(cal["nis_threshold"], color="k", ls="--", lw=1, label="NIS alarm threshold")
    axes[1].set_yscale("log"); axes[1].set_ylabel("NIS (horizontal)"); axes[1].legend(fontsize=8)
    axes[2].plot(t, stat, lw=0.9, color="tab:red", label="CUSUM statistic")
    axes[2].axhline(cal["h"], color="k", ls="--", lw=1, label="CUSUM threshold")
    for a in alarms[:1]:
        axes[2].axvline(t[a], color="tab:red", ls=":", lw=1.5, label="first alarm")
    axes[2].set_ylabel("CUSUM"); axes[2].set_xlabel("time (s)"); axes[2].legend(fontsize=8)
    for ax in axes:
        if win:
            ax.axvspan(win[0], win[1], color="orange", alpha=0.18)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Innovation plot: {mid:g} m/s drift (consistent velocity), shaded = attack", y=0.995)
    fig.tight_layout(); fig.savefig(FIG_DIR / "M6_fig1_innovation_plot.png", dpi=150); plt.close(fig)

    # ---- Figure 2: the filter absorbs slow drift into its own state ----
    lo, hi = rates[0], rates[np.argmin(np.abs(np.array(rates) - 2.0))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, rate in zip(axes, (lo, hi)):
        r = pick(res, "sweep", rate=rate)
        d = load(r)
        true_e, true_n = d["gps_e"] - d["offset_e"], d["gps_n"] - d["offset_n"]
        err = np.hypot(d["est_e"] - true_e, d["est_n"] - true_n)
        ax.plot(d["t_s"], d["offset_m"], color="tab:red", lw=1.6, label="spoofed offset (truth)")
        ax.plot(d["t_s"], err, color="tab:green", lw=1.2, label="filter position error")
        if r["detected"]:
            ax.axvline(r["first_alarm_s"], color="k", ls=":", label="CUSUM alarm")
        w = attack_window(d)
        if w:
            ax.axvspan(w[0], w[1], color="orange", alpha=0.15)
        ax.set_title(f"{rate:g} m/s drift", fontsize=10); ax.set_xlabel("time (s)")
        ax.set_ylabel("metres"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("The filter absorbs the spoofed position in both cases; only fast drift is caught, "
                 "by its large innovations at the start", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "M6_fig2_drift_absorption.png", dpi=150); plt.close(fig)

    # ---- Figure 3: detection rate by attack type and velocity mode ----
    syn = summary[summary["set"] == "syn"]
    groups = [(t_, m_) for t_ in TYPE_SHORT for m_ in ("consistent", "position_only")]
    fig, ax = plt.subplots(figsize=(12, 4.5)); w = 0.38
    for k, det in enumerate(("cusum", "nis_test")):
        vals, ns = [], []
        for t_, m_ in groups:
            s = syn[(syn.attack_type == t_) & (syn.velocity_mode == m_) & (syn.detector == det)]
            vals.append(s["detection_rate"].iloc[0] if len(s) else np.nan)
            ns.append(int(s["n"].iloc[0]) if len(s) else 0)
        x = np.arange(len(groups))
        ax.bar(x + (k - 0.5) * w, vals, w, color=COLOURS[det], label=LABELS[det])
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{TYPE_SHORT[t_]}\n{'C' if m_ == 'consistent' else 'P'}\n(n={int(syn[(syn.attack_type == t_) & (syn.velocity_mode == m_)]['n'].max() or 0)})"
                        for t_, m_ in groups], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("detection rate (test flights)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9); ax.set_title("Detection by attack type   C = consistent velocity, P = position only")
    fig.tight_layout(); fig.savefig(FIG_DIR / "M6_fig3_detection_by_attack.png", dpi=150); plt.close(fig)

    # ---- Figure 4: preliminary Stealth Frontier ----
    sw = res[(res["set"] == "sweep") & res["attacked"] & (res["detector"] == "cusum")]
    ref = sw.assign(max_possible=sw["drift_rate_mps"] * (sw["attack_end_s"] - sw["attack_start_s"])
                    ).groupby("drift_rate_mps")["max_possible"].median()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ref.index, ref.values, "k--", lw=1, label="no detector (drift to end of flight)")
    for det in ("cusum", "nis_test"):
        f = frontier[frontier.detector == det]
        ax.plot(f["drift_rate_mps"], f["median_displacement_m"].clip(lower=0.05), "o-",
                color=COLOURS[det], label=LABELS[det])
        ax.fill_between(f["drift_rate_mps"], f["median_displacement_m"].clip(lower=0.05),
                        f["q75_displacement_m"].clip(lower=0.05), color=COLOURS[det], alpha=0.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("attacker drift rate r (m/s)"); ax.set_ylabel("displacement before the alarm D(r) (m)")
    ax.set_title("Preliminary Stealth Frontier (consistent-velocity attacker)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "M6_fig4_stealth_frontier.png", dpi=150); plt.close(fig)

    for f in sorted(FIG_DIR.glob("M6_fig*.png")):
        print("Saved:", f)


if __name__ == "__main__":
    main()