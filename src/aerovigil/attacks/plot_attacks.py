"""Step 7: figures that document the attack set (saved to reports/figures/module04)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import get_columns, load_yaml, resolve  # noqa: E402
from aerovigil.attacks.injector import EARTH_R  # noqa: E402
from aerovigil.attacks.profiles import ATTACK_TYPES  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures" / "module04"


def main():
    cfg = load_yaml(ROOT / "configs" / "attacks" / "attacks.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols, scale = get_columns(cfg, pre)
    out_root = resolve(cfg["paths"]["output_dir"])
    man = pd.read_csv(out_root / "attack_manifest.csv")
    main_set = man[man.attack_set == "main"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: one example offset profile per attack type (median-offset example).
    fig, axes = plt.subplots(len(ATTACK_TYPES), 1, figsize=(9, 11), sharex=False)
    for ax, typ in zip(axes, ATTACK_TYPES):
        sub = main_set[main_set.attack_type == typ].sort_values("max_offset_m")
        m = sub.iloc[len(sub) // 2]
        g = pd.read_parquet(
            out_root / "attacked" / m["file"],
            columns=["time_from_attack_start_s", "attack_label", "spoof_offset_m"],
        )
        a = g[g.attack_label == 1]
        ax.plot(a["time_from_attack_start_s"], a["spoof_offset_m"], lw=1.3)
        ax.set_title(
            f"{typ}  ({m['velocity_mode']}, max {m['max_offset_m']:.1f} m)", fontsize=10
        )
        ax.set_ylabel("offset (m)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time since attack start (s)")
    fig.suptitle("Synthetic spoofing offset profiles (one example per type)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "M4_fig1_offset_profiles.png", dpi=150)
    plt.close(fig)

    # Figure 2: true vs spoofed path for a linear-drift flight.
    m = (
        main_set[main_set.attack_type == "linear_drift"]
        .sort_values("max_offset_m")
        .iloc[-1]
    )
    clean_dir = resolve(cfg["paths"]["clean_flights_dir"])
    rep = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    src = pd.read_parquet(
        clean_dir / rep.set_index("flight_id").loc[m["source_flight_id"], "file"]
    )
    att = pd.read_parquet(out_root / "attacked" / m["file"])

    def local(df):
        lat0, lon0 = np.radians(src[cols["lat"]].iloc[0]), np.radians(
            src[cols["lon"]].iloc[0]
        )
        y = (np.radians(df[cols["lat"]]) - lat0) * EARTH_R
        x = (np.radians(df[cols["lon"]]) - lon0) * EARTH_R * np.cos(lat0)
        return x, y

    fig, ax = plt.subplots(figsize=(7, 6))
    x, y = local(src)
    ax.plot(x, y, color="tab:blue", lw=1.5, label="true GPS path")
    x, y = local(att)
    on = att[cols["label"]] == 1
    ax.plot(x[on], y[on], color="tab:red", lw=1.5, label="spoofed GPS path")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.set_title(
        f"Linear drift {m['drift_rate_mps']:.2f} m/s, {m['velocity_mode']} velocity"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "M4_fig2_true_vs_spoofed_path.png", dpi=150)
    plt.close(fig)

    # Figure 3: parameter coverage of the main set.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    data = [
        main_set.loc[main_set.attack_type == t, "max_offset_m"] for t in ATTACK_TYPES
    ]
    axes[0].boxplot(data, tick_labels=[t.replace("_", "\n") for t in ATTACK_TYPES])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("peak spoofed offset (m)")
    axes[0].set_title("Peak offset by attack type")
    axes[1].hist(main_set["duration_s"], bins=25, color="tab:green")
    axes[1].set_xlabel("attack duration (s)")
    axes[1].set_title("Attack duration")
    axes[2].hist(
        main_set["start_s"] / main_set["flight_duration_s"], bins=20, color="tab:orange"
    )
    axes[2].set_xlabel("attack start (fraction of flight)")
    axes[2].set_title("Attack start")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "M4_fig3_parameter_coverage.png", dpi=150)
    plt.close(fig)

    # Figure 4: sweep - peak offset grows with drift rate.
    sw = man[man.attack_set == "sweep"]
    if len(sw):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(sw["drift_rate_mps"], sw["max_offset_m"], s=18, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("drift rate (m/s)")
        ax.set_ylabel("peak offset until end of flight (m)")
        ax.set_title("Drift-rate sweep (test flights)")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "M4_fig4_sweep.png", dpi=150)
        plt.close(fig)

    for f in sorted(FIG_DIR.glob("M4_fig*.png")):
        print("Saved:", f)


if __name__ == "__main__":
    main()
