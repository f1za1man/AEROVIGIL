"""Step 6: calibrate the detectors on normal data and evaluate them on test flights.

Thresholds come from one shared false-alarm budget (flight-hours of NORMAL
train+validation time). Attacks are used only to pick the CUSUM reference k
(validation flights) and to report results (test flights).

Usage:
    python src\\aerovigil\\kalman\\evaluate_detectors.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import load_split_map  # noqa: E402
from aerovigil.kalman.common import load_yaml, out_dir, resolve  # noqa: E402
from aerovigil.kalman.detectors import (
    W_COLS,
    attack_window,
    calibrate_threshold,  # noqa: E402
    cusum_alarm_counts,
    cusum_alarms,
    fit_standardiser,
    flight_metrics,
    nis_alarm_counts,
    nis_alarms,
)


def load_set(folder, meta):
    """Innovation tables joined with their metadata row (dict per flight)."""
    items = []
    for fid, m in meta.items():
        p = folder / m["file"]
        if p.exists():
            items.append((fid, pd.read_parquet(p), m))
    return items


def main():
    cfg = load_yaml(ROOT / "configs" / "kalman" / "kalman.yaml")
    dcfg, fcfg = cfg["detector"], cfg["filter"]
    base = out_dir(cfg)
    rate = fcfg["rate_hz"]

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))

    meta_orig = {
        str(r["flight_id"]): {
            "file": r["file"],
            "set": "orig",
            "split": split_map[str(r["flight_id"])],
            "mission_type": r["mission_type"],
            "attack_type": "original" if r["attack_rows"] > 0 else "none",
            "velocity_mode": "na",
            "rate": np.nan,
        }
        for _, r in report.iterrows()
    }

    def man_meta(sub, name):
        return {
            r["attacked_flight_id"]: {
                "file": r["file"],
                "set": name,
                "split": r["split"],
                "mission_type": r["mission_type"],
                "attack_type": r["attack_type"],
                "velocity_mode": r["velocity_mode"],
                "rate": r["drift_rate_mps"] if name == "sweep" else np.nan,
            }
            for _, r in sub.iterrows()
        }

    sets = {
        "orig": load_set(base / "innovations" / "orig", meta_orig),
        "syn": load_set(
            base / "innovations" / "syn", man_meta(man[man.attack_set == "main"], "syn")
        ),
        "sweep": load_set(
            base / "innovations" / "sweep",
            man_meta(man[man.attack_set == "sweep"], "sweep"),
        ),
    }

    print("=" * 65)
    print("AEROVIGIL - DETECTOR CALIBRATION AND EVALUATION")
    print("=" * 65)
    print("Flights with innovations:", {k: len(v) for k, v in sets.items()})

    # ---------------- calibration data: NORMAL time of train + validation flights ------------
    parts_w, parts_nis, parts_valid, parts_t, offset = [], [], [], [], 0.0
    for fid, df, m in sets["orig"]:
        if m["split"] not in dcfg["calibration_splits"]:
            continue
        win = attack_window(df)
        t = df["t_s"].to_numpy()
        ok = df["valid"].to_numpy() & (df["label"].to_numpy() == 0)
        if win is not None:
            ok &= t < win[0]
        parts_w.append(df[W_COLS].to_numpy(float))
        parts_nis.append(df["nis_h"].to_numpy(float))
        parts_valid.append(ok)
        parts_t.append(t + offset)
        offset += float(t.max()) + 1000.0
    w_cal, nis_cal = np.vstack(parts_w), np.concatenate(parts_nis)
    valid_cal, t_cal = np.concatenate(parts_valid), np.concatenate(parts_t)
    hours = float(valid_cal.sum() / rate / 3600.0)
    allowed = dcfg["false_alarm_budget_per_hour"] * hours
    mu, sd = fit_standardiser(w_cal[valid_cal])
    z_cal = (w_cal - mu) / sd
    print(
        f"\nCalibration: {hours:.2f} normal flight-hours (train+validation) | budget "
        f"{dcfg['false_alarm_budget_per_hour']:g}/h -> at most {allowed:.1f} alarms"
    )
    print(
        f"Standardiser centre {np.round(mu, 3).tolist()} scale {np.round(sd, 3).tolist()}"
    )

    h_grid = np.geomspace(1.0, 2000.0, 90)
    cusum_cal = {}
    for k in dcfg["cusum_k_grid"]:
        counts = cusum_alarm_counts(
            z_cal, valid_cal, t_cal, k, h_grid, dcfg["refractory_s"]
        )
        cusum_cal[k] = calibrate_threshold(counts, h_grid, allowed)
        print(
            f"  CUSUM k={k:<4g} -> h = {cusum_cal[k]:7.2f}  "
            f"(alarms on calibration data: {int(counts[np.searchsorted(h_grid, cusum_cal[k])])})"
        )
    c_grid = np.unique(
        np.append(
            np.quantile(nis_cal[valid_cal], np.linspace(0.90, 0.99999, 120)),
            nis_cal[valid_cal].max() * 1.01,
        )
    )
    c_counts = nis_alarm_counts(nis_cal, valid_cal, t_cal, c_grid, dcfg["refractory_s"])
    nis_c = calibrate_threshold(c_counts, c_grid, allowed)
    print(
        f"  NIS test        -> threshold = {nis_c:7.2f}  (chi-square(4) 99.9% point is 18.47)"
    )

    def run_detectors(df, k):
        z = (df[W_COLS].to_numpy(float) - mu) / sd
        t, v = df["t_s"].to_numpy(), df["valid"].to_numpy()
        return {
            "cusum": cusum_alarms(z, v, t, k, cusum_cal[k], dcfg["refractory_s"]),
            "nis_test": nis_alarms(
                df["nis_h"].to_numpy(float), v, t, nis_c, dcfg["refractory_s"]
            ),
        }

    # ---------------- choose k on validation attacks ----------------
    tune_flights = [
        (f, d, m)
        for s in ("orig", "syn")
        for f, d, m in sets[s]
        if m["split"] == dcfg["tuning_split"] and attack_window(d) is not None
    ]
    choice = []
    for k in dcfg["cusum_k_grid"]:
        res = [
            flight_metrics(d, run_detectors(d, k)["cusum"], rate)
            for _, d, _ in tune_flights
        ]
        det = np.array([r["detected"] for r in res])
        delay = np.array([r["delay_s"] for r in res])
        choice.append(
            {
                "k": k,
                "h": cusum_cal[k],
                "detection_rate": float(det.mean()),
                "median_delay_s": float(np.nanmedian(delay)) if det.any() else np.inf,
            }
        )
    choice_df = pd.DataFrame(choice)
    best = choice_df.sort_values(
        ["detection_rate", "median_delay_s"], ascending=[False, True]
    ).iloc[0]
    k_star = float(best["k"])
    print(f"\nCUSUM reference k chosen on {len(tune_flights)} validation attacks:")
    print(choice_df.round(3).to_string(index=False))
    print(f"  -> k = {k_star:g}, h = {cusum_cal[k_star]:.2f}")

    # ---------------- evaluate on the test split ----------------
    rows = []
    for s in ("orig", "syn", "sweep"):
        for fid, df, m in sets[s]:
            if m["split"] != dcfg["evaluation_split"]:
                continue
            alarms = run_detectors(df, k_star)
            for det_name, idx in alarms.items():
                r = flight_metrics(df, idx, rate)
                rows.append(
                    {
                        "flight_id": fid,
                        "set": s,
                        "split": m["split"],
                        "mission_type": m["mission_type"],
                        "attack_type": m["attack_type"],
                        "velocity_mode": m["velocity_mode"],
                        "drift_rate_mps": m["rate"],
                        "detector": det_name,
                        **r,
                    }
                )
    res = pd.DataFrame(rows)
    res.to_csv(base / "alarms.csv", index=False)

    att = res[res["attacked"] & res["set"].isin(["orig", "syn"])]
    summary = (
        att.groupby(["set", "attack_type", "velocity_mode", "detector"])
        .agg(
            n=("flight_id", "count"),
            detection_rate=("detected", "mean"),
            median_delay_s=("delay_s", "median"),
            median_displacement_m=("displacement_m", "median"),
        )
        .reset_index()
    )
    summary.to_csv(base / "summary_by_attack.csv", index=False)

    fa = (
        res[res["set"] == "orig"]
        .groupby("detector")
        .agg(alarms=("false_alarms", "sum"), hours=("normal_hours", "sum"))
        .reset_index()
    )
    fa["false_alarms_per_hour"] = fa["alarms"] / fa["hours"]
    fa.to_csv(base / "false_alarms.csv", index=False)

    sw = res[(res["set"] == "sweep") & res["attacked"]]
    frontier = (
        sw.groupby(["detector", "drift_rate_mps"])
        .agg(
            n=("flight_id", "count"),
            detection_rate=("detected", "mean"),
            median_displacement_m=("displacement_m", "median"),
            q75_displacement_m=("displacement_m", lambda x: x.quantile(0.75)),
            median_delay_s=("delay_s", "median"),
        )
        .reset_index()
    )
    frontier.to_csv(base / "sweep_frontier.csv", index=False)
    json.dump(
        {
            "k": k_star,
            "h": cusum_cal[k_star],
            "nis_threshold": nis_c,
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "calibration_hours": hours,
            "budget_per_hour": dcfg["false_alarm_budget_per_hour"],
        },
        (base / "detector_calibration.json").open("w"),
        indent=2,
    )

    pd.set_option("display.width", 200)
    print("\n" + "-" * 65)
    print(f"TEST FLIGHTS: {res.groupby('set')['flight_id'].nunique().to_dict()}")
    print(
        "\nFALSE ALARMS on normal time of original test flights (budget "
        f"{dcfg['false_alarm_budget_per_hour']:g}/h)"
    )
    print(fa.round(3).to_string(index=False))
    print("\nDETECTION BY ATTACK (test flights)")
    print(summary.round(3).to_string(index=False))
    print(
        "\nPRELIMINARY STEALTH FRONTIER (sweep, consistent-velocity attacker, test flights)"
    )
    print(frontier.round(2).to_string(index=False))
    print(f"\nSaved to: {base}")
    print("=" * 65)


if __name__ == "__main__":
    main()
