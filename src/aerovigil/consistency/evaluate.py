r"""AEROVIGIL Module 09: calibrate the consistency detector and evaluate it.

Calibration uses the same flights Module 07 v2 set aside for it: every second normal
training flight, about 1.6 h of flight that no model and no threshold in this project
has been fitted on. The budget is the same 1 alarm per flight-hour that Modules 06, 07
and 08 used. So the numbers this writes sit directly beside theirs.

Evaluation covers the test split only: the original flights for false alarms and for
the dataset's own attacks, the Module 04 main set for per-family detection in both
velocity modes, and the drift sweep for the frontier that Module 06 reported and
Module 08 showed to be an artefact of the attacker's switch-on.

Writes consistency_calibration.json, alarms.csv, summary_by_attack.csv,
       false_alarms.csv and sweep_frontier.csv in data/processed/module09

Usage:
    python src\aerovigil\consistency\evaluate.py
    python src\aerovigil\consistency\evaluate.py --limit 6
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.consistency.detector import (
    BUDGET_PER_HOUR,
    CHANNELS,
    REFRACTORY_S,
    calibrate,
    channel_values,
    fit_standardiser,
    flight_verdict,
    per_channel_scores,
    score_rows,
)
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.lstm.build_sequences_v2 import (
    BASE_REPORT,
    MANIFEST,
    ROOT,
    SPLITS,
    roles_for,
)

M03_BASE = ROOT / "data" / "processed" / "module03_v2" / "base"
M04_BASE = ROOT / "data" / "processed" / "module04_v2" / "base"
OUT_DIR = ROOT / "data" / "processed" / "module09"
MODULE06 = {0.05: 0.05, 0.10: 0.05, 0.25: 0.57, 0.50: 1.00,
            1.00: 1.00, 2.00: 1.00, 5.00: 1.00}  # fmt: skip


def prepared(path):
    """Features, time, truth and the detector's channels for one flight on disk."""
    g = pd.read_parquet(path)
    f = build_flight_v2(g)
    return {
        "values": channel_values(f),
        "t": f["t_s"].to_numpy(float),
        "label": g["attack_label"].to_numpy(float),
        "offset_m": g["offset_m"].to_numpy(float),
    }


def jobs(limit=None):
    """Calibration flights and evaluation flights, as lists of metadata rows."""
    rep = pd.read_csv(BASE_REPORT).sort_values("flight_id")
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    role = roles_for(rep, splits)
    calib, test = [], []
    for _, r in rep.iterrows():
        fid = r["flight_id"]
        row = {
            "path": M03_BASE / r["file"], "flight_id": fid, "set": "orig",
            "mission_type": r["mission"], "velocity_mode": "na",
            "attack_type": "original" if r["attacked"] else "none",
            "drift_rate_mps": np.nan,
        }  # fmt: skip
        if role[fid] == "calib" and not r["attacked"]:
            calib.append(row)
        elif role[fid] == "test":
            test.append(row)
    man = pd.read_csv(MANIFEST)
    man = man[man["split"] == "test"]
    for _, r in man.iterrows():
        test.append({
            "path": M04_BASE / r["file"], "flight_id": r["attacked_flight_id"],
            "set": r["attack_set"], "mission_type": r["mission_type"],
            "attack_type": r["attack_type"], "velocity_mode": r["velocity_mode"],
            "drift_rate_mps": float(r.get("drift_rate_mps", np.nan)),
        })  # fmt: skip
    if limit:
        calib = calib[:limit]
        by = {}
        for row in test:
            by.setdefault((row["set"], row["attack_type"]), []).append(row)
        test = [r for rows in by.values() for r in rows[:limit]]
    return calib, test


def run(limit=None, budget=BUDGET_PER_HOUR, output_dir=None):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    calib_rows, test_rows = jobs(limit)
    print("=" * 74)
    print("AEROVIGIL - MODULE 09 CONSISTENCY DETECTOR (course and velocity channels)")
    print("=" * 74)
    print(f"Channels: {', '.join(CHANNELS)}")
    print(
        f"Calibration flights {len(calib_rows)}   evaluation flights {len(test_rows)}"
        f"   budget {budget:g} alarm/h   refractory {REFRACTORY_S:g} s"
    )
    started = time.time()
    prep = [prepared(r["path"]) for r in calib_rows]
    mu, sd = fit_standardiser(np.concatenate([p["values"] for p in prep]))
    print(
        f"\nStandardiser centre {np.round(mu, 5).tolist()}  scale {np.round(sd, 5).tolist()}"
    )
    clean = []
    for p in prep:
        s, v = score_rows(p["values"], mu, sd)
        clean.append((s, p["t"], v))
    thr, hours, allowed, used = calibrate(clean, budget, REFRACTORY_S)
    print(
        f"Threshold {thr:.2f} from {hours:.2f} clean flight-hours "
        f"(at most {allowed} alarms, used {used})   [{time.time() - started:.0f} s]"
    )
    rows = []
    for n, r in enumerate(test_rows, start=1):
        p = prepared(r.pop("path"))
        s, v = score_rows(p["values"], mu, sd)
        verdict = flight_verdict(s, p["t"], v, p, thr, REFRACTORY_S)
        chans = per_channel_scores(p["values"], mu, sd)
        on = p["label"] == 1
        peaks = {
            f"peak_{name.replace('res_', '').replace('_roll_mean', '')}": (
                float(np.nanmax(z[on]))
                if on.any() and np.isfinite(z[on]).any()
                else np.nan
            )
            for name, z in chans.items()
        }
        rows.append({**r, **verdict, **peaks})
        if n % 100 == 0 or n == len(test_rows):
            print(f"  scored {n}/{len(test_rows)}  ({time.time() - started:.0f} s)")
    alarms = pd.DataFrame(rows)
    alarms.to_csv(out / "alarms.csv", index=False)
    (out / "consistency_calibration.json").write_text(json.dumps({
        "channels": list(CHANNELS), "mu": mu.tolist(), "sd": sd.tolist(),
        "threshold": thr, "calibration_hours": hours, "allowed_alarms": allowed,
        "alarms_used": used, "budget_per_hour": budget,
        "refractory_s": REFRACTORY_S,
    }, indent=2))  # fmt: skip
    report(alarms, out, time.time() - started)
    return alarms, thr


def report(alarms, out, elapsed):
    pd.set_option("display.width", 220)
    att = alarms[alarms["attacked"] & alarms["set"].isin(["orig", "syn"])]
    summary = (
        att.groupby(["set", "attack_type", "velocity_mode"])
        .agg(n=("flight_id", "count"), detection_rate=("detected", "mean"),
             median_delay_s=("delay_s", "median"),
             median_displacement_m=("displacement_m", "median"),
             peak_course=("peak_course_heading", "median"),
             peak_velocity=("peak_horizontal_velocity", "median"))
        .reset_index()
    )  # fmt: skip
    summary.to_csv(out / "summary_by_attack.csv", index=False)
    print("\nDETECTION BY ATTACK (test flights)")
    print(summary.round(3).to_string(index=False))
    normal = alarms[(alarms["set"] == "orig") & (~alarms["attacked"])]
    fa = pd.DataFrame([{
        "alarms": int(normal["false_alarms"].sum()),
        "hours": float(normal["monitored_hours"].sum()),
    }])  # fmt: skip
    fa["alarms_per_hour"] = fa["alarms"] / fa["hours"]
    fa.to_csv(out / "false_alarms.csv", index=False)
    print("\nFALSE ALARMS on the normal test flights")
    print(fa.round(3).to_string(index=False))
    sweep = alarms[(alarms["set"] == "sweep") & alarms["attacked"]]
    frontier = (
        sweep.groupby("drift_rate_mps")
        .agg(n=("flight_id", "count"), detection_rate=("detected", "mean"),
             median_delay_s=("delay_s", "median"),
             median_displacement_m=("displacement_m", "median"),
             peak_course=("peak_course_heading", "median"))
        .reset_index()
    )  # fmt: skip
    if len(frontier):
        frontier["module06_detection_rate"] = frontier["drift_rate_mps"].map(MODULE06)
        frontier.to_csv(out / "sweep_frontier.csv", index=False)
        print("\nSTEALTH FRONTIER: the consistency detector against Module 06's filter")
        print(frontier.round(3).to_string(index=False))
    print(f"\nElapsed {elapsed:.0f} s   saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=float, default=BUDGET_PER_HOUR)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.limit, a.budget, a.output_dir)


if __name__ == "__main__":
    main()
